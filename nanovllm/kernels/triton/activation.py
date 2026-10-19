"""使用固定上限的 program 网格，循环计算 SiLU(gate) × up。

输入最后一维为 [gate, up]，前导维度按连续布局合并成 token 行。每个 program
依次处理若干 M 行块，每个行块内部再循环读取 N 通道块；program 数量最多为
设备 SM 数的四倍，批次增大时通过循环消化额外工作。输出独立分配，输入保持原值。
"""

import torch
import triton
import triton.language as tl

from nanovllm.ir.ops.activation import silu_and_mul
from nanovllm.platforms import current_platform


@triton.autotune(
    configs=[
        # 联合搜索行数、通道数和线程数，覆盖小块低开销与大块减少循环两种取向。
        triton.Config(
            {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n},
            num_warps=num_warps,
        )
        for block_m in (1, 2, 4, 8)
        for block_n in (256, 512, 1024)
        for num_warps in (2, 4, 8)
    ],
    key=["num_tokens", "hidden_size"],
)
@triton.jit
def silu_and_mul_kernel(
    input_ptr,
    output_ptr,
    num_tokens,
    hidden_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """按 grid-stride 分配 M 行块，在每个行块内顺序遍历 hidden 通道。

    input_ptr 指向连续 [num_tokens, 2 * hidden_size] 数据，output_ptr 指向独立的
    连续 [num_tokens, hidden_size] 输出。两种块尺寸均为 2 的幂；每个输出元素
    由唯一 program 写入，行尾和通道尾通过二维 mask 保护。
    """
    pid_m = tl.program_id(0)  # 本 program 的首个 M 行块编号。
    num_programs_m = tl.num_programs(0)  # 固定 worker 数，也是下一轮行块的步长。
    num_blocks_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)  # 全批次需要处理的逻辑 M 块数。
    row_offsets = tl.arange(0, BLOCK_SIZE_M)  # 当前块内的相对行坐标。
    col_offsets = tl.arange(0, BLOCK_SIZE_N)  # 每次读取的相对通道坐标。

    # 例如启动 4 个 program 时，program 0 依次处理 M 块 0、4、8……。
    for block_m in range(pid_m, num_blocks_m, num_programs_m):
        rows = (block_m * BLOCK_SIZE_M + row_offsets).to(tl.int64)
        # 每个 program 内循环遍历当前行块的全部 hidden 通道。
        for block_n in range(tl.cdiv(hidden_size, BLOCK_SIZE_N)):
            cols = block_n * BLOCK_SIZE_N + col_offsets
            mask = (rows[:, None] < num_tokens) & (cols[None, :] < hidden_size)
            # 输入行宽是 2H，gate 位于每行前半部分，up 位于后半部分。
            gate_offsets = rows[:, None] * (2 * hidden_size) + cols[None, :]
            up_offsets = gate_offsets + hidden_size
            gate = tl.load(input_ptr + gate_offsets, mask=mask, other=0).to(tl.float32)
            up = tl.load(input_ptr + up_offsets, mask=mask, other=0).to(tl.float32)

            # 中间激活保持 FP32，留在 program 内，不生成独立的显存中间张量。
            activated = gate * tl.sigmoid(gate)
            result = activated * up
            # 输出行宽是 H，store 时转换为 output Tensor 对应的精度。
            output_offsets = rows[:, None] * hidden_size + cols[None, :]
            tl.store(output_ptr + output_offsets, result, mask=mask)


def _supports_silu_and_mul(x: torch.Tensor) -> bool:
    """只接受连续 CUDA 浮点输入，最后一维必须包含两个非空、等长分片。"""
    return (
        x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.ndim >= 1
        and x.shape[-1] > 0
        and x.shape[-1] % 2 == 0
        and x.is_contiguous()
    )


@silu_and_mul.register_impl("triton", supports_args=_supports_silu_and_mul)
def silu_and_mul_triton(x: torch.Tensor) -> torch.Tensor:
    """分配输出并启动持久化 SiLU × Mul，接受任意连续前导维度。

    Args:
        x: 连续 CUDA FP16/BF16/FP32 Tensor，形状为 [..., 2H]，H 为正整数。
            通过 IrOp 调用时先由 supports_args 检查输入，impl_fn 是已验证入口。

    Returns:
        独立的 [..., H] 输出，device 和 dtype 与 x 相同，输入内容保持原值。

    Side Effects:
        在当前 CUDA stream 启动 kernel，设备 SM 数由平台接口统一缓存。
        首次遇到新的调优 key 时可能触发 JIT、重复执行和计时同步；命中调优
        缓存后正常发射 kernel。空前导维度直接返回空输出。FP16/BF16 在 kernel
        中使用 FP32 中间值，与 PyTorch eager 的逐算子舍入可能存在细小差异。
    """
    hidden_size = x.shape[-1] // 2  # 每个 gate/up 分片的通道数。
    num_tokens = x.numel() // x.shape[-1]  # 将连续前导维度合并，兼容一维与多维输入。
    output = torch.empty(
        (*x.shape[:-1], hidden_size),
        device=x.device,
        dtype=x.dtype,
    )
    if num_tokens == 0:
        return output  # 空输入不查询设备属性，也不启动零尺寸 grid。
    num_sms = current_platform.num_compute_units(x.get_device())  # 复用平台按设备缓存的 SM 数。
    # 与 CuTe 一样将网格上限设为 SM 数的四倍，小批次按实际 M 行块数启动。
    grid = lambda meta: (
        min(
            triton.cdiv(num_tokens, meta["BLOCK_SIZE_M"]),
            num_sms * 4,
        ),
    )
    silu_and_mul_kernel[grid](
        x,
        output,
        num_tokens,
        hidden_size=hidden_size,
    )
    return output
