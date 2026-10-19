"""持久化 SiLU × Mul：一 warp 一行，官方 autotune_jit 搜索行数和线程元素数。"""

from collections.abc import Callable
from functools import lru_cache

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import autotune_jit
from cuda.bindings.driver import CUstream
import torch

from nanovllm.ir.ops.activation import silu_and_mul
from nanovllm.platforms import current_platform

@cute.kernel
def silu_and_mul_kernel(
    x: cute.Tensor,
    output: cute.Tensor,
    num_tokens: cutlass.Int64,
    hidden_size: cutlass.Constexpr[int],
    rows_per_block: cutlass.Constexpr[int],
    elements_per_thread: cutlass.Constexpr[int],
) -> None:
    """由连续输入 [T, 2H] 写入独立输出 [T, H]，中间计算保持 FP32。"""
    thread_id, _, _ = cute.arch.thread_idx()
    block_id, _, _ = cute.arch.block_idx()
    num_blocks, _, _ = cute.arch.grid_dim()
    warp_id = thread_id // 32  # 每个 warp 领取当前行块中的一行。
    lane_id = thread_id % 32  # lane 编号决定本线程在列块中的连续分片。

    layout = cute.make_layout(elements_per_thread)  # 每个线程持有连续的一组元素。
    # 跨过整个网格的行块，64 位行索引保护较大 Tensor 的地址计算。
    row = cutlass.Int64(block_id) * rows_per_block + cutlass.Int64(warp_id)
    while row < num_tokens:
        col = cutlass.Int32(lane_id) * elements_per_thread
        while col < hidden_size:
            gate_offset = row * (2 * hidden_size) + col
            up_offset = gate_offset + hidden_size
            output_offset = row * hidden_size + col
            mask = None
            if cutlass.const_expr(hidden_size % elements_per_thread == 0):
                # 完整分片显式传递整除信息；访存宽度仍受原始指针实际对齐限制。
                gate_offset = cute.assume(gate_offset, divby=elements_per_thread)
                up_offset = cute.assume(up_offset, divby=elements_per_thread)
                output_offset = cute.assume(output_offset, divby=elements_per_thread)
            else:
                # 通道尾部统一使用 mask，无需再复制一份标量计算。
                valid = cute.make_rmem_tensor(layout, cutlass.Boolean)
                for value_id in cutlass.range_constexpr(elements_per_thread):
                    valid[value_id] = col + value_id < hidden_size
                mask = valid.load()
            gate_view = cute.make_tensor(x.iterator + gate_offset, layout)
            up_view = cute.make_tensor(x.iterator + up_offset, layout)
            output_view = cute.make_tensor(output.iterator + output_offset, layout)
            gate = gate_view.load(mask=mask).to(cutlass.Float32)
            up = up_view.load(mask=mask).to(cutlass.Float32)
            # 分母至少为 1，ftz 满足向量近似倒数的要求。
            inv = cute.rcp(1.0 + cute.exp(-gate, fastmath=True), approx=True, ftz=True)
            output_view.store((gate * inv * up).to(output.element_type), mask=mask)
            col += 32 * elements_per_thread
        row += cutlass.Int64(num_blocks) * rows_per_block


@lru_cache(maxsize=None)
def _get_autotuned_silu_and_mul(
    device_index: int,
    dtype: torch.dtype,
    hidden_size: int,
    input_alignment: int,
) -> Callable[..., None]:
    """缓存元数据特化的调优入口；官方装饰器按 token 数缓存最佳配置。"""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "CuTe SiluAndMul must be autotuned before CUDA Graph capture; run eager warmup first"
        )
    elements_per_128_bits = 16 // dtype.itemsize  # 候选元素数直接由 PyTorch dtype 换算。
    max_blocks = current_platform.num_compute_units(device_index) * 4  # 保留四倍 SM 上限。

    @autotune_jit(
        params_dict={
            "rows_per_block": [1, 2, 4, 8],
            "elements_per_thread": [
                elements_per_128_bits // 2, elements_per_128_bits,
                elements_per_128_bits * 2, elements_per_128_bits * 4,
            ],
        },
        update_on_change=["num_tokens"],
        warmup_iterations=5,
        iterations=30,
    )
    @cute.jit
    def launch(
        x: cute.Tensor,
        output: cute.Tensor,
        num_tokens: cutlass.Int64,
        stream: CUstream,
        rows_per_block: cutlass.Constexpr[int] = 4,
        elements_per_thread: cutlass.Constexpr[int] = elements_per_128_bits,
    ) -> None:
        """接收 DLPack 推导类型和布局的 Tensor，按候选配置发射 kernel。"""
        num_blocks = (num_tokens + rows_per_block - 1) // rows_per_block
        if num_blocks > max_blocks:
            num_blocks = cutlass.Int64(max_blocks)
        silu_and_mul_kernel(
            x, output, num_tokens, hidden_size, rows_per_block, elements_per_thread,
        ).launch(
            grid=(cutlass.Int32(num_blocks), 1, 1),
            block=(32 * rows_per_block, 1, 1),
            stream=stream,
        )

    return launch


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


@silu_and_mul.register_impl("cutedsl", supports_args=_supports_silu_and_mul)
def silu_and_mul_cutedsl(x: torch.Tensor) -> torch.Tensor:
    """分配独立的 [..., H] 输出，通过官方调优器选择配置并发射 kernel。

    Args:
        x: 连续 CUDA FP16/BF16/FP32 Tensor，形状为 [..., 2H]，H 为正整数。

    Side Effects:
        首次编译和冷 L2 调优需在 Graph 捕获前预热；后续使用当前 stream。
        调优入口按设备、精度、H、地址对齐特化，输入保持原值。
    """
    hidden_size = x.shape[-1] // 2
    num_tokens = x.numel() // x.shape[-1]
    output = torch.empty((*x.shape[:-1], hidden_size), device=x.device, dtype=x.dtype)
    if num_tokens == 0:
        return output  # 空输入不查询设备或启动网格。
    input_alignment = min(16, x.data_ptr() & -x.data_ptr())  # 兼容带存储偏移的连续输入。
    # 一维视图共享原存储，动态长度使编译结果可复用到不同 token 数。
    x_cute = from_dlpack(x.detach().view(-1), assumed_align=input_alignment).mark_layout_dynamic(0)
    output_cute = from_dlpack(output.view(-1), assumed_align=16).mark_layout_dynamic(0)
    with torch.cuda.device(x.device):
        launcher = _get_autotuned_silu_and_mul(
            x.get_device(), x.dtype, hidden_size, input_alignment,
        )
        key = (num_tokens,)  # 其余特化条件由缓存工厂区分。
        current_stream = torch.cuda.current_stream(x.device)
        if key not in launcher._best_kernel:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "CuTe SiluAndMul must be autotuned before CUDA Graph capture; run eager warmup first"
                )
            default_stream = torch.cuda.default_stream(x.device)  # 官方计时器使用默认 stream。
            default_stream.wait_stream(current_stream)  # 保证调优开始前输入已经生成。
            with torch.cuda.stream(default_stream):
                launcher(x_cute, output_cute, num_tokens, CUstream(default_stream.cuda_stream))
            current_stream.wait_stream(default_stream)  # 衔接装饰器最后一次异步 kernel。
        # 命中缓存后不再调用调优器或修改 stream，保持原有 forward 的执行顺序。
        compiled = launcher._best_kernel[key]
        compiled(x_cute, output_cute, num_tokens, CUstream(current_stream.cuda_stream))
    return output
