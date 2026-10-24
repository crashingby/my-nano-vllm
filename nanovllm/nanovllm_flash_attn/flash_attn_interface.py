# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2023, Tri Dao.
"""移植 vLLM 的 FA2 前向参数流，直接调用现有 flash-attn CUDA 扩展。

导入时仅加载动态库，不查询 GPU 或创建 CUDA 上下文；本模块捕获 ImportError
并记录可用性，由包入口在扩展不可用时抛出 ImportError，沿用 vLLM 的流程。
这里使用上游 ``flash_attn_2_cuda`` 的 21 参数 ABI，
不复制 vLLM 自行构建的 C++ 扩展，也不经过上游公开函数的额外输出分配。
当前仅提供无量化 FP16/BF16 推理，没有 autograd、FA3/FA4 或调度元数据。
"""

from __future__ import annotations

import torch  # 必须先加载 PyTorch 的共享库，再动态链接 flash-attn 扩展。

try:
    import flash_attn_2_cuda as _fa2_extension

    FA2_AVAILABLE: bool = True  # 表示扩展可导入，不代表当前进程存在可用 GPU。
    FA2_UNAVAILABLE_REASON: str | None = None
except ImportError as import_error:
    _fa2_extension = None  # 保留失败状态，由包入口统一检查并拒绝导入。
    FA2_AVAILABLE = False
    FA2_UNAVAILABLE_REASON = str(import_error)  # 保留动态链接失败的具体原因。

DEFAULT_FA_VERSION: int = 2  # 当前没有移植 FA3/FA4 的扩展和 Python 调度接口。


def _is_fa2_supported() -> tuple[bool, str | None]:
    """同时检查扩展和当前 GPU；查询发生在显式调用时，而非包导入时。

    Returns:
        (is_supported, unsupported_reason):
            is_supported: 扩展与当前 GPU 都满足 FA2 执行条件时为 True。
            unsupported_reason: 不可执行时的原因字符串；支持时为 None。
    """
    if not FA2_AVAILABLE:
        return False, f"FA2 is unavailable due to: {FA2_UNAVAILABLE_REASON}"
    from nanovllm.platforms import current_platform

    # FA2 CUDA kernel 要求 Ampere 或更新架构；无可用设备也返回不支持。
    if not current_platform.has_device_capability(80):
        return False, "FA2 requires an available CUDA device with compute capability >= 8.0"
    return True, None


def is_fa_version_supported(fa_version: int) -> bool:
    """返回指定 FA 版本是否可执行；FA3/FA4 明确返回 False。

    Returns:
        is_supported: 指定 FA 版本当前可执行时为 True；未移植版本返回 False。
    """
    return fa_version_unsupported_reason(fa_version) is None


def fa_version_unsupported_reason(fa_version: int) -> str | None:
    """给出扩展、GPU 能力或未移植版本的原因，支持时返回 None。

    Returns:
        unsupported_reason: 指定 FA 版本无法执行的原因字符串；支持时为 None。
    """
    if fa_version == 2:
        return _is_fa2_supported()[1]
    if fa_version in (3, 4):
        return f"FA{fa_version} is not implemented; nano-vllm currently supports only FA2"
    raise ValueError(f"Unsupported FA version: {fa_version}")


def maybe_contiguous(tensor: torch.Tensor | None) -> torch.Tensor | None:
    """仅在 head 最内层维度不连续时复制，保留 paged cache 的真实外层 stride。

    Returns:
        tensor: 满足 head 最内层连续约束的 tensor；无需复制或输入为 None 时原样返回。
    """
    if tensor is not None and tensor.stride(-1) != 1:
        return tensor.contiguous()
    return tensor


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    q_v: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: list[int] | tuple[int, int] | None = None,
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    block_table: torch.Tensor | None = None,
    return_softmax_lse: bool = False,
    out: torch.Tensor | None = None,
    scheduler_metadata: torch.Tensor | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    num_splits: int = 0,
    fa_version: int = DEFAULT_FA_VERSION,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """沿用 vLLM 参数流执行 FA2，支持 packed/paged KV、MQA/GQA 和原地输出。

    Args:
        q: ``[total_q, num_heads, head_size]`` 的 CUDA FP16/BF16 Query。
        k: packed 时为 ``[total_k, num_kv_heads, head_size]``；paged 时为
            ``[num_blocks, page_size, num_kv_heads, head_size]``，允许外层非连续。
        v: 与 k 的形状、精度和设备相同的 Value。
        cu_seqlens_q: ``int32[batch_size + 1]``，packed Query 的请求边界。
        cu_seqlens_k: 非 paged 的 packed KV 请求边界，与 seqused_k 二选一。
        seqused_k: ``int32[batch_size]``，每请求当前可读取的 KV 长度；paged
            缓存直接传 seq_lens，不需要从它重新计算 KV 前缀和。
        max_seqlen_q: 本批次最大的 Query 长度，使用已有 CPU 统计值。
        max_seqlen_k: 本批次最大的 KV 长度，使用已有 CPU 统计值。
        block_table: ``int32[batch_size, max_num_blocks]``，逻辑页到物理页的映射。
        softmax_scale: QK 分数缩放系数，缺省为 ``head_size ** -0.5``。
        causal: 采用右下角对齐的 causal mask；分块 prefill/decode 应传 True。
        out: 可选预分配输出。直接交给 CUDA 扩展写入，返回同一对象。
        return_softmax_lse: 为 True 时额外返回 ``float32[num_heads, total_q]`` LSE。
        fa_version: 当前只支持 2；不会把请求的 FA3/FA4 静默降级为 FA2。
        num_splits: 当前只能为 0；已安装上游扩展不能显式指定 KV 分割数量。
        window_size: 只接受 None 或 ``(-1, -1)``，当前没有 sliding window。
        q_v: FA3 兼容参数，当前必须为 None。
        scheduler_metadata: FA3 调度元数据，当前必须为 None。
        q_descale: Query 量化反缩放参数，当前必须为 None。
        k_descale: Key 量化反缩放参数，当前必须为 None。
        v_descale: Value 量化反缩放参数，当前必须为 None。
        dropout_p: 当前只提供推理前向，必须为 0.0。
        softcap: logits 软截断参数，当前必须为 0.0。
        alibi_slopes: ALiBi 位置偏置参数，当前必须为 None。
        deterministic: backward 确定性兼容参数，当前必须为 False。
        return_attn_probs: 当前不导出完整注意力概率，必须为 False。

    Returns:
        默认返回单个 tensor；启用额外返回时返回二元组：
            output: 默认返回的 Attention 输出 tensor。
            softmax_lse: return_softmax_lse=True 时额外返回的 log-sum-exp，此时整体为 (output, softmax_lse)。

    Side Effects:
        out 非空时直接写入该 tensor；绝不更新 KV cache。缓存更新必须由后端
        在此调用前完成。没有 backward，调用方必须保证只用于推理。

    Notes:
        上游 GQA 单 token 优化可能复制多 KV head 的 out，或改变多请求 LSE
        的头次序。出现这些情况时仅将 kernel 的 max_seqlen_q 上界提升到 2；
        cu_seqlens_q 的真实长度保持原样，Attention 结果的语义不变。
    """
    # 与 vLLM 相同：先检查参数关系，普通 tensor 校验交给扩展的 C++ 入口。
    assert cu_seqlens_k is not None or seqused_k is not None, (
        "cu_seqlens_k or seqused_k must be provided"
    )
    assert cu_seqlens_k is None or seqused_k is None, (
        "cu_seqlens_k and seqused_k cannot be provided at the same time"
    )
    assert block_table is None or seqused_k is not None, (
        "seqused_k must be provided if block_table is provided"
    )

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    real_window_size: tuple[int, int]
    if window_size is None:
        real_window_size = (-1, -1)  # 默认采用完整上下文。
    else:
        assert len(window_size) == 2  # 沿用 vLLM 对左右窗口参数的约定。
        real_window_size = (window_size[0], window_size[1])
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]

    # 与 vLLM 一样创建占位；是否使用它由下面的 packed/paged 路径决定。
    dummy_cu_seqlens_k = torch.empty_like(cu_seqlens_q)

    if fa_version != 2:
        reason = fa_version_unsupported_reason(fa_version)  # 未知版本保持 vLLM 的 ValueError。
        raise NotImplementedError(reason)
    if num_splits != 0:
        raise NotImplementedError(
            "FA2 upstream varlen_fwd cannot specify num_splits; use num_splits=0"
        )
    # 不接受没有实现的非默认参数，避免接口对齐后产生默默失效的模型语义。
    unsupported = {
        "q_v": q_v is not None,
        "scheduler_metadata": scheduler_metadata is not None,
        "q_descale": q_descale is not None,
        "k_descale": k_descale is not None,
        "v_descale": v_descale is not None,
        "dropout_p": dropout_p != 0.0,
        "softcap": softcap != 0.0,
        "alibi_slopes": alibi_slopes is not None,
        "deterministic": deterministic,
        "return_attn_probs": return_attn_probs,
        "window_size": real_window_size != (-1, -1),
    }
    for name, enabled in unsupported.items():
        if enabled:
            raise NotImplementedError(f"nano-vllm FA2 does not support non-default {name}")
    # 上游 FA2 的 packed KV 用 cu_k 定位物理起点，即使传入 seqused_k 也仍读取它。
    # 这是当前 ABI 不支持的组合；不可将未初始化的占位作为 packed KV 起点传入。
    if block_table is None and cu_seqlens_k is None:
        raise NotImplementedError("Non-paged KV requires cu_seqlens_k with upstream FA2")
    kernel_max_seqlen_q = max_seqlen_q  # 这里只是 kernel launch 上界，真实长度仍来自 cu_q。
    if max_seqlen_q == 1 and q.shape[1] > k.shape[-2]:
        batch_size = cu_seqlens_q.numel() - 1
        # 上游 GQA decode 的 transpose/reshape(out) 在这些布局下可能复制，
        # 导致原 buffer 未写入；单请求或单 KV head 的连续 out 可以继续优化。
        output_may_copy = out is not None and (
            (batch_size > 1 and k.shape[-2] > 1) or not out.is_contiguous()
        )
        # 该优化的 LSE 末尾直接 reshape，不能恢复多请求的 head/group 顺序。
        lse_may_reorder = return_softmax_lse and batch_size > 1
        if output_may_copy or lse_may_reorder:
            kernel_max_seqlen_q = 2  # 提升 launch 上界，实际 Query 长度保持不变。
    # seqused_k 路径由 kernel 直接读取每请求长度；ABI 仍要求 cu_k 有合法形状。
    # 与 vLLM 一样仅创建占位，不能用 cumsum 替换真正的 seqused_k 参数。
    result = _fa2_extension.varlen_fwd(
        q,
        k,
        v,
        out,  # 原始 output buffer 直接进入 CUDA kernel，无返回结果 copy_。
        cu_seqlens_q,
        dummy_cu_seqlens_k if cu_seqlens_k is None else cu_seqlens_k,
        seqused_k,
        None,  # 当前没有 leftpad_k。
        block_table,
        alibi_slopes,  # 当前只允许默认 None；保留参考源码的参数传递顺序。
        kernel_max_seqlen_q,
        max_seqlen_k,
        dropout_p,  # 当前只允许默认 0.0。
        softmax_scale,
        False,  # 不调用 zero_tensors，以免改变调用方的输出初始化策略。
        causal,
        real_window_size[0],
        real_window_size[1],  # 当前只允许完整上下文窗口。
        softcap,  # 当前只允许默认 0.0。
        return_softmax_lse and dropout_p > 0,  # 与 vLLM 一致，LSE 单独返回。
        None,  # 无 dropout，无需随机数生成器。
    )
    # result 为上游的 [out, lse, dropout_mask, rng_state]；本接口暴露 output 和 softmax_lse。
    output = result[0] if out is None else out  # 显式保证预分配 buffer 的 Python 对象身份。
    softmax_lse = result[1]
    return (output, softmax_lse) if return_softmax_lse else output
