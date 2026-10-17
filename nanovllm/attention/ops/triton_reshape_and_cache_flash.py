# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""移植 vLLM 的分页缓存写入流程，保留当前 FP16/BF16 CUDA 路径。"""

import torch
import triton
import triton.language as tl


@triton.jit
def _reshape_and_cache_flash_kernel(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    key_stride_token: tl.int64,
    key_stride_head: tl.int64,
    key_stride_dim: tl.int64,
    value_stride_token: tl.int64,
    value_stride_head: tl.int64,
    value_stride_dim: tl.int64,
    key_cache_stride_block: tl.int64,
    key_cache_stride_slot: tl.int64,
    key_cache_stride_head: tl.int64,
    key_cache_stride_dim: tl.int64,
    value_cache_stride_block: tl.int64,
    value_cache_stride_slot: tl.int64,
    value_cache_stride_head: tl.int64,
    value_cache_stride_dim: tl.int64,
    slot_mapping_stride: tl.int64,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    """每个 program 将一个 token 的一段 head/dim 数据写入指定物理页。

    所有 offset 都使用 tensor 的真实 stride，允许从 combined KV cache
    取得 ``unbind(1)`` 视图，也允许 K/V 使用不同的页、head 和 dim stride。
    """
    token_idx = tl.program_id(axis=0)  # 第一维仅覆盖 slot_mapping 的有效输入范围。
    slot_idx = tl.load(slot_mapping_ptr + token_idx * slot_mapping_stride).to(tl.int64)
    if slot_idx < 0:
        return  # padding token 不读取 K/V，也不修改任何缓存位置。

    block_idx = slot_idx // block_size  # slot 的商表示物理页编号。
    block_offset = slot_idx % block_size  # slot 的余数表示页内 token 位置。
    tile_idx = tl.program_id(axis=1)  # 同一 token 的大 head 数据分成多个 tile。
    tile_pos = (tile_idx * TILE_SIZE + tl.arange(0, TILE_SIZE)).to(tl.int64)
    cur_head = tile_pos // head_size  # 展开坐标还原为 KV head 编号。
    cur_dim = tile_pos % head_size  # 保留真实 dim stride，不能假定最后一维连续。
    mask = tile_pos < num_heads * head_size  # 最后一个 tile 可能超出 head 数据范围。

    key_offsets = (
        token_idx * key_stride_token
        + cur_head * key_stride_head
        + cur_dim * key_stride_dim
    )
    value_offsets = (
        token_idx * value_stride_token
        + cur_head * value_stride_head
        + cur_dim * value_stride_dim
    )
    key = tl.load(key_ptr + key_offsets, mask=mask, other=0)  # 仅加载本轮这个 token 的 K。
    value = tl.load(value_ptr + value_offsets, mask=mask, other=0)  # V 允许独立的 stride。

    key_cache_offsets = (
        block_idx * key_cache_stride_block
        + block_offset * key_cache_stride_slot
        + cur_head * key_cache_stride_head
        + cur_dim * key_cache_stride_dim
    )
    value_cache_offsets = (
        block_idx * value_cache_stride_block
        + block_offset * value_cache_stride_slot
        + cur_head * value_cache_stride_head
        + cur_dim * value_cache_stride_dim
    )
    tl.store(key_cache_ptr + key_cache_offsets, key, mask=mask)  # K/V 同精度直接写入。
    tl.store(value_cache_ptr + value_cache_offsets, value, mask=mask)


def triton_reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str = "auto",
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None:
    """校验非量化分页缓存布局，并异步启动原地写入 kernel。

    Args:
        key: ``[>=N, H, D]`` 的 CUDA FP16/BF16 tensor，可以有尾部 padding。
        value: 与 key 同形状和精度；可以有不同的实际 stride。
        key_cache: ``[B, page_size, H, D]`` 的 CUDA 缓存视图。
        value_cache: 与 key_cache 同形状和精度；不要求物理连续。
        slot_mapping: ``[N]`` 的 CUDA int32/int64 物理 slot，负值跳过。
        kv_cache_dtype: 当前支持 ``auto``、``float16`` 和 ``bfloat16``。
        k_scale: 保留上游参数，当前非量化路径不会读取。
        v_scale: 保留上游参数，当前非量化路径不会读取。

    Side Effects:
        在当前 CUDA stream 写入 cache；不做 CPU fallback，不同步或复制整个缓存。
        调用方须保证非负 slot 小于 ``B * page_size``，且同批次没有重复有效 slot。
    """
    supported_cache_dtypes = {
        "auto": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }  # 直接调用此入口时也须拒绝未实现的量化格式。
    if kv_cache_dtype not in supported_cache_dtypes:
        raise NotImplementedError(f"Unsupported kv_cache_dtype={kv_cache_dtype!r}.")
    if key.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Expected FP16/BF16 key, got {key.dtype}.")
    if not (key.dtype == value.dtype == key_cache.dtype == value_cache.dtype):
        raise ValueError("key, value, key_cache and value_cache must have the same dtype.")
    requested_dtype = supported_cache_dtypes[kv_cache_dtype]  # auto 不额外转换缓存精度。
    if requested_dtype is not None and key_cache.dtype != requested_dtype:
        raise ValueError("kv_cache_dtype does not match the actual cache dtype.")

    # 输入/cache 均有明确的维度语义，避免错误布局被误解释成有效物理地址。
    if key.ndim != 3 or value.shape != key.shape:
        raise ValueError("key and value must have the same [num_tokens, num_heads, head_size] shape.")
    if key_cache.ndim != 4 or value_cache.shape != key_cache.shape:
        raise ValueError("Caches must have the same [num_blocks, block_size, num_heads, head_size] shape.")
    if key_cache.shape[2:] != key.shape[1:]:
        raise ValueError("Input and cache num_heads/head_size dimensions must match.")
    if slot_mapping.ndim != 1 or slot_mapping.dtype not in (torch.int32, torch.int64):
        raise ValueError("slot_mapping must be a one-dimensional int32/int64 tensor.")
    num_tokens = slot_mapping.numel()  # 与 vLLM 一致，写入范围由 slot mapping 决定。
    if num_tokens > key.shape[0]:
        raise ValueError("slot_mapping cannot contain more entries than input tokens.")
    num_heads, head_size = key.shape[1:]  # 每个 token 需要写入的 KV head 和通道数。
    block_size = key_cache.shape[1]  # 页大小来自当前缓存布局，而非固定常量。
    if num_heads <= 0 or head_size <= 0 or block_size <= 0:
        raise ValueError("num_heads, head_size and block_size must be positive.")

    tensors = (key, value, key_cache, value_cache, slot_mapping)
    if not all(tensor.is_cuda and tensor.device == key.device for tensor in tensors):
        raise ValueError("Inputs, caches and slot_mapping must be on the same CUDA device.")
    if num_tokens == 0:
        return  # 空批次不启动 kernel；只检查 host 可见的形状/设备信息。

    # 小批次不需要 autotune；tile 限制避免高 head 数产生过大的单 program。
    tile_size = min(512, triton.next_power_of_2(num_heads * head_size))
    grid = (num_tokens, triton.cdiv(num_heads * head_size, tile_size))
    _reshape_and_cache_flash_kernel[grid](
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        *key.stride(),
        *value.stride(),
        *key_cache.stride(),
        *value_cache.stride(),
        slot_mapping.stride(0),
        num_heads=num_heads,
        head_size=head_size,
        block_size=block_size,
        TILE_SIZE=tile_size,
        num_warps=4,
    )
