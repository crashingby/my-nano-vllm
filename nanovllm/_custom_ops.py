# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""提供与 vLLM 同名的 CUDA 算子入口，当前只保留非量化 KV cache 写入。"""

import torch


def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str = "auto",
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None:
    """按物理 slot 原地更新 FlashAttention 使用的分页 K/V cache。

    保留 vLLM 的参数名和调用顺序，但实际调用本项目的 Triton kernel，
    不依赖 vLLM 编译出来的 ``torch.ops._C_cache_ops``。目前支持 FP16/BF16，
    输入和缓存必须同精度；``k_scale``、``v_scale`` 在非量化路径中不参与计算。

    Args:
        key: ``[>=num_tokens, num_kv_heads, head_size]`` 的本轮 K。
        value: 与 key 同形状、同精度的本轮 V。
        key_cache: ``[num_blocks, block_size, num_kv_heads, head_size]`` 的 K 视图。
        value_cache: 与 key_cache 同形状的 V 视图；允许不同的物理 stride。
        slot_mapping: ``[num_tokens]`` 的物理 slot，负值表示不写入的 padding。
        kv_cache_dtype: ``auto``、``float16`` 或 ``bfloat16``，显式名称须匹配缓存。
        k_scale: 为兼容上游接口保留的 K 量化缩放因子。
        v_scale: 为兼容上游接口保留的 V 量化缩放因子。

    Side Effects:
        在当前 CUDA stream 上原地修改缓存；不会复制整个缓存或同步 GPU。
        有效 slot 的范围和唯一性由调用方保证，未映射的位置保持原值。
    """
    supported_cache_dtypes = {
        "auto": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }  # 明确拒绝 FP8 等尚未实现的缓存格式，避免静默丢失量化语义。
    if kv_cache_dtype not in supported_cache_dtypes:
        raise NotImplementedError(
            f"reshape_and_cache_flash does not support kv_cache_dtype={kv_cache_dtype!r}."
        )

    # 同精度直接存储才与当前后端一致；kernel 不承担隐式精度转换。
    if key.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Expected FP16/BF16 key, got {key.dtype}.")
    if not (key.dtype == value.dtype == key_cache.dtype == value_cache.dtype):
        raise ValueError("key, value, key_cache and value_cache must have the same dtype.")
    requested_dtype = supported_cache_dtypes[kv_cache_dtype]  # auto 使用缓存自身的精度。
    if requested_dtype is not None and key_cache.dtype != requested_dtype:
        raise ValueError(
            f"kv_cache_dtype={kv_cache_dtype!r} does not match cache dtype={key_cache.dtype}."
        )

    # 导入入口时不初始化 Triton；只有真正执行缓存写入才加载 kernel 模块。
    from nanovllm.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash,
    )

    triton_reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )
