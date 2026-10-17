# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""沿用 vLLM 的 FA 包入口，扩展不可用时在导入阶段报错。"""

from nanovllm.nanovllm_flash_attn.flash_attn_interface import (
    FA2_AVAILABLE,
    fa_version_unsupported_reason,
    flash_attn_varlen_func,
    is_fa_version_supported,
)

# 当前仅支持 FA2，保留参考源码的扩展可用性检查，裁剪 FA3 分支。
if not FA2_AVAILABLE:
    raise ImportError(
        "nanovllm.nanovllm_flash_attn requires the CUDA flash attention extension "
        "(flash_attn_2_cuda)."
    )

__all__ = [
    "fa_version_unsupported_reason",
    "flash_attn_varlen_func",
    "is_fa_version_supported",
]
