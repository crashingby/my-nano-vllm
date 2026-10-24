# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""沿用 vLLM 的 FlashAttention 导入和能力查询入口，当前只支持 CUDA FA2。

计算接口由 nanovllm_flash_attn 适配已安装的 CUDA 扩展，缓存写入接口由
_custom_ops 提供。这里负责报告能力，不准备批次 metadata，也不初始化设备。
"""

from __future__ import annotations

from nanovllm._custom_ops import reshape_and_cache_flash
from nanovllm.nanovllm_flash_attn import (
    fa_version_unsupported_reason,
    flash_attn_varlen_func,
    is_fa_version_supported,
)
from nanovllm.platforms import current_platform


def get_flash_attn_version(
    requires_alibi: bool = False,
    head_size: int | None = None,
    head_size_v: int | None = None,
    has_sinks: bool = False,
) -> int | None:
    """按当前平台、扩展可用性和 head 结构选择支持的 FA 版本。

    沿用 vLLM 命名与查询顺序；当前只会选择 FA2，没有 FA3/FA4 的版本覆盖、
    量化或后端回退。导入本模块不会查硬件，显式调用本函数才查询 CUDA 能力。

    Args:
        requires_alibi: 是否要求 ALiBi；当前模型执行路径未支持。
        head_size: Query/Key head 维度；直接调用扩展要求是 8 的倍数且不超过 256。
        head_size_v: Value head 维度；当前必须与 Query/Key 相同。
        has_sinks: 是否要求 attention sinks；当前未支持。

    Returns:
        fa_version: 当前扩展和 GPU 支持时返回 2；无可用 GPU、缺少扩展或要求不支持的
            功能时返回 None。安装了扩展不代表当前机器具备可运行的 CUDA GPU。
    """
    if not current_platform.is_cuda():
        return None
    # 先排除项目未支持的模型特性，避免为已知不兼容的配置查询 CUDA。
    if requires_alibi or has_sinks:
        return None
    if head_size is not None and (
        head_size <= 0 or head_size > 256 or head_size % 8 != 0
    ):
        return None
    if head_size_v is not None and head_size_v != head_size:
        return None
    # 版本支持检查同时确认 .so 已加载及 GPU compute capability >= 8.0。
    if not is_fa_version_supported(2):
        return None
    return 2


def is_flash_attn_varlen_func_available() -> bool:
    """判断当前 CUDA 平台是否已加载 FA2 扩展，不查询或初始化 GPU。

    此接口只描述函数是否可导入。是否能在当前设备执行，应另外通过
    get_flash_attn_version 或 is_fa_version_supported 查询。

    Returns:
        is_available: CUDA 平台的 FA2 varlen 函数已成功导入时为 True，不探测 GPU 能力。
    """
    # 与 vLLM CUDA 分支一致：模块已经成功导入，说明包入口的扩展检查已通过。
    return current_platform.is_cuda()


def flash_attn_supports_quant_query_input() -> bool:
    """报告当前移植接口不支持量化 Query，只接受 FP16/BF16。

    Returns:
        supports_quant_query_input: 是否支持量化 Query；当前移植接口始终返回 False。
    """
    return False


def flash_attn_supports_sinks() -> bool:
    """报告当前 FA2 接口不支持 attention sinks。

    Returns:
        supports_sinks: 是否支持 attention sinks；当前接口始终返回 False。
    """
    return False


def flash_attn_supports_mla() -> bool:
    """报告当前普通 decoder Attention 路径不支持 MLA。

    Returns:
        supports_mla: 是否支持 MLA；当前普通 decoder Attention 路径返回 False。
    """
    return False


__all__ = [
    "fa_version_unsupported_reason",
    "flash_attn_supports_mla",
    "flash_attn_supports_quant_query_input",
    "flash_attn_supports_sinks",
    "flash_attn_varlen_func",
    "get_flash_attn_version",
    "is_fa_version_supported",
    "is_flash_attn_varlen_func_available",
    "reshape_and_cache_flash",
]
