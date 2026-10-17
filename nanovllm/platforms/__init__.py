# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""导出 CUDA 平台单例；构造单例不执行硬件检测或 CUDA 初始化。"""

from nanovllm.platforms.cuda import CudaPlatform, CudaPlatformBase
from nanovllm.platforms.interface import DeviceCapability, Platform, PlatformEnum

current_platform: CudaPlatform = CudaPlatform()  # 当前仅支持 CUDA，无需插件检测或动态选择。

__all__ = [
    "CudaPlatform",
    "CudaPlatformBase",
    "DeviceCapability",
    "Platform",
    "PlatformEnum",
    "current_platform",
]
