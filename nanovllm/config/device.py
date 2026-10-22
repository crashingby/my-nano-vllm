"""模型执行设备配置；配置构造阶段不探测或初始化 CUDA。"""

from dataclasses import field
from typing import Literal

import torch
from pydantic import ConfigDict, SkipValidation

from nanovllm.config.utils import config

# 沿用 vLLM 的设备类型别名，只保留当前支持的自动选择与 CUDA 后端。
Device = Literal["auto", "cuda"]


# 设备对象交由 __post_init__ 规范化；Pydantic 仅建立其类型边界。
@config(config=ConfigDict(arbitrary_types_allowed=True), slots=True)
class DeviceConfig:
    """保存模型执行设备，保留 vLLM 的 device/device_type 两个成员。

    当前只支持 CUDA，因此 auto 直接选择 CUDA，不在无 GPU 时退回 CPU。
    配置构造只规范化设备描述；CUDA 可用性、可见设备范围和显式编号是否与
    local_rank 冲突，由 Worker.init_device() 检查。UniProcExecutor 使用显式设备
    编号作为 local_rank；未指定编号时使用 0，对应进程可见的第一张卡。

    Attributes:
        device: 沿用 vLLM 的类型声明，支持 auto、cuda 或 torch.device，构造后统一为
            torch.device。无 index 时由 Worker 的 local_rank 绑定设备；
            显式 index 可以通过 cuda:N 字符串或 torch.device 指定，Worker 会
            检查它是否与 local_rank 一致；编号受 CUDA_VISIBLE_DEVICES 影响。
            None 保留在类型声明中，但当前 CUDA 路径不支持空执行设备。
        device_type: 从 device 派生的后端类型，当前只能为 cuda，不含设备编号。
    """

    # 缺省不指定编号，由 Worker 按 local_rank 绑定当前 CUDA 设备。
    device: SkipValidation[Device | torch.device | None] = "auto"
    # 派生类型不允许单独传入，以免与 device 矛盾。
    device_type: str = field(init=False)

    def __post_init__(self) -> None:
        """解析设备描述并校验当前支持的执行后端。

        Side Effects:
            将 device 规范化并写入 device_type；不分配显存、不切换当前 GPU。

        Raises:
            ValueError: 设备描述非法，或请求了当前未实现的非 CUDA 后端。
        """
        # auto 是当前唯一后端 CUDA 的别名，不探测硬件，也不选择其他执行后端。
        device = "cuda" if self.device == "auto" else self.device
        if not isinstance(device, (str, torch.device)):
            raise ValueError(f"Invalid execution device: {device!r}")
        try:
            # 这里只解析名称，torch.device 不会初始化 CUDA runtime。
            resolved_device = torch.device(device)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid execution device: {device!r}") from exc
        # 执行设备必须为 CUDA；LoadConfig 的加载设备覆盖不改变这个限制。
        if resolved_device.type != "cuda":
            raise ValueError(
                f"Only CUDA execution is supported, got {resolved_device}."
            )
        self.device = resolved_device  # 后续消费者统一取得 torch.device 对象。
        self.device_type = resolved_device.type  # 保存后端名，不包含具体设备编号。
