"""模型执行设备配置；配置构造阶段不探测或初始化 CUDA。"""

from dataclasses import dataclass, field
from typing import Literal

import torch
from pydantic import SkipValidation

# 沿用 vLLM 的设备类型别名，只保留当前支持的自动选择与 CUDA 后端。
Device = Literal["auto", "cuda"]


@dataclass(slots=True)
class DeviceConfig:
    """保存模型执行设备，保留 vLLM 的 device/device_type 两个成员。

    当前只支持 CUDA，因此 auto 直接选择 CUDA，不在无 GPU 时退回 CPU。
    配置可在创建 Worker 前构造；硬件可用性由 Worker.init_device() 检查。

    Attributes:
        device: 沿用 vLLM 的类型声明，支持 auto、cuda 或 torch.device，构造后统一为
            torch.device。无 index 时由 Worker 的 local_rank 绑定设备；
            显式 index 通过 torch.device 指定，且必须与 local_rank 一致。
            None 保留在类型声明中，但当前 CUDA 路径不支持空执行设备。
        device_type: 从 device 派生的后端类型，当前只能为 cuda，不含设备编号。
    """

    # 不写死 cuda:0，使相同配置可交给不同 TP Worker。
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
        # MVP 无跨平台探测体系，auto 选择唯一支持的 CUDA 后端。
        device = "cuda" if self.device == "auto" else self.device
        if not isinstance(device, (str, torch.device)):
            raise ValueError(f"Invalid execution device: {device!r}")
        try:
            # 这里只解析名称，torch.device 不会初始化 CUDA runtime。
            resolved_device = torch.device(device)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid execution device: {device!r}") from exc
        # LoadConfig.device 可以指定 CPU 暂存加载，但不表示支持 CPU 推理。
        if resolved_device.type != "cuda":
            raise ValueError(
                f"Only CUDA execution is supported, got {resolved_device}."
            )
        self.device = resolved_device
        self.device_type = resolved_device.type
