# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""保留 vLLM 的平台与设备能力接口，目前只定义 CUDA 所需的公共约定。"""

from __future__ import annotations

import enum
import platform
from typing import Any, NamedTuple

import torch

from nanovllm.logger import init_logger

logger = init_logger(__name__)


def in_wsl() -> bool:
    """按 vLLM 的 uname 检测流程判断宿主是否处于 WSL，不查询 CUDA。"""
    return "microsoft" in " ".join(platform.uname()).lower()


class PlatformEnum(enum.Enum):
    """标识实现的平台；UNSPECIFIED 仅用于基类，不代表 CPU 回退。"""

    CUDA = enum.auto()  # 当前唯一可用于模型执行的平台。
    UNSPECIFIED = enum.auto()  # 为尚未指定实现的 Platform 基类保留标记。


class DeviceCapability(NamedTuple):
    """保存 CUDA compute capability，例如 SM 8.9 表示为 (8, 9)。"""

    major: int  # 架构主版本；10.x、11.x 等新架构仍可表示。
    minor: int  # 同一架构主版本中的次版本。

    def __lt__(self, other: Any) -> bool:
        """按主版本、次版本依次比较，保持 vLLM 的 NamedTuple 语义。"""
        if not isinstance(other, DeviceCapability):
            return NotImplemented  # 交给另一操作数处理普通 tuple 等类型。
        return (self.major, self.minor) < (other.major, other.minor)

    def __le__(self, other: Any) -> bool:
        """判断当前能力是否不高于另一个设备能力。"""
        if not isinstance(other, DeviceCapability):
            return NotImplemented
        return (self.major, self.minor) <= (other.major, other.minor)

    def __eq__(self, other: Any) -> bool:
        """比较完整能力版本，而非仅比较架构主版本。"""
        if not isinstance(other, DeviceCapability):
            return NotImplemented
        return (self.major, self.minor) == (other.major, other.minor)

    def __ge__(self, other: Any) -> bool:
        """判断当前能力是否满足另一个设备能力的最低要求。"""
        if not isinstance(other, DeviceCapability):
            return NotImplemented
        return (self.major, self.minor) >= (other.major, other.minor)

    def __gt__(self, other: Any) -> bool:
        """判断当前能力是否高于另一个设备能力。"""
        if not isinstance(other, DeviceCapability):
            return NotImplemented
        return (self.major, self.minor) > (other.major, other.minor)

    def __hash__(self) -> int:
        """与等值 tuple 使用相同哈希，允许设备能力用于字典或集合。"""
        return hash((self.major, self.minor))

    def as_version_str(self) -> str:
        """返回日志与架构检查使用的 major.minor 格式。"""
        return f"{self.major}.{self.minor}"

    def to_int(self) -> int:
        """返回 major * 10 + minor，例如 8.9 为 89、10.0 为 100。"""
        assert 0 <= self.minor < 10  # 整数格式为次版本保留一位十进制数。
        return self.major * 10 + self.minor


class Platform:
    """统一硬件身份、能力查询和显式设备操作，不在构造时访问硬件。"""

    _enum: PlatformEnum = PlatformEnum.UNSPECIFIED  # 子类替换为实际平台标记。
    device_name: str = "unspecified"  # 平台名称，不是某一 GPU 的型号。
    device_type: str = ""  # 子类提供 torch.device 使用的设备类型。
    dispatch_key: str = "CPU"  # 沿用 vLLM 基类默认值；CUDA 子类覆盖为 CUDA。
    dist_backend: str = ""  # 子类声明 torch.distributed 的通信后端。
    device_control_env_var: str = ""  # 子类声明控制可见设备的环境变量。
    _global_graph_pool: Any | None = None  # 首次请求时创建 PyTorch Graph 分配池的共享标识。

    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        """返回通用浮点精度顺序；具体平台根据硬件能力覆盖。"""
        return [torch.bfloat16, torch.float16, torch.float32]

    def is_cuda(self) -> bool:
        """判断平台身份，不调用 torch.cuda.is_available 或初始化设备。"""
        return self._enum == PlatformEnum.CUDA

    def is_cuda_alike(self) -> bool:
        """判断是否采用 CUDA 设备接口；当前只支持 CUDA 平台。"""
        return self.is_cuda()

    def is_unspecified(self) -> bool:
        """判断是否仍使用未指定硬件实现的 Platform 基类。"""
        return self._enum == PlatformEnum.UNSPECIFIED

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        """查询逻辑设备能力；没有实现或没有可用设备时返回 None。"""
        return None

    @classmethod
    def has_device_capability(
        cls,
        capability: tuple[int, int] | int,
        device_id: int = 0,
    ) -> bool:
        """判断设备能力是否不低于要求，例如 SM 8.9 满足 80 或 (8, 0)。"""
        current_capability = cls.get_device_capability(device_id=device_id)
        if current_capability is None:
            return False  # 缺少硬件信息不能视为满足 kernel 的最低架构要求。
        if isinstance(capability, tuple):
            return current_capability >= capability
        return current_capability.to_int() >= capability

    @classmethod
    def is_device_capability(
        cls,
        capability: tuple[int, int] | int,
        device_id: int = 0,
    ) -> bool:
        """判断设备能力是否精确匹配；SM 8.9 不匹配 80，但匹配 (8, 9)。"""
        current_capability = cls.get_device_capability(device_id=device_id)
        if current_capability is None:
            return False
        if isinstance(capability, tuple):
            return current_capability == capability
        return current_capability.to_int() == capability

    @classmethod
    def is_device_capability_family(cls, capability: int, device_id: int = 0) -> bool:
        """判断架构主版本族，例如 SM 10.3 与 100、103 都属于 10.x。

        capability 使用与 to_int 相同的完整整数格式，10.x 应传 100，不能传 10。
        """
        current_capability = cls.get_device_capability(device_id=device_id)
        if current_capability is None:
            return False
        return current_capability.to_int() // 10 == capability // 10

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """设置当前线程的执行设备；实际设备初始化由平台子类负责。"""
        raise NotImplementedError

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        """设置所有本平台设备的随机种子，影响之后的随机运算。"""
        raise NotImplementedError

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """获取指定逻辑设备的型号；实际查询可能初始化设备运行时。"""
        raise NotImplementedError

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """获取指定逻辑设备的总显存，单位为字节。"""
        raise NotImplementedError

    @classmethod
    def inference_mode(cls) -> torch.inference_mode:
        """创建推理上下文；退出后恢复调用方原有的 autograd 状态。"""
        return torch.inference_mode(mode=True)

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        """检查 host staging buffer 是否可锁页，沿用 vLLM 的 WSL 限制。

        这里只检查宿主环境；具体设备平台还可检查设备运行时是否可用。
        """
        if in_wsl():
            logger.warning("WSL detected; using pin_memory=False for host buffers.")
            return False  # 当前采用与参考源码相同的保守策略。
        return True

    @classmethod
    def get_current_memory_usage(cls, device: torch.types.Device | None = None) -> float:
        """获取当前设备已分配的显存字节数；具体统计流程由平台实现。"""
        raise NotImplementedError

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype) -> None:
        """在模型分配前校验运行精度；不支持时由平台实现抛出异常。"""
        raise NotImplementedError

    @classmethod
    def num_compute_units(cls, device_id: int = 0) -> int:
        """查询计算单元数量；CUDA 平台中对应 SM 数量。"""
        raise NotImplementedError

    @classmethod
    def get_infinity_values(cls, dtype: torch.dtype) -> tuple[float, float]:
        """返回给定精度使用的负、正无穷值；CUDA 无需特殊有限值替代。"""
        return float("-inf"), float("inf")

    @classmethod
    def can_update_inplace(cls) -> bool:
        """允许执行器原地更新持久输入 buffer 和 KV cache。"""
        return True

    @classmethod
    def supports_fp8(cls) -> bool:
        """查询硬件是否支持 FP8；不代表项目已实现 FP8 模型或缓存。"""
        return False

    @classmethod
    def is_fp8_fnuz(cls) -> bool:
        """查询硬件偏好的 FP8 编码；CUDA 使用 OCP 编码而非 FNUZ。"""
        return False

    @classmethod
    def fp8_dtype(cls) -> torch.dtype:
        """返回 OCP FP8 dtype 标识；此接口不执行量化或分配 FP8 tensor。"""
        return torch.float8_e4m3fn

    @classmethod
    def use_custom_allreduce(cls) -> bool:
        """是否启用自定义 all-reduce；当前只使用 torch.distributed NCCL。"""
        return False

    @classmethod
    def opaque_attention_op(cls) -> bool:
        """是否把 Attention 注册为 opaque custom op；当前仍是 Python 调用。"""
        return False

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        """是否启用混合类型 KV cache；当前 Runner 只支持完整 Attention。"""
        return False

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        """是否提供执行器的 static graph wrapper；当前尚未接入该编译接口。"""
        return False

    @classmethod
    def support_deep_gemm(cls) -> bool:
        """是否可使用 DeepGEMM 实现；当前没有引入其依赖和 kernel。"""
        return False

    @classmethod
    def use_custom_op_collectives(cls) -> bool:
        """是否使用自定义通信算子；当前使用原生 torch.distributed 接口。"""
        return False

    @classmethod
    def is_arch_support_pdl(cls) -> bool:
        """查询架构是否支持 PDL；具体 GPU 平台负责架构判断。"""
        return False

    def __getattr__(self, key: str) -> Any:
        """按 vLLM 流程转发未显式定义的属性到 torch 的设备模块。

        例如 CUDA 平台的 synchronize、Stream、Event、current_device 可以直接
        复用 torch.cuda 实现。属性访问不执行返回的函数，调用时才有相应副作用。
        普通未知属性按参考源码告警并返回 None；未知 dunder 属性抛 AttributeError，
        避免 pickle 等协议把缺失的特殊方法当作真实实现。
        """
        if key.startswith("__") and key.endswith("__"):
            raise AttributeError(key)
        device = getattr(torch, self.device_type, None)  # 仅获取模块，不初始化设备。
        if device is not None:
            attr = getattr(device, key, None)
            if attr is not None:
                return attr
        logger.warning(
            "Current platform %s does not have '%s' attribute.", self.device_type, key
        )
        return None

    def get_global_graph_pool(self) -> Any:
        """首次调用时创建 PyTorch graph pool 标识，为本进程同一平台复用。

        该标识用于提示 PyTorch 缓存分配器让多次捕获共享 Graph 私有分配池，
        不是 CUDA Runtime 的 cudaMemPool_t，也不预分配指定大小的显存。
        本接口不会捕获或回放模型；首次调用可能初始化 CUDA。
        """
        cls = self.__class__
        if cls._global_graph_pool is None:
            cls._global_graph_pool = self.graph_pool_handle()
        return cls._global_graph_pool
