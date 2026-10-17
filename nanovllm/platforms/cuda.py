# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""通过 PyTorch 实现 CUDA 平台操作，显式调用之前不查询或初始化 GPU。"""

from __future__ import annotations

import torch

from nanovllm.logger import init_logger
from nanovllm.platforms.interface import DeviceCapability, Platform, PlatformEnum

logger = init_logger(__name__)


class CudaPlatformBase(Platform):
    """实现 CUDA 公共操作，具体设备能力、型号与拓扑由 CudaPlatform 提供。

    保留 vLLM 的方法名称和调用顺序。硬件能力查询与执行器功能开关分开：
    例如支持 FP8 的 GPU 仍不能在本项目中运行尚未实现的 FP8 Attention。
    """

    _enum: PlatformEnum = PlatformEnum.CUDA  # 平台身份不依赖当前机器有无可用 GPU。
    device_name: str = "cuda"  # 与 torch.device 的设备类型保持一致。
    device_type: str = "cuda"  # 用于创建模型参数和推理输入 tensor。
    dispatch_key: str = "CUDA"  # 为后续 CUDA 算子注册提供统一的 dispatch key。
    dist_backend: str = "nccl"  # CUDA 设备使用 NCCL 进行 tensor parallel 通信。
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"  # 查询使用进程可见的逻辑设备。

    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        """按架构返回计算精度列表；第一项可作为模型自动精度的优先选择。"""
        if self.has_device_capability(80):
            return [torch.bfloat16, torch.float16, torch.float32]  # Ampere 起支持 BF16。
        if self.has_device_capability(60):
            return [torch.float16, torch.float32]  # Pascal、Volta、Turing 不具备 BF16。
        return [torch.float32]  # 未知能力也不能声明设备支持半精度执行。

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """切换当前 CUDA 设备并执行一次小分配，立即完成该设备的初始化。

        沿用 vLLM 的 eager 设置流程；错误的设备或不可用的 CUDA 由 PyTorch 报错。
        """
        torch.cuda.set_device(device)  # 设置此线程的当前设备，后续分配遵循该设置。
        _ = torch.zeros(1, device=device)  # 触发实际设备初始化，避免设置仍处于延迟状态。

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        """更新所有可见 CUDA 设备的随机种子；PyTorch 负责尚未初始化的设备。"""
        torch.cuda.manual_seed_all(seed)

    @classmethod
    def get_cuda_runtime_major(cls) -> int:
        """返回 torch.version.cuda 的主版本；CPU build 或未知版本返回 0。"""
        major = (torch.version.cuda or "0").split(".", 1)[0]  # 无需初始化 CUDA 即可读取。
        return int(major) if major.isdigit() else 0

    @classmethod
    def device_count(cls) -> int:
        """返回当前进程可见的逻辑 GPU 数量，尊重 CUDA_VISIBLE_DEVICES。

        当前直接采用 PyTorch 查询，不复制 vLLM 的私有 stateless/NVML 辅助函数；
        应在 Worker 初始化阶段读取，不在初始化 CUDA 后重新修改可见设备集合。
        """
        return torch.cuda.device_count()

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        """CUDA 可用且不处于 WSL 时，允许为 CPU staging tensor 分配锁页内存。"""
        # 无 CUDA 时不能将 CPU 接口测试误判为可分配 CUDA pinned memory。
        return super().is_pin_memory_available() and torch.cuda.is_available()

    @classmethod
    def get_current_memory_usage(cls, device: torch.types.Device | None = None) -> float:
        """按 vLLM 流程查询当前 PyTorch 分配量，单位为字节。

        先释放 allocator 的空闲缓存，再把指定设备的峰值重置到当前分配量，
        最后读取 max_memory_allocated。因此它会改变 allocator 缓存和峰值统计，
        不可用于读取先前 forward 的峰值，也不等同于总显存减去空闲显存。
        """
        torch.cuda.empty_cache()  # 与参考流程一致：释放空闲缓存，不释放仍活跃的 tensor。
        torch.cuda.reset_peak_memory_stats(device)  # 旧峰值被覆盖，调用方必须知晓此副作用。
        return torch.cuda.max_memory_allocated(device)

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype) -> None:
        """在 Worker 选定设备后检查模型精度，不支持的 BF16 明确报错。

        沿用 vLLM 的 SM 8.0 门槛；这里查询当前逻辑设备，确保异构多卡时校验
        本 Worker 所在的 GPU。模型 dtype 范围仍由 ModelConfig 负责解析和限制。
        """
        if dtype != torch.bfloat16:
            return  # 参考实现只在此接口附加 BF16 的架构限制。
        device_id = torch.cuda.current_device()
        if cls.has_device_capability(80, device_id=device_id):
            return
        capability = cls.get_device_capability(device_id)
        gpu_name = cls.get_device_name(device_id)
        compute_str = (
            "does not have a compute capability"
            if capability is None
            else f"has compute capability {capability.as_version_str()}"
        )
        raise ValueError(
            "Bfloat16 is only supported on GPUs with compute capability of at least 8.0. "
            f"Your {gpu_name} GPU {compute_str}. Use dtype='float16' instead."
        )

    @classmethod
    def supports_fp8(cls) -> bool:
        """按 vLLM 查询硬件 FP8 能力：SM 8.9 及以上；不启用 FP8 执行路径。"""
        return cls.has_device_capability(89)

    @classmethod
    def is_arch_support_pdl(cls) -> bool:
        """查询当前设备是否为支持 PDL 的 SM 9.x 或更新架构。

        PDL 是 Programmatic Dependent Launch 的硬件能力；这个查询不会启用
        FA2 中不存在的 PDL 参数。无可用设备或运行时未就绪时保守返回 False。
        """
        try:
            device_id = torch.cuda.current_device()
            capability = cls.get_device_capability(device_id)
        except (RuntimeError, AssertionError):
            return False  # 仅处理设备不可用，不吞掉类型或实现错误。
        return capability is not None and capability.major >= 9

    @classmethod
    def is_integrated_gpu(cls, device_id: int = 0) -> bool:
        """查询 GPU 是否共享系统内存；旧 PyTorch 未暴露该属性时返回 False。

        新版参考源码直接读取 is_integrated；当前 PyTorch 版本可能没有这个
        属性。缺少属性时只能报告未确认，不能据此启用 UMA 专用显存策略。
        """
        device_props = torch.cuda.get_device_properties(device_id)
        is_integrated = getattr(device_props, "is_integrated", None)
        if is_integrated is None:
            logger.debug(
                "PyTorch does not expose is_integrated for CUDA device %s.", device_id
            )
            return False
        return bool(is_integrated)

    @classmethod
    def num_compute_units(cls, device_id: int = 0) -> int:
        """返回指定逻辑 GPU 的 SM 数量，可供 kernel 并行度配置使用。"""
        return torch.cuda.get_device_properties(device_id).multi_processor_count

    @classmethod
    def insert_blocks_to_device(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        """按第一维 block 下标拷贝到设备缓存，原地修改 dst_cache 指定的块。

        两组下标按位置一一对应，例如源 [2, 0] 写到目标 [1, 3]。调用方保证
        对应块的 shape/dtype 一致、下标有效且目标下标不重复。这里只提供复制
        原语，不分配缓存、不更新 block table，也不提供 KV offload 调度策略。
        """
        _src_cache = src_cache[src_block_indices]  # 先取出源块，避免同存储复制覆盖源数据。
        dst_cache[dst_block_indices] = _src_cache.to(dst_cache.device)

    @classmethod
    def swap_out_blocks_to_host(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        """按对应下标将设备块拷回已有 CPU 缓存，原地修改目标块。

        调用方提供 CPU dst_cache 和有效、唯一的目标下标；形状与 dtype 必须
        匹配。沿用参考源码的 .cpu() 流程，不承诺异步 D2H 拷贝。
        """
        _src_cache = src_cache[src_block_indices]
        dst_cache[dst_block_indices] = _src_cache.cpu()

    @classmethod
    def is_fully_connected(cls, device_ids: list[int]) -> bool:
        """判断设备集合是否通过 NVLink 全互联；具体平台决定查询方式。"""
        raise NotImplementedError


class CudaPlatform(CudaPlatformBase):
    """采用 vLLM NonNvmlCudaPlatform 的 PyTorch 查询路径，不依赖 NVML。"""

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        """查询指定逻辑 GPU 的能力；没有可用 CUDA 时返回 None。

        本方法仅在显式调用时访问 CUDA。PyTorch 的属性查询可能初始化运行时，
        因此不具备 vLLM NVML 路径保证的 stateless 特性，也不缓存不可用的结果。
        """
        if not torch.cuda.is_available():
            return None  # 允许 CPU 上导入和验证接口，不把平台身份切换为 CPU。
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """返回指定逻辑 GPU 的型号；不可用或无效设备由 PyTorch 报错。"""
        return torch.cuda.get_device_name(device_id)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """查询指定逻辑 GPU 的总显存字节数，使用进程可见的设备编号。"""
        device_props = torch.cuda.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool:
        """沿用 NonNvmlCudaPlatform 策略，未检测 NVLink 时保守返回 False。

        参数是物理 GPU 编号，不是 CUDA_VISIBLE_DEVICES 重排后的逻辑编号。
        当前没有 NVML 拓扑查询；peer access 可用也不代表 NVLink 全互联。
        """
        return False
