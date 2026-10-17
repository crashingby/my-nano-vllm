"""GPU 显存快照与 KV cache 预算辅助函数。"""

from __future__ import annotations

import math
import time
import contextlib
import gc
from collections.abc import Generator
from dataclasses import dataclass, field
from types import TracebackType

import torch
import torch.types

from nanovllm.config.cache import CacheConfig
from .mem_constants import GiB_bytes, KiB_bytes, MiB_bytes


def format_kib(b: int) -> str:
    return f"{round(b / KiB_bytes, 2)}"


def format_mib(b: int) -> str:
    return f"{round(b / MiB_bytes, 2)}"


def format_gib(b: int) -> str:
    """将字节数转换为便于日志展示的 GiB 字符串。

    Args:
        num_bytes: 待转换的字节数。

    Returns:
        保留两位小数的 GiB 数值字符串，不额外附加单位。例如 ``"12.34"``。
    """
    return f"{round(b / GiB_bytes, 2)}"


class DeviceMemoryProfiler:
    """统计一段代码执行前后 PyTorch tensor 显存占用的净增量。

    常用于包围模型权重加载：进入时记录基线，退出时记录加载后的占用，
    二者之差近似表示新加载权重的显存大小。这里直接采用 vLLM CUDA platform
    的采样流程，不引入其他设备平台的抽象。

    统计范围是 PyTorch allocator 的 allocated bytes，不包括未使用的缓存、
    NCCL 等非 PyTorch 分配，也不是整张 GPU 的总占用或代码执行期间的峰值。
    如果区间内还释放了原有 tensor，净增量也可能为负数。

    Attributes:
        device: 要读取统计的 CUDA 设备；None 表示当前 CUDA 设备。
            调用方应先设置当前设备，尤其是 empty_cache() 不接受 device 参数。
        initial_memory: 进入 with 时的 tensor 显存占用，单位为字节；进入后可读。
        final_memory: 退出 with 时的 tensor 显存占用，单位为字节；退出后可读。
        consumed_memory: final_memory - initial_memory，单位为字节；退出后可读。
    """

    def __init__(self, device: torch.types.Device | None = None) -> None:
        """保存采样设备，暂不读取显存，也不初始化 CUDA。

        Args:
            device: CUDA 设备编号、字符串、torch.device 或 None。
                实际统计在进入和退出 with 时完成。
        """
        self.device = device  # 两次采样使用同一设备，None 时依赖当前设备。

    def current_memory_usage(self) -> float:
        """按 vLLM 的 CUDA 采样流程读取当前 tensor 显存占用。

        Returns:
            PyTorch allocator 中仍被 tensor 占用的显存字节数。
            保留 vLLM 的 float 返回类型注解，CUDA API 实际返回整数。

        Side Effects:
            执行 Python 垃圾回收、释放 allocator 的空闲缓存，并重置目标设备的
            峰值统计。因此不要在另一个需要保留峰值的 profiling 区间内调用。
        """
        # 先回收不可达对象，避免其持有的 tensor 被误计为仍在使用的显存。
        gc.collect()
        # 将空闲缓存归还 CUDA；不会释放仍被活跃 tensor 使用的显存。
        torch.cuda.empty_cache()
        # 将历史峰值重置为当前占用，不保留上一段执行过程的最高水位。
        torch.cuda.reset_peak_memory_stats(self.device)
        # 重置后立即读取，得到采样边界的占用，而不是 with 区间内的峰值。
        return torch.cuda.max_memory_allocated(self.device)

    def __enter__(self) -> DeviceMemoryProfiler:
        """记录执行前的显存基线，供退出时计算差值。

        Returns:
            当前 profiler 对象，可通过 with ... as profiler 读取统计结果。

        Side Effects:
            创建 initial_memory 字段，并执行一次带清理操作的显存采样。
        """
        self.initial_memory = self.current_memory_usage()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """记录执行后占用并计算净增量，不吞掉 with 内抛出的异常。

        Args:
            exc_type: with 内抛出的异常类型；正常退出时为 None。
            exc_val: with 内抛出的异常对象；正常退出时为 None。
            exc_tb: 异常的调用栈信息；正常退出时为 None。

        Returns:
            None，表示不会抑制原异常；异常退出时也会尝试记录显存结果。

        Side Effects:
            创建 final_memory 和 consumed_memory 字段，执行采样及垃圾回收。
            这里只观测显存，不负责卸载模型或回滚区间内已完成的分配。
        """
        self.final_memory = self.current_memory_usage()
        # 只计算边界差值：临时 tensor 如果已被释放，就不会计入这一净增量。
        self.consumed_memory = self.final_memory - self.initial_memory

        # 保留 vLLM 退出时的垃圾回收步骤；不重新改写已记录的统计结果。
        gc.collect()


@dataclass
class MemorySnapshot:
    """记录某一时刻单张 CUDA GPU 的显存状态。

    Worker 在 NCCL process group 初始化后、模型权重加载前创建 ``init_snapshot``。
    这份快照是后续显存 profiling 的基线：它包含外部进程占用的显存和 NCCL 已分配的
    communicator buffer，但不包含本 Worker 即将加载的模型权重与 KV cache。

    Attributes:
        torch_peak: PyTorch allocator 记录的历史峰值 allocated bytes。
        free_memory: 设备当前空闲显存字节数。
        total_memory: 设备总显存字节数。
        cuda_memory: 当前已被 CUDA 使用的总显存，即 ``total_memory - free_memory``。
        torch_memory: PyTorch caching allocator 已向 CUDA 保留的显存字节数。
        non_torch_memory: CUDA 已用显存中不属于 PyTorch allocator 的部分，例如 NCCL
            communicator、某些 attention backend 的 workspace 或其他 CUDA 库内存。
        timestamp: 最近一次采样的 Unix 时间戳，单位为秒。
        device: 调用方指定的 CUDA device。可以是 ``torch.device``、字符串、设备编号
            或 ``None``；当前 nano-vLLM 的 Worker 会传入 ``self.device``。
        auto_measure: 是否在对象构造完成后立即采样显存。
    """
    # 以下字段由 measure() 填充；默认值仅用于构造“差值快照”。
    torch_peak: int = 0
    free_memory: int = 0
    total_memory: int = 0
    cuda_memory: int = 0
    torch_memory: int = 0
    non_torch_memory: int = 0
    timestamp: float = 0.0

    # device_ 是规范化后的 torch.device，供后续所有 CUDA API 使用。
    device: torch.types.Device = None
    auto_measure: bool = True

    def __post_init__(self) -> None:
        """
        规范化设备并按需立即采集显存状态。

        Raises:
            RuntimeError: 当前 CUDA 不可用时抛出。nano-vLLM 的 GPU Worker 不支持
                在 CPU 路径上创建显存快照。
        """

        # None 表示使用当前 CUDA device；其他形式统一转换为 torch.device。
        if self.device is None:
            self.device_ = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            self.device_ = torch.device(self.device)

        if self.device_.type != "cuda":
            raise ValueError(
                "MemorySnapshot currently supports CUDA devices only, got "
                f"{self.device_}."
            )

        # 常规路径在构造后立刻保存基线；差值快照会显式传 auto_measure=False。
        if self.auto_measure:
            self.measure()

    def measure(self) -> None:
        """刷新当前设备的显存统计。

        Side Effects:
            覆写本对象的全部显存统计字段与 ``timestamp``。不会分配或释放任何 GPU
            内存，只读取 CUDA runtime 与 PyTorch allocator 的状态。
        """
        device = self.device_
        # allocated peak 与 memory_reserved 的语义不同：前者可衡量一次 forward 的
        # 峰值激活开销，后者表示 PyTorch caching allocator 当前保留的 CUDA 内存。
        self.torch_peak = torch.cuda.memory_stats(device).get(
            "allocated_bytes.all.peak", 0
        )
        # mem_get_info 返回 (free_bytes, total_bytes)，包含 PyTorch 之外的 CUDA 用户。
        self.free_memory, self.total_memory = torch.cuda.mem_get_info(device)
        # CUDA 总占用包括 PyTorch、NCCL、第三方 CUDA 库与其他进程占用的设备内存。
        self.cuda_memory = self.total_memory - self.free_memory
        # PyTorch 已从 CUDA runtime 获取、但可能暂时未分配给 tensor 的缓存也应计入。
        self.torch_memory = torch.cuda.memory_reserved(device)
        # 该差值可近似观察 NCCL 等非 PyTorch 组件的显存变化。
        self.non_torch_memory = self.cuda_memory - self.torch_memory
        # 时间戳用于后续 profiling 结果记录这次快照的持续时间。
        self.timestamp = time.time()
    def __sub__(self, other: MemorySnapshot) -> MemorySnapshot:
        """计算同一 CUDA device 上两份显存快照的字段差值。

        该接口当前尚未被调用，但后续 ``memory_profiling()`` 需要它比较 forward 前后
        的峰值激活和非 PyTorch 内存增量，因此现在保留 vLLM 的基础接口。

        Args:
            other: 更早采集的同设备显存快照。

        Returns:
            一个 ``auto_measure=False`` 的新快照。每个数值字段均为
            ``self - other`` 的字节数或秒数差值。

        Raises:
            ValueError: 两份快照来自不同 CUDA device 时抛出。
        """
        if self.device_ != other.device_:
            raise ValueError(
                "The two snapshots must use the same device: "
                f"{self.device_} vs. {other.device_}."
            )

        return MemorySnapshot(
            torch_peak=self.torch_peak - other.torch_peak,
            free_memory=self.free_memory - other.free_memory,
            total_memory=self.total_memory - other.total_memory,
            cuda_memory=self.cuda_memory - other.cuda_memory,
            torch_memory=self.torch_memory - other.torch_memory,
            non_torch_memory=self.non_torch_memory - other.non_torch_memory,
            timestamp=self.timestamp - other.timestamp,
            device=self.device_,
            auto_measure=False,
        )
    def __repr__(self) -> str:
        """返回便于日志阅读的 GiB 格式显存摘要。"""
        return (
            f"torch_peak={format_gib(self.torch_peak)}GiB, "
            f"free_memory={format_gib(self.free_memory)}GiB, "
            f"total_memory={format_gib(self.total_memory)}GiB, "
            f"cuda_memory={format_gib(self.cuda_memory)}GiB, "
            f"torch_memory={format_gib(self.torch_memory)}GiB, "
            f"non_torch_memory={format_gib(self.non_torch_memory)}GiB, "
            f"timestamp={self.timestamp}, "
            f"auto_measure={self.auto_measure}"
        )

@dataclass
class MemoryProfilingResult:
    """一次模型 profile run 的显存统计结果。

    Attributes:
        non_kv_cache_memory: 除 KV cache 外，Worker 认为长期占用的显存总量。
            它由模型权重、profile 期间 PyTorch 峰值激活显存、以及非 PyTorch
            分配的 CUDA 显存组成。
        torch_peak_increase: profile run 期间 PyTorch allocator 的峰值显存增量。
            主要反映 forward 过程中 activation、临时 tensor 等显存需求。
        non_torch_increase: 从 Worker 初始化快照到 profile 结束期间，CUDA 已用
            显存增加但不属于 PyTorch caching allocator 的部分。例如 CUDA runtime、
            NCCL communicator、第三方 CUDA 库的内部 buffer 等。
        weights_memory: 已加载模型权重占用的 PyTorch 显存，由调用方在进入
            memory_profiling 时传入。
        before_create: Worker 初始化时记录的显存快照。它早于模型构建和权重加载。
        before_profile: 正式 profile 前的显存快照。此时模型权重已经加载完成。
        after_profile: profile run 完成后的显存快照。
        profile_time: profile run 区间的墙钟时间，单位为秒。
    """
    non_kv_cache_memory: int = 0
    torch_peak_increase: int = 0
    non_torch_increase: int = 0
    weights_memory: int = 0
    before_create: MemorySnapshot = field(default_factory=MemorySnapshot)
    profile_time: float = 0.0
    def __post_init__(self) -> None:
        """为 profile 前后快照固定与初始化快照相同的 CUDA device。"""
        device = self.before_create.device
        self.before_profile = MemorySnapshot(device=device, auto_measure=False)
        self.after_profile = MemorySnapshot(device=device, auto_measure=False)
    def __repr__(self) -> str:
        """返回便于日志输出的显存 profile 摘要。"""
        return (
            "MemoryProfilingResult("
            f"weights_memory={format_gib(self.weights_memory)}, "
            f"torch_peak_increase={format_gib(self.torch_peak_increase)}, "
            f"non_torch_increase={format_gib(self.non_torch_increase)}, "
            f"non_kv_cache_memory={format_gib(self.non_kv_cache_memory)}, "
            f"profile_time={self.profile_time:.3f}s)"
        )
    
@contextlib.contextmanager
def memory_profiling(
    baseline_snapshot: MemorySnapshot,
    weights_memory: int = 0,
) -> Generator[MemoryProfilingResult, None, None]:
    """统计一次模型 profile run 对 GPU 显存的真实需求。

    调用方应在 ``with`` 块中执行一次代表最坏情况的 forward，例如最大 batch
    下的 ``model_runner.profile_run()``。退出 ``with`` 后，返回的结果对象会
    填充峰值激活显存、非 PyTorch 显存增长和非 KV cache 总显存。

    Args:
        baseline_snapshot: Worker 初始化完成后记录的显存快照。它通常早于模型
            权重加载，用于计算非 PyTorch CUDA 显存的总增量。
        weights_memory: 已加载模型权重占用的 PyTorch 显存，单位为字节。

    Yields:
        用于承载本次 profile 结果的 ``MemoryProfilingResult``。进入 ``with``
        块前已记录 ``before_profile``；退出后会填充完整统计数据。
    单张 GPU 上的显存可以分成三类：

    1. 当前 nano-vLLM 实例以外的对象占用的显存。
       例如同一张 GPU 上其他进程、驱动程序或先前已经存在的 CUDA 分配。

    2. 当前 nano-vLLM 实例中由 PyTorch 管理的显存。
       例如模型权重、forward 中创建的 activation tensor、临时 tensor。

    3. 当前 nano-vLLM 实例使用、但不由 PyTorch allocator 管理的显存。
       例如 NCCL communicator、CUDA runtime、attention backend 内部 buffer
       或其他第三方 CUDA 库分配的显存。

    一个定量例子：

    创建当前 nano-vLLM 实例之前：

        类别 1：1 GiB
        类别 2：0 GiB
        类别 3：0 GiB

    创建当前 nano-vLLM 实例并加载模型后，即正式 profile 前：

        类别 1：1 GiB
        类别 2：2 GiB，模型权重占用 2 GiB
        类别 3：0.5 GiB，例如 NCCL 初始化所需显存

    profile forward 执行期间的峰值：

        类别 1：1 GiB
        类别 2：4 GiB，模型权重 2 GiB，加上峰值 activation tensor 2 GiB
        类别 3：1 GiB，例如 NCCL 加 attention backend buffer

    profile forward 结束并清理临时 activation 后：

        类别 1：1 GiB
        类别 2：3 GiB，部分 activation 已经释放或回收到 PyTorch cache
        类别 3：1 GiB

    因此，不能分给 KV cache 的非 KV cache 显存总量为 5 GiB：

    a. 模型权重：2 GiB
    b. 为峰值 activation 预留：2 GiB
    c. 非 PyTorch CUDA 组件：1 GiB

    ``weights_memory`` 直接得到 a。

    ``torch.cuda.memory_stats()["allocated_bytes.all.peak"]`` 在 profile
    区间内的增长量得到 b。

    从创建 Worker 的基线快照到 profile 结束，``non_torch_memory`` 的增长量
    得到 c。
    """
    # 清理不再引用的 Python 对象及 PyTorch allocator 缓存，减少历史临时分配
    # 对本次 profile 的干扰。
    gc.collect()
    torch.cuda.empty_cache()
    # 从当前时刻重新统计 PyTorch allocator 峰值。之后 profile_run 产生的最大
    # activation / 临时 tensor 占用会体现在 allocated_bytes.all.peak 中。
    torch.cuda.reset_peak_memory_stats(baseline_snapshot.device)
    result = MemoryProfilingResult(
        before_create=baseline_snapshot,
        weights_memory=weights_memory,
    )
    result.before_profile.measure()

    yield result

    # profile_run 已完成；再次清理缓存后测量稳定状态下的显存占用。
    gc.collect()
    torch.cuda.empty_cache()
    result.after_profile.measure()
    # 两个 profile 快照的差值反映本轮 forward 的峰值 PyTorch 显存和耗时。
    profile_diff = result.after_profile - result.before_profile

    # 与 Worker 初始化时的快照比较，用来捕获模型加载、CUDA runtime 或第三方库
    # 带来的非 PyTorch 显存增长。
    create_diff = result.after_profile - result.before_create

    result.torch_peak_increase = profile_diff.torch_peak
    result.non_torch_increase = create_diff.non_torch_memory
    result.profile_time = profile_diff.timestamp

    # 这部分显存不能分配给 KV cache；剩余显存才是后续 KV cache 预算的候选。
    result.non_kv_cache_memory = (
        result.weights_memory
        + result.torch_peak_increase
        + result.non_torch_increase
    )