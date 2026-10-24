"""管理 CUDA Worker 的启动、显存预算、执行入口和关闭流程。

Executor 通过 WorkerWrapperBase 调用本模块的 Worker；Worker 负责绑定设备、
初始化通信环境、测量可用显存，并编排模型加载、KV 初始化和启动预热。
GPUModelRunner 持有模型、KV tensor、固定输入缓冲区及请求镜像，负责实际执行。

当前只接入单卡同步 Executor。保留 vLLM 的 Worker 接口与调用顺序，省略尚未
支持的 PP/DP、CuMem sleep/wake、KV transfer、profiler 和模型级编译分支。
"""

from __future__ import annotations

import torch
import gc
from nanovllm.config import NanovllmConfig,set_current_config
from nanovllm.config.compilation import CUDAGraphMode
from nanovllm.core.sched.output import SchedulerOutput
from nanovllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from nanovllm.kv_cache_interface import KVCacheConfig, KVCacheSpec
from nanovllm.logger import init_logger
from nanovllm.model_executor.warmup.kernel_warmup import kernel_warmup
from nanovllm.outputs import ModelRunnerOutput,AsyncModelRunnerOutput
from nanovllm.platforms import current_platform
from nanovllm.worker.gpu_model_runner import GPUModelRunner
from nanovllm.worker.worker_base import WorkerBase,CompilationTimes
from nanovllm.utils.mem_utils import (
    MemorySnapshot,
    format_gib,memory_profiling,
    memory_profiling,
    )
from nanovllm.utils.mem_constants import GiB_bytes
from nanovllm.utils.gc_utils import freeze_gc_heap
from .utils import request_memory

logger = init_logger(__name__)


class Worker(WorkerBase):
    """为一个模型执行 rank 管理 CUDA 资源，并向 Executor 暴露统一入口。

    当前 UniProcExecutor 只创建 rank 0 的 Worker。rank 表示通信身份，local_rank
    表示本机可见 GPU 编号；保留两者以对齐 vLLM，设备绑定始终使用 local_rank。

    启动和执行按以下顺序组织。KV spec 与可用字节数交给 Engine 生成
    KVCacheConfig；Worker 接收该分配方案后，才创建正式 KV cache 和捕获图。

    .. code-block:: text

        __init__
        -> init_device
        -> load_model
        -> get_kv_cache_spec / determine_available_memory
        -> initialize_from_config
        -> compile_or_warm_up_model
        -> execute_model -> sample_tokens（forward 返回 None 时）
        -> shutdown

    Attributes:
        device: init_device() 绑定的具体 CUDA device；构造 Worker 时尚未绑定。
        model_runner: 实际持有模型与运行时状态的 GPUModelRunner。
        init_snapshot: 分布式初始化后、Runner 和模型加载前的设备显存基线。
        requested_memory: 根据设备总显存与 gpu_memory_utilization 得到的实例
            总预算，单位为字节；包含权重、激活、Graph 与 KV cache。
        non_torch_memory: profiling 得到的非 PyTorch 显存增量，单位为字节。
        peak_activation_memory: Graph profiling 前保存的 PyTorch 峰值激活增量。
        cudagraph_memory_estimate: 正式捕获前估算的 Graph 显存，禁用时为零。
        available_kv_cache_memory_bytes: 从总预算扣除非 KV 需求后的候选 KV 预算，
            由 Engine 换算为物理 block 数；不代表已经分配的 tensor 大小。
    """
    def __init__(
        self,
        vllm_config: NanovllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        """保存执行配置和 rank 信息，设备与模型资源由后续生命周期接口创建。

        Args:
            vllm_config: Engine 下发的完整配置；WorkerBase 保存各子配置的引用。
            local_rank: 当前进程可见 GPU 的本地编号，用于 init_device() 绑定设备。
            rank: 当前模型执行 rank 的通信身份，单卡 Executor 固定为 0。
            distributed_init_method: 创建默认进程组的 rendezvous 地址；当前由
                UniProcExecutor 提供本机 tcp:// 地址。
            is_driver_worker: 是否为向 Engine 返回结果的 Worker；当前单卡为 True。

        Side Effects:
            WorkerBase 写入配置别名、rank 及初始化占位状态，不初始化 CUDA，
            不构造 GPUModelRunner，也不加载权重。
        """
        # 配置和 rank 的保存与校验统一复用基类，实际资源创建留给 init_device()。
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
    
    def init_device(self) -> None:
        """绑定 CUDA 设备、建立分布式环境并创建 GPUModelRunner。

        沿用 vLLM 的顺序：设置设备、检查模型 dtype、初始化分布式环境，
        清理临时对象并采集显存基线，最后创建 Runner。此方法先于 load_model()
        调用；基线既不包含随后分配的 Runner 输入缓冲区，也不包含模型权重。

        DeviceConfig 已限定执行后端为 CUDA，local_rank 直接来自 Executor，
        因此执行单一 CUDA 路径；模型层构造时查询的 world / TP group 必须先建立。

        Raises:
            RuntimeError: CUDA 不可用或 ``local_rank`` 不对应可见 GPU 时抛出。
            ValueError: 显式设备编号冲突、模型 dtype 不受设备支持，或启动时
                空闲显存不足以满足请求预算时抛出。

        Side Effects:
            切换当前 CUDA device、初始化 process group，设置 device、
            init_snapshot、requested_memory 与 model_runner。模型层稍后构造时
            依赖这些设备与分布式状态。
        """
        # GPU 可用性在资源创建入口检查，避免配置构造阶段就触发 CUDA 初始化。
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. nano-vLLM currently requires a CUDA GPU."
            )

        # visible_device_count 是当前进程可见设备数，编号范围受 CUDA_VISIBLE_DEVICES 影响。
        visible_device_count = current_platform.device_count()
        if not 0 <= self.local_rank < visible_device_count:
            raise RuntimeError(
                "local_rank is outside the visible CUDA device range: "
                f"local_rank={self.local_rank}, "
                f"num_visible_devices={visible_device_count}."
            )

        # configured_device 保存配置中的设备描述；显式 cuda:N 必须与 local_rank 一致，
        # 无编号的 cuda 则由本 Worker 绑定，避免模型加载和输入分配落到不同设备。
        configured_device = self.device_config.device
        if (
            configured_device.index is not None
            and configured_device.index != self.local_rank
        ):
            raise ValueError(
                f"Configured device {configured_device} conflicts with "
                f"Worker local_rank={self.local_rank}."
            )
        # self.device 是实际执行设备；后续不带编号的 CUDA 分配也应遵循此绑定。
        self.device = torch.device(f"cuda:{self.local_rank}")
        # 平台封装调用 torch.cuda.set_device 并做一次小分配，立即完成设备初始化。
        current_platform.set_device(self.device)
        # 精度检查必须在切换设备之后，才能查询当前 Worker 对应 GPU 的架构能力。
        current_platform.check_if_supports_dtype(self.vllm_config.model_config.dtype)

        # 分布式初始化早于显存快照，已经产生的通信内存纳入启动基线。
        # 即使单卡也建立 world / TP group，供模型层查询 rank 和 world size；
        # 延迟到后续 forward 才发生的非 PyTorch 分配由 memory_profiling 统计。
        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )

        # 先回收不可达 Python 对象，使其不再持有临时 tensor。
        gc.collect()
        # 再归还 allocator 的空闲缓存；仍被对象引用的显存不会被 empty_cache() 释放。
        torch.cuda.empty_cache()

        # init_snapshot 保存测量对象，self.init_snapshot 使该基线在后续 profiling 可用。
        self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)
        # request_memory 按设备总显存计算预算，并检查启动空闲显存能否满足该预算。
        self.requested_memory = request_memory(init_snapshot, self.cache_config)
        logger.debug("worker init memory snapshot: %r", self.init_snapshot)
        logger.debug(
            "worker requested memory: %sGiB", format_gib(self.requested_memory)
        )
        # 创建并保存唯一 Runner；构造期只分配运行时缓冲区，权重仍由 load_model() 加载。
        self.model_runner:GPUModelRunner = GPUModelRunner(self.vllm_config, self.device)

    def load_model(self,*, load_dummy_weights: bool = False):
        """在已绑定设备上委托 Runner 构造模型、加载权重并建立执行包装。

        调用前 init_device() 已创建 Runner 和分布式环境。模型构造期间的
        Attention 层需要读取当前配置，并登记到 static_forward_context；
        因此 Worker 显式安装配置上下文，退出后恢复此前的上下文。

        Args:
            load_dummy_weights: 透传给 Runner 的同名标记；Worker 不据此选择
                Loader。当前正常启动使用 False，实际加载能力由 Runner 决定。

        Side Effects:
            Runner 保存模型引用、权重显存统计和可能的 CUDA Graph Wrapper，
            模型层填充静态注册表。此时尚未分配正式 KV cache。
        """
        # 当前使用默认 allocator；上游临时调整 max_split_size_mb 的封装暂未接入。
        with set_current_config(self.vllm_config):
            # 模型构造与加载全部归 Runner，Worker 不持有第二份模型实例。
            self.model_runner.load_model(load_dummy_weights=load_dummy_weights)

    def get_model(self) -> torch.nn.Module:
        """查询 Runner 持有的原始模型，绕过外层执行 Wrapper。

        供需要访问模型 module 的调用方使用；返回原对象引用，不复制权重。
        Wrapper 的识别与 unwrap() 由 Runner 完成，Worker 只转发查询。

        Returns:
            model: 当前 GPUModelRunner 持有的已加载语言模型。

        Raises:
            ValueError: 模型尚未加载时，由 ``GPUModelRunner.get_model()`` 抛出。
        """
        return self.model_runner.get_model()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """查询已加载模型各 Attention 层的 KV 存储规格，供 Engine 规划分配。

        在 load_model() 完成后调用。Runner 从静态层注册表收集 KVCacheSpec，
        Worker 不自行遍历模型或计算分配大小。返回结果与可用显存字节数一起
        交给 Engine，生成后续 initialize_from_config() 接收的 KVCacheConfig。

        Returns:
            kv_cache_spec: 从稳定 Attention layer 全名到 ``KVCacheSpec`` 的映射。例如 Qwen3 的
                key 形如 ``"model.layers.0.self_attn.attn"``。不需要独立 KV cache 的
                layer 不出现在映射中；规格本身不持有已分配的 GPU KV tensor。
        """
        # 只查询规格，不改变 block 数、KV 层绑定或 Scheduler 的请求状态。
        return self.model_runner.get_kv_cache_spec()

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """测量非 KV 显存需求，并返回正式 KV cache 的候选预算（字节）。

        在 load_model() 之后、initialize_from_config() 之前调用。已加载权重
        提供 weights_memory，最大 token budget 的 dummy forward 和采样负载
        提供激活需求；Graph profiling 使用可清理的临时 KV，正式 KV 尚未分配。

        沿用 vLLM 的顺序：执行 profile_run()、保存 PyTorch 峰值，再估算
        CUDA Graph 显存；退出 memory_profiling() 后读取完整统计，使用保存的
        峰值重算激活增量，避免临时 KV 和 Graph 分配再次计入激活预算。

        预算公式为 requested_memory - 权重 - 峰值激活增量 - 非 PyTorch
        显存增量 - Graph 估算。当前始终应用 Graph 估算；显式
        CacheConfig.kv_cache_memory_bytes 预算分支尚未接入。

        Returns:
            available_memory: 交给 Engine KV cache planner 的候选显存字节数，不是物理 block 数。
                不将不足的预算截断为零；是否足以容纳请求由 planner 校验。

        Raises:
            AssertionError: profiling 后的空闲显存大于初始化基线，可能发生
                外部进程释放显存等变化，无法继续依据原基线规划。
            RuntimeError: 模型 profiling、dummy 预热或 Graph 捕获失败。

        Side Effects:
            执行模拟模型和采样负载，重置并读取 CUDA 显存统计；启用 Graph 时
            临时分配、捕获和清理 profiling KV cache 与图。保存 non_torch_memory、
            peak_activation_memory、cudagraph_memory_estimate 和最终 KV 预算。
            模拟采样会消耗默认 RNG 状态，不将模拟结果提交给真实请求。
        """
        # memory_profiling 先清理缓存、重置峰值并记录 before_profile；profile_result
        # 在 with 退出后才填充 after_profile、non_torch_increase 和非 KV 总量。
        # model_memory_usage 是 Runner 测得的权重显存字节数，作为独立项传入统计。
        with memory_profiling(
            self.init_snapshot,
            weights_memory=int(self.model_runner.model_memory_usage),
        ) as profile_result:
            # 先测模型和 sampler 的负载，得到 Graph profiling 介入前的激活需求。
            self.model_runner.profile_run()
            # profile_torch_peak 是 Graph profiling 前的峰值 allocated bytes，
            # 不是 memory_reserved；稍后用它减去 before_profile 的峰值得到激活增量。
            profile_torch_peak = torch.cuda.memory_stats(self.device).get(
                "allocated_bytes.all.peak", 0
            )
            # cudagraph_memory_estimate 是临时捕获推导的字节估算；禁用时保持零。
            cudagraph_memory_estimate = 0
            # 仅启用 CUDA Graph 时调用 Runner 估算；该调用会分配临时 KV 并捕获图，
            # 可能提高 allocator 峰值，所以必须在保存 profile_torch_peak 之后执行。
            if (
                current_platform.is_cuda()
                and self.vllm_config.compilation_config.cudagraph_mode
                != CUDAGraphMode.NONE
            ):
                cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory()

        # 退出上下文后的默认峰值包含临时 Graph profiling；改用事先保存的模型峰值，
        # 避免临时 KV / Graph 被当成正式推理激活，Graph 开销随后单独扣除。
        profile_result.torch_peak_increase = (
            profile_torch_peak - profile_result.before_profile.torch_peak
        )
        # 替换激活增量后同步重算 non_kv_cache_memory；它不包含单独的 Graph 估算。
        # non_torch_increase 来自初始化基线到 profiling 结束的非 PyTorch 显存差，
        # 可反映模型加载、NCCL、CUDA 库等在基线之后产生的常驻分配。
        profile_result.non_kv_cache_memory = (
            profile_result.non_torch_increase
            + profile_result.torch_peak_increase
            + profile_result.weights_memory
        )
        # 保留上游变量名：estimate 是估算结果，estimate_applied 是预算实际扣除项。
        # 当前没有 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS 开关，两者始终相等。
        cudagraph_memory_estimate_applied = cudagraph_memory_estimate
        # 将最终统计保存到 Worker，后续日志或预热阶段可复用，单位均为字节。
        self.non_torch_memory = profile_result.non_torch_increase
        self.peak_activation_memory = profile_result.torch_peak_increase
        self.cudagraph_memory_estimate = cudagraph_memory_estimate
        # free_gpu_memory 是 profiling 结束并清理临时资源后的设备空闲字节数。
        # 它用于诊断基线变化；最终 KV 预算仍从 requested_memory 扣除各项需求得到。
        free_gpu_memory = profile_result.after_profile.free_memory
        # 与上游相同：拒绝空闲显存反而增长的 profile 结果，不用不稳定的基线规划 KV。
        assert self.init_snapshot.free_memory >= free_gpu_memory, (
            "GPU free memory increased while profiling. The memory baseline is "
            "unstable, likely because another process released GPU memory: "
            f"initial_free={format_gib(self.init_snapshot.free_memory)} GiB, "
            f"current_free={format_gib(free_gpu_memory)} GiB."
        )
        # available_kv_cache_memory_bytes 是候选 KV 字节预算，尚未换算或分配物理 blocks。
        # 总预算 10 GiB 减权重 2、激活 1、非 PyTorch 0.5、Graph 0.5，available_memory 为 6 GiB。
        self.available_kv_cache_memory_bytes = (
            self.requested_memory
            - profile_result.non_kv_cache_memory
            - cudagraph_memory_estimate_applied
        )
        # unrequested_memory 是启动空闲显存中未纳入实例预算的部分，只用于日志。
        # free_gpu_memory - unrequested_memory 展示请求预算内当前尚空闲的部分；
        # 其中还未预留后续激活与正式 Graph，因此与最终 KV 预算含义不同。
        unrequested_memory = (
            self.init_snapshot.free_memory - self.requested_memory
        )
        logger.debug(
            "Initial free memory: %s GiB; requested memory: %.4f util, %s GiB.",
            format_gib(self.init_snapshot.free_memory),
            self.cache_config.gpu_memory_utilization,
            format_gib(self.requested_memory),
        )
        logger.debug(
            "Free memory after profiling: %s GiB total, %s GiB within "
            "the requested budget.",
            format_gib(free_gpu_memory),
            format_gib(free_gpu_memory - unrequested_memory),
        )
        logger.debug("Memory profiling result: %r", profile_result)
        logger.info_once(
            "Available KV cache memory: %s GiB.",
            format_gib(self.available_kv_cache_memory_bytes),
        )

        return int(self.available_kv_cache_memory_bytes)
    
    def initialize_from_config(
        self,
        kv_cache_config:KVCacheConfig,
    )->None:
        """接收 Engine 的 KV 分配方案，回填容量镜像并委托 Runner 创建缓存。

        determine_available_memory() 仅返回字节预算；KVCacheConfig 中的 block
        数、tensor 分配计划和 group 关系由 Engine 确定。此方法在显存规划后、
        compile_or_warm_up_model() 前调用，随后预热和捕获都使用这批正式 tensor。

        例如 kv_cache_config.num_blocks=129 时，cache_config.num_gpu_blocks
        写入 129；其中 block 0 为保留空块，正常请求最多使用其余 128 个块。

        Args:
            kv_cache_config: 当前 rank 的实际分配方案。num_blocks 是物理总容量，
                kv_cache_tensors 描述各段存储，kv_cache_groups 描述共享页表关系。

        Side Effects:
            回写 cache_config.num_gpu_blocks，并由 Runner 创建和绑定正式 KV cache、
            Attention metadata Builder 及 CUDA Graph Dispatcher keys。
        """
        # num_gpu_blocks 是正式规划结果的运行时镜像，不是用户配置的 block 数覆盖值。
        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        # 当前使用默认 PyTorch allocator，没有 CuMemAllocator / sleep-wake，
        # 因而省略上游 _maybe_get_memory_pool_context(tag="kv_cache")。
        self.model_runner.initialize_kv_cache(kv_cache_config)

        # 当前 FullAttention 路径省略由 Mamba / hybrid 模型触发的按块清零调度链，
        # 因此无需调用上游的 _init_kv_zero_meta()。

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput|AsyncModelRunnerOutput | None:
        """执行一轮调度计划；返回 None 时由调用方继续触发 sample_tokens()。

        Args:
            scheduler_output: Scheduler 本轮快照，含请求状态变化、KV block 映射
                增量及每请求执行 token 数。Worker 原样交给 Runner。

        Returns:
            model_output: 有调度 token 时，Runner 执行 forward、暂存 logits 并返回 None；
                下一轮 execute_model() 前必须先消费这轮采样状态。
                空调度返回空 ModelRunnerOutput，调用方可直接处理该结果。

        Raises:
            RuntimeError: 上一轮 logits 尚未消费，或底层模型执行失败。

        Side Effects:
            Runner 同步请求镜像、准备输入、执行模型并写入 KV；有 forward 时
            设置 execute_model_state。Worker 不单独维护第二套请求或采样状态。
        """
        # FULL decode / eager 的选择、padding 和缓存写入都保留在 Runner 内部。
        return self.model_runner.execute_model(scheduler_output)

    @torch.inference_mode()
    def sample_tokens(self) -> ModelRunnerOutput|AsyncModelRunnerOutput:
        """消费最近一次 forward 的 logits，返回已按请求过滤的 CPU 采样结果。

        在 execute_model() 返回 None 后调用一次。Runner 完成 temperature
        采样、过滤尚未完成 prefill 的请求，并回写有效生成 token。

        Returns:
            output: 同步时为包含请求行映射及有效 token IDs 的 ModelRunnerOutput；
                异步时为 AsyncModelRunnerOutput，调用 get_output() 后得到可读的 CPU 结果。

        Raises:
            RuntimeError: 没有待采样状态，例如尚未 forward 或重复消费同一轮结果。

        Side Effects:
            Runner 清空 execute_model_state，更新有效请求的 token 镜像；
            Scheduler 的权威进度仍由上层根据返回结果提交。
        """
        # 沿用 Runner 的同步 CPU 结果协议，Worker 不额外复制或重新排列采样行。
        return self.model_runner.sample_tokens()

    def compile_or_warm_up_model(self) -> CompilationTimes:
        """在正式 KV cache 分配后预热内核、捕获 Graph 并预热采样路径。

        必须在 initialize_from_config() 完成后、正式请求进入 InputBatch 前调用。
        此时模型、KV tensor、Attention Builder 和 Dispatcher keys 已就绪，
        启动执行才能使用与真实推理一致的地址和布局。

        沿用 vLLM 的顺序：编译尺寸预热、kernel_warmup、capture_model、
        显存诊断、词表投影和 sampler 预热，最后冻结 Python GC 跟踪堆并返回
        编译统计。当前没有模型级编译，warmup_sizes 保持空列表。

        enforce_eager=True 只跳过 Graph 捕获，内核和 sampler 仍会预热。
        FA2 内核预热由 kernel_warmup() 使用至多 16 token 的 mixed batch 完成，
        实际 token 数还受 token budget 和 max_model_len 限制；Graph 每个档位的
        预热归 capture_model()，最后的普通 dummy 则为 sampler 准备输入。

        Returns:
            compilation_times: CompilationTimes，language_model 为模型级编译统计秒数。
                当前该统计默认 0.0，不包含本方法的总耗时、Graph 捕获时间，
                也不自动累计局部 @torch.compile 函数的首次 JIT 时间。

        Side Effects:
            更新 Runner 的 dummy 输入和 metadata，触发内核首次初始化；启用
            Graph 时持有正式捕获条目与 pool 分配。采样预热消耗默认 RNG，
            但不回写真实请求。freeze_gc_heap() 改变进程级 Python GC 跟踪状态。
            显存建议只用于日志，不重新分配 KV，也不修改已有显存预算。

        Raises:
            AssertionError: dummy 输入超出容量，或 Graph keys 尚未初始化。
            RuntimeError: Attention 预热、CUDA Graph 捕获或采样失败。
        """
        # warmup_sizes 保存模型级编译需要额外执行的 token 尺寸；保留上游变量名。
        # 当前未接入 CompilationMode / compile_sizes / compile ranges，所以列表为空。
        warmup_sizes: list[int] = []
        # 大尺寸优先的循环沿用上游结构；当前没有任何额外编译尺寸，循环不会执行。
        for size in sorted(warmup_sizes, reverse=True):
            # size 是待预热的 packed token 数，未来由编译配置填充。
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(size)
        
        # 与 vLLM 一样在 Graph capture 前预热执行内核；enforce_eager 也保留此步骤。
        # 当前 FA2 的真实 Attention 预热由该入口完成，Graph 各档预热仍归 Runner。
        kernel_warmup(self)
        # cuda_graph_memory_bytes 是正式捕获测得的显存增量，未捕获时保持零。
        # 它与 determine_available_memory() 中临时捕获得到的 estimate 分开记录。
        cuda_graph_memory_bytes = 0
        # enforce_eager 禁止正式捕获；capture_model 内部另行处理 NONE 策略和逐档预热。
        if not self.model_config.enforce_eager:
            cuda_graph_memory_bytes = self.model_runner.capture_model()
        # 只有已有正数估算时才输出比较，避免跳过 profiling 的路径访问不存在的字段。
        if (
            hasattr(self, "cudagraph_memory_estimate")
            and self.cudagraph_memory_estimate > 0
        ):
            # GiB 仅供日志换算：输入为字节，输出为保留两位小数的二进制 GiB。
            GiB = lambda b: round(b / GiB_bytes, 2)
            # diff 是实测和估算的绝对字节差；它只帮助评估估算误差，不修正 KV 规划。
            diff = abs(cuda_graph_memory_bytes - self.cudagraph_memory_estimate)
            logger.info(
                "CUDA graph pool memory: %s GiB (actual), %s GiB (estimated), "
                "difference: %s GiB (%.1f%%).",
                GiB(cuda_graph_memory_bytes),
                GiB(self.cudagraph_memory_estimate),
                GiB(diff),
                # 百分比以实测值为基准；max(..., 1) 防止实测为零时发生除零。
                100 * diff / max(cuda_graph_memory_bytes, 1),
            )
        # 自动预算路径且已有激活统计时，汇总正式捕获后的显存诊断建议。
        # kv_cache_memory_bytes 当前只参与这个日志分支，尚未接入显式 KV 预算计算。
        if self.cache_config.kv_cache_memory_bytes is None and hasattr(
            self, "peak_activation_memory"
        ):
            # 预算阶段使用 Graph estimate；这里用正式 capture 的实测值重算诊断数值。
            # redundancy_buffer_memory 是建议值中的 150 MiB 余量，不是新分配的 tensor，
            # 也不会从已生效的 available_kv_cache_memory_bytes 中再次扣除。
            redundancy_buffer_memory = 150 * (1 << 20)

            # non_kv_cache_memory 汇总权重、峰值激活、非 PyTorch 和实测 Graph 占用，
            # 单位为字节；这份局部诊断值不覆盖 determine_available_memory() 的结果。
            non_kv_cache_memory = (
                self.model_runner.model_memory_usage
                + self.peak_activation_memory
                + self.non_torch_memory
                + cuda_graph_memory_bytes
            )
            # kv_cache_memory_bytes_to_gpu_limit 假定使用启动时所有空闲显存，
            # 扣除非 KV 需求和建议余量后，得到可考虑的 KV 字节数。
            kv_cache_memory_bytes_to_gpu_limit = (
                self.init_snapshot.free_memory
                - non_kv_cache_memory
                - redundancy_buffer_memory
            )
            # kv_cache_memory_bytes_to_requested_limit 则遵守当前实例的请求预算，
            # 其与上一个建议值的差异来自启动空闲显存和 requested_memory 的差值。
            kv_cache_memory_bytes_to_requested_limit = (
                int(self.requested_memory)
                - non_kv_cache_memory
                - redundancy_buffer_memory
            )

            # msg 汇总启动基线、实测占用和建议值；保留当前日志文本。
            # 其中 --kv-cache-memory 提示尚无本项目的显式预算分支，不能视为已支持。
            msg = (
                f"Free memory on device "
                f"({format_gib(self.init_snapshot.free_memory)}/"
                f"{format_gib(self.init_snapshot.total_memory)} GiB) on startup. "
                f"Desired GPU memory utilization is "
                f"({self.cache_config.gpu_memory_utilization}, "
                f"{format_gib(self.requested_memory)} GiB). "
                f"Actual usage is {format_gib(self.model_runner.model_memory_usage)} "
                f"GiB for weight, {format_gib(self.peak_activation_memory)} GiB "
                f"for peak activation, {format_gib(self.non_torch_memory)} GiB "
                f"for non-torch memory, and {format_gib(cuda_graph_memory_bytes)} "
                f"GiB for CUDAGraph memory. Replace gpu_memory_utilization "
                f"({format_gib(kv_cache_memory_bytes_to_requested_limit)} GiB) to fit "
                f"into requested memory, or `--kv-cache-memory="
                f"{kv_cache_memory_bytes_to_gpu_limit}` "
                f"({format_gib(kv_cache_memory_bytes_to_gpu_limit)} GiB) to fully "
                f"utilize gpu memory. Current kv cache memory in use is "
                f"{format_gib(self.available_kv_cache_memory_bytes)} GiB."
            )

            logger.debug(msg)
        # max_num_reqs 是 sampler 预热覆盖的最大有效请求数；每请求安排一个 query，
        # 因而同时受 max_num_seqs 和单轮 token budget 限制，不包含 Graph padding 行。
        max_num_reqs = min(
            self.scheduler_config.max_num_seqs,
            self.scheduler_config.max_num_batched_tokens,
        )
        # 此步骤放在 capture_model() 之后，沿用上游顺序，避免捕获阶段的 empty_cache()
        # 清掉刚形成的采样 allocator 缓存。NONE 明确禁止此处再次捕获或回放 Graph。
        # 这里只需获得采样输入形状，不传 force_attention，使用 Attention 占位路径。
        # hidden_states 为全部 token 行；last_hidden_states 为每请求最后一个 query 行；
        # 当前每请求一个 token，因此两者行数均为 max_num_reqs。
        hidden_states, last_hidden_states = self.model_runner._dummy_run(
            num_tokens=max_num_reqs,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        )
        # _dummy_sampler_run 按这些行的形状另建随机输入，预热 LMHead 和 temperature
        # sampler；不消费真实 execute_model_state，也不将候选 token 回写请求。
        self.model_runner._dummy_sampler_run(hidden_states=last_hidden_states)
        # 预热结束后冻结当前 Python GC 跟踪对象，减少常驻对象的后续循环 GC 扫描。
        # 这是进程级 GC 策略，不是冻结 CUDA Graph pool，也不是禁用未来对象的 GC。
        freeze_gc_heap()
        # 返回 compilation_times，language_model 取配置累计秒数；当前没有该编译链，默认 0.0。
        return CompilationTimes(
            language_model=self.compilation_config.compilation_time,
        )

    def shutdown(self)->None:
        """恢复 Python GC 扫描范围，并委托 Runner 清理模型执行资源。

        调用方应先停止提交模型执行任务。保持 vLLM 的关闭顺序：先解除启动预热
        对 Python 对象的 GC 冻结，再调用已创建 Runner 的 shutdown()。
        Worker 尚未完成 init_device() 时也允许进入此接口。

        Side Effects:
            gc.unfreeze() 将冻结对象放回普通 GC 跟踪集合，并不立即回收它们。
            Runner 随后解除 KV 绑定、清理静态层注册表、模型引用及 RoPE 缓存；
            对象底层存储能否回收取决于是否仍有其他引用。此接口不销毁进程组。
            清理期间的异常继续向上传播。
        """
        # freeze_gc_heap() 影响整个进程，关闭时恢复被冻结对象参与循环 GC 的能力。
        gc.unfreeze()
        # model_runner 可能仍为 None；只有设备初始化已创建 Runner 时才委托清理。
        if model_runner := getattr(self, "model_runner", None):
            model_runner.shutdown()


def init_worker_distributed_environment(
    vllm_config: NanovllmConfig,
    rank: int,
    distributed_init_method: str | None = None,
    local_rank: int = -1,
    backend: str = "nccl",
) -> None:
    """先建立默认 world，再建立或校验当前 Worker 使用的 TP group。

    由 Worker.init_device() 在绑定设备后、采集显存基线和构造模型前调用。
    即使单卡也创建这些 group，保证模型层可统一查询 rank 和 world size。
    本函数只编排初始化顺序，进程组状态由 distributed.parallel_state 持有。

    保留 vLLM 的入口及先 world、后模型并行的调用顺序；当前仅配置 TP 分组，
    不包含上游其他并行组、custom all-reduce 或 connector 的初始化。

    Args:
        vllm_config: 当前 Worker 的完整 nano-vLLM 配置。
        rank: 当前 Worker 在默认分布式 world 中的 global rank。
        distributed_init_method: 进程会合地址，单卡 Executor 传入本机 tcp:// 地址；
            为 None 或空字符串时使用 env://，会合所需环境变量由调用方提供。
        local_rank: 已绑定的本机可见 CUDA device 编号。默认 -1 时由下层根据
            初始化方式解析，本函数不负责调用 torch.cuda.set_device()。
        backend: 期望的 PyTorch distributed backend；CUDA 平台当前提供 nccl。

    Raises:
        RuntimeError: 通信初始化失败，或已有 world group 与当前默认进程组冲突。
        AssertionError: 已有 TP group 大小与配置不一致。

    Side Effects:
        首次调用时创建默认 PyTorch process group、world group 和 TP group；
        后续调用复用并校验已有分组。初始化时已发生的通信分配进入随后采集的
        Worker 显存基线，延迟到 forward 的分配由后续 profiling 统计。
    """
    # parallel_config 提供默认 world 大小和 TP 分组大小，不从可见 GPU 数量推导。
    parallel_config = vllm_config.parallel_config
    # init_method 保留上游的默认规则；env:// 交由 PyTorch 读取会合环境变量。
    init_method = distributed_init_method or "env://"

    # 默认 world 是后续 TP 分组的基础；通信库可能延迟分配内部 buffer，
    # 因而不能把本次初始化完成理解为所有通信显存都已分配。
    init_distributed_environment(
        world_size=parallel_config.world_size,
        rank=rank,
        distributed_init_method=init_method,
        local_rank=local_rank,
        backend=backend,
    )

    # 再从默认 world 划分或校验 TP 子组。当前配置只有 TP，world_size 与 TP size
    # 相同，但仍分别传给对应接口；单卡 Executor 下两者都为 1。
    ensure_model_parallel_initialized(
        tensor_model_parallel_size=parallel_config.tensor_parallel_size,
    )
