"""定义 Engine 访问 Worker 的公共生命周期、RPC 与模型执行接口。

Executor 负责创建 Worker、组织调用并适配结果。Engine 通过此边界查询 KV 规格与
显存预算，将规划好的 KVCacheConfig 下发，再提交 SchedulerOutput 和消费采样结果。
请求调度与 KV 规划位于 Engine / Scheduler 一侧，输入准备与模型执行位于 Worker /
GPUModelRunner 一侧。

当前具体实现为单卡 UniProcExecutor，接口沿用 vLLM 的 rank 有序结果协议。
profile、sleep/wake 和额外 dummy batch 入口仍然保留，其 Worker 能力尚未接入；
这些预留方法不参与当前正常启动和推理流程。
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

from nanovllm.config import NanovllmConfig
from nanovllm.logger import init_logger

from nanovllm.kv_cache_interface import KVCacheConfig, KVCacheSpec
from nanovllm.core.sched.output import SchedulerOutput
from nanovllm.outputs import ModelRunnerOutput
from nanovllm.worker.worker_base import WorkerBase,CompilationTimes

logger = init_logger(__name__)


# _R 表示一次 Worker 调用的返回类型；普通 RPC 在外层包装 list 或 Future[list]。
# callable 的返回类型可帮助类型检查器推导 _R；字符串方法名的返回类型由调用方约定。
_R = TypeVar("_R")
# 不可恢复故障的通知回调类型；当前单进程实现未保存或触发此类回调。
FailureCallback = Callable[[], None]


class Executor(ABC):
    """约定 Worker 的创建、KV 初始化、执行调用及结果返回方式。

    当前 get_class() 只选择 TP=1 的 UniProcExecutor，其 Worker 与调用方位于
    同一进程。基类仍以按 rank 排列的列表表示普通 RPC 结果，模型执行接口则
    向 Engine 返回单个 ModelRunnerOutput、None 或子类提供的单结果 Future。

    构造完成时 Worker 已加载权重，正式 KV cache 尚未分配。调用方应先查询
    get_kv_cache_specs() 与 determine_available_memory()，用规划器生成配置，
    再调用 initialize_from_config() 分配正式 KV 并预热，随后才能执行真实请求。

    execute_model() 的默认实现仅处理同步调用；sample_tokens() 保留上游的
    取首项写法。当前 UniProcExecutor 覆盖这两个方法，负责单结果与 Future
    包装，因此它的 Future 路径不经过基类的列表下标操作。

    Attributes:
        vllm_config: 当前 Engine 的完整配置引用，编译统计会在预热后写回该对象。
        is_sleeping: 休眠协议的状态标记，构造完成时为 False；不表示 KV 已初始化。
        sleeping_tags: 休眠协议中尚待恢复的资源名集合，初始为空。当前保留
            weights / kv_cache 的状态更新逻辑，但 GPU Worker 尚未实现休眠接口。
    """
    # Executor 的 PP 能力声明；当前执行与结果收集均未接入 pipeline parallelism。
    supports_pp: bool = False
    @staticmethod
    def get_class(vllm_config: NanovllmConfig) -> type[Executor]:
        """根据已支持的并行范围，返回具体 Executor 类但不创建其实例。

        Args:
            vllm_config: 当前完整配置，使用 parallel_config.tensor_parallel_size
                判断能否选择单进程实现。配置对象可以表达大于 1 的 TP size，
                但当前没有对应的多进程 Executor。

        Returns:
            executor_class: UniProcExecutor 类；Worker 创建与模型加载发生在随后构造实例时。

        Raises:
            NotImplementedError: tensor_parallel_size 不为 1。
        """
        # 在创建 Worker 和 CUDA 资源前拒绝当前执行器不能承载的 TP 配置。
        if vllm_config.parallel_config.tensor_parallel_size != 1:
            raise NotImplementedError(
                "Tensor parallel execution is not implemented yet"
            )

        # 延迟导入具体子类，避免 abstract 与 uniproc_executor 在模块加载时循环依赖。
        from nanovllm.executor.uniproc_executor import UniProcExecutor
        return UniProcExecutor
    def __init__(self, vllm_config: NanovllmConfig) -> None:
        """保存配置、创建并加载 Worker，随后初始化预留的休眠状态。

        _init_executor() 由具体子类实现，当前完成 Wrapper / Worker 创建、
        设备初始化和权重加载。显存 profiling、正式 KV 分配和启动预热仍由
        调用方在构造完成后逐步触发。

        Args:
            vllm_config: 当前完整配置，直接保存引用，不复制子配置。

        Side Effects:
            具体子类创建 Worker、设备和通信环境并加载权重。构造异常继续
            向上传播；成功后设置 is_sleeping=False、sleeping_tags=set()。
        """
        # Worker 创建与后续 RPC 均使用同一配置；预热统计也写回此对象。
        self.vllm_config=vllm_config
        # 由具体子类完成资源创建；返回时权重已加载，正式 KV cache 仍待规划。
        self._init_executor()
        # 初始未休眠是控制状态，不代表已经分配 KV 或完成预热。
        self.is_sleeping = False

        # 预留部分唤醒时的资源集合；只有 sleep() 的 Worker 调用成功后才填入标签。
        self.sleeping_tags: set[str] = set()
    @abstractmethod
    def _init_executor(self)->None:
        """由具体子类创建 Worker，初始化设备并加载权重。

        构造 Executor 时调用，完成后应能查询完整 KVCacheSpec 和执行显存
        profiling。正式 KV 的容量取决于 Engine 规划，此阶段不分配最终 KV。

        Side Effects:
            由子类保存 Worker 端点，并创建相应的设备、通信环境和模型资源。
        """
        raise NotImplementedError
    def initialize_from_config(
        self,
        kv_cache_configs:list[KVCacheConfig],
    )->None:
        """下发正式 KV 分配方案，完成预热并汇总模型级编译时间。

        在权重加载、显存 profiling 和 Engine KV 规划后调用。先将完整配置列表
        交给 Wrapper 按 global_rank 取出本地方案；初始化成功后，再触发
        compile_or_warm_up_model()。该顺序使正式 Graph 捕获使用最终 KV 地址。

        Args:
            kv_cache_configs: 按全局 Worker rank 排列的分配计划。单卡输入为
                [config0]，Wrapper 最终调用 worker.initialize_from_config(config0)。

        Side Effects:
            Worker 分配并绑定正式 KV，创建 Attention metadata Builder，预热
            模型执行与采样路径，并按配置捕获 Graph。最后回写
            compilation_config.compilation_time，单位为秒；该统计不等于整个
            预热或 Graph 捕获的墙钟时间。任一步骤失败均继续向上传播。
        """
        # 配置列表作为一个位置参数传给 Wrapper，由其按 global_rank 选择本地配置。
        self.collective_rpc(
            "initialize_from_config",
            args=(kv_cache_configs,),
        )
        # compilation_times 按 RPC rank 保存每个 Worker 的 CompilationTimes；
        # 先完成正式 KV 初始化，再执行内核、Graph 和 sampler 的启动准备。
        compilation_times: list[CompilationTimes]=self.collective_rpc("compile_or_warm_up_model")
        # compilation_times 非空时才回填统计；当前列表只有一个 Worker 的编译时间记录。
        if compilation_times:
            # t 是某个 Worker 的秒数统计；沿用 vLLM 取最大值，表达并行启动的编译耗时。
            self.vllm_config.compilation_config.compilation_time = max(
                t.language_model for t in compilation_times
            )
            
    def register_failure_callback(
        self,
        callback: FailureCallback,
    ) -> None:
        """保留不可恢复故障回调的注册入口，当前默认实现不保存回调。

        单进程执行中的异常通过调用或 Future.result() 返回调用方，没有后台
        故障通知线程。后续其他 Executor 可覆盖本方法保存并触发回调。

        Args:
            callback: 出现不可恢复故障时应被调用的无参函数；当前不会执行它。
        """
        # 仅丢弃本次形参引用，不建立回调注册表或额外监控资源。
        del callback
    def determine_available_memory(self) -> list[int]:
        """触发各 Worker 的显存测量，并返回按 rank 排列的候选 KV 字节预算。

        在模型加载后、正式 KV 分配前调用。具体 profiling 由 GPU Worker
        组织，Executor 仅收集结果；Engine 再结合 KVCacheSpec 将预算换算为
        KVCacheConfig，不在此接口内选择或分配物理 block。

        Returns:
            available_memory: list[int]，外层下标对应 Worker rank，元素单位为字节。当前单卡为
                [rank0_budget]，与 get_kv_cache_specs() 返回列表的同一项配对。

        Side Effects:
            Worker 执行模拟模型、采样和可能的 Graph profiling，更新显存统计。
            Executor 不再次计算预算，测量失败时异常继续向上传播。
        """
        # available_memory 保留多 Worker 的列表协议；单卡也不提前拆掉这一层列表。
        available_memory: list[int] = self.collective_rpc(
            "determine_available_memory"
        )
        return available_memory
    def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
        """收集各 Worker 已加载 Attention 层的 KV 存储规格。

        查询发生在 load_model() 完成后，不分配 KV tensor。Executor 的复数
        接口调用 Worker 的单数 get_kv_cache_spec()，并在外层保留 rank 维度。

        Returns:
            kv_cache_specs: 每个元素都是 layer_name -> KVCacheSpec 字典，描述本地模型层的
                block size、KV head 数、head dimension 与 dtype。单卡形如
                [{"model.layers.0.self_attn.attn": spec, ...}]，与显存预算列表对齐。
        """
        # kv_cache_specs 的外层下标为 Worker rank，内部 key 为模型层稳定全名。
        kv_cache_specs: list[dict[str, KVCacheSpec]] = self.collective_rpc(
            "get_kv_cache_spec"
        )
        return kv_cache_specs
    @overload
    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        non_block: Literal[False] = False,
    ) -> list[_R]:
        """声明同步 RPC 的类型：按 Worker rank 返回 list[_R]。

        Returns:
            outputs: 按 Worker rank 排列的同步 RPC 返回值列表。
        """
    @overload
    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        non_block: Literal[True] = True,
    ) -> Future[list[_R]]:
        """声明 Future 形式的 RPC 类型：Future 的结果为整个 rank 有序列表。

        Returns:
            future: 异步调用包装器，其 result() 为按 Worker rank 排列的完整结果列表。
        """
    
    @abstractmethod
    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        non_block: bool = False,
    ) :
        """由具体 Executor 调用 Worker，并组织普通 RPC 的 rank 有序结果。

        此抽象接口约定同步返回 list[_R]，Future 形式返回 Future[list[_R]]。
        UniProcExecutor 额外提供 single_value 参数，供它自己的执行入口直接
        获得单结果；普通 KV 查询仍遵守本接口的外层列表约定。

        Args:
            method: Worker 方法名，或接收执行端点为首个参数的 callable。当前
                UniProc 使用 WorkerWrapperBase 作为端点，属性查询再转发给 Worker。
            timeout: 预留的 RPC 等待上限，单位为秒，None 表示不指定上限。
                当前单进程实现不使用该值，不能据此中断已经开始的同步调用。
            args: 目标方法的位置参数；callable 的端点参数由调用机制额外传入。
            kwargs: 目标方法的关键字参数，None 表示使用空字典。
            non_block: 是否要求 Future 形式的返回值。实际执行是否在后台重叠
                由具体实现决定；当前 UniProc 在本线程调用后封装已完成的 Future。

        Returns:
            同步为按 Worker rank 排列的结果列表；Future 形式的 result() 为
            同样的结果列表。Future[list[_R]] 与 list[Future[_R]] 是不同协议。

        Notes:
            本接口组织控制命令与结果；具体执行器决定是否需要通信或序列化。
            当前 UniProc 直接调用本地对象，不建立额外的 RPC 连接。
        """
        raise NotImplementedError

    @overload
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        non_block: Literal[False] = False,
    ) -> ModelRunnerOutput | None:
        """声明同步执行结果：返回 ModelRunnerOutput 或表示待采样的 None。

        Returns:
            model_output: 同步执行的单个 ModelRunnerOutput，或表示仍需调用 sample_tokens() 的 None。
        """

    @overload
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        non_block: Literal[True] = True,
    ) -> Future[ModelRunnerOutput | None]:
        """声明子类可提供的单结果 Future；基类默认实现拒绝此调用方式。

        Returns:
            future: 结果为单个 ModelRunnerOutput 或 None 的 Future；由具体执行器提供该能力。
        """

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        non_block: bool = False,
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        """提供同步执行的默认实现，返回 rank 0 的输出或待采样标记。

        正式 KV 初始化和预热后调用。当前 UniProcExecutor 覆盖此方法，
        同时支持普通返回和单结果 Future；本方法仅为同步调用提供默认路径。

        Args:
            scheduler_output: 本轮请求变化、KV block 变化与逐请求 token 执行量。
            non_block: 是否请求 Future 返回；基类为 True 时直接报错，具体
                子类需在覆盖方法中完成对应包装。

        Returns:
            model_output: rank 0 的 ModelRunnerOutput 或 None。None 表示本轮 forward 已暂存
                logits，调用方应先调用 sample_tokens()，再提交下一轮执行。

        Raises:
            NotImplementedError: 对基类默认实现传入 non_block=True。

        Side Effects:
            委托 Worker 更新请求镜像；有调度 token 时准备输入、执行模型并写入
            KV。Executor 不维护第二份请求状态，Worker 执行异常继续向上传播。
        """
        # 基类的后续逻辑只处理结果列表；Future 的单结果适配交给具体子类覆盖。
        if non_block:
            raise NotImplementedError(
                f"{type(self).__name__} does not implement asynchronous "
                "execute_model"
            )

        # outputs 按 Worker rank 排列；本同步路径返回 outputs[0]，调用方命名为 model_output。
        outputs: list[ModelRunnerOutput | None] = self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
        )
        return outputs[0]
    @overload
    def sample_tokens(
        self,
        non_block: Literal[False] = False,
    ) -> ModelRunnerOutput:
        """声明同步采样类型：返回单个 ModelRunnerOutput。

        Returns:
            model_output: 当前批次采样得到的单个 ModelRunnerOutput。
        """
        ...

    @overload
    def sample_tokens(
        self,
        non_block: Literal[True],
    ) -> Future[ModelRunnerOutput]:
        """声明子类提供的 Future 采样类型：Future 的结果为单个输出对象。

        Returns:
            future: 结果为单个 ModelRunnerOutput 的 Future，由具体执行器提供。
        """
        ...

    def sample_tokens(
        self,
        non_block: bool = False,
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        """保留 vLLM 的采样默认路径，从同步 RPC 结果列表取 rank 0 的输出。

        在 execute_model() 返回 None 后调用。当前 UniProcExecutor 覆盖本方法，
        通过 single_value=True 直接返回单结果或单结果 Future；其正常执行路径
        不经过这里的列表下标操作。

        Args:
            non_block: 透传给 collective_rpc 的返回形式标记。本默认实现只适用
                于返回列表的同步路径，Future 返回需要具体子类覆盖适配。

        Returns:
            model_output: 同步列表中的 rank 0 ModelRunnerOutput。注解中的 Future 返回能力
                由具体子类提供，本方法不会将 Future[list[...]] 转换为单结果 Future。

        Side Effects:
            Worker 消费待采样 logits、过滤候选并回写 token 镜像，结果交给调用方
            更新 Scheduler。Executor 不缓存或重复消费采样状态。
        """
        # output 在同步路径中是按 rank 排列的列表，单卡时形如 [model_runner_output]。
        output = self.collective_rpc(
            "sample_tokens",
            non_block=non_block,
        )

        # 此取首项操作针对列表；当前 UniProc 的 Future 路径由它自己的同名方法处理。
        return output[0]
    def execute_dummy_batch(self) -> None:
        """保留向 Worker 提交额外 dummy batch 的接口，当前 GPU Worker 尚未实现。

        正常启动预热已由 initialize_from_config() 内的 compile_or_warm_up_model()
        完成。本方法只转发同名命令；当前 UniProc 调用会因目标方法不存在而抛出
        NotImplementedError，保留此入口不表示已具备额外 dummy 执行能力。
        """
        # 将预留命令交给统一 RPC 解析，当前不以其他 Worker 方法替代该命令。
        self.collective_rpc("execute_dummy_batch")
    def profile(
        self,
        is_start: bool = True,
        profile_prefix: str | None = None,
    ) -> None:
        """保留性能 profiler 的启动与停止入口，当前 GPU Worker 尚未实现。

        该接口用于性能 trace；显存预算测量仍通过 determine_available_memory()
        触发，后者不依赖本方法。当前 UniProc 找不到对应 Worker 方法时抛出
        NotImplementedError。

        Args:
            is_start: 透传的开始 / 停止标记，True 表示开始，False 表示停止。
            profile_prefix: 透传的可选输出文件前缀，具体含义留给 Worker 实现。
        """
        # 两个参数按 Worker 接口顺序作为位置参数传入，不在 Executor 内创建 profiler。
        self.collective_rpc(
            "profile",
            args=(is_start, profile_prefix),
        )
    @abstractmethod
    def check_health(self) -> None:
        """由具体 Executor 定义健康检查方式，异常通过抛出暴露。

        当前 UniProcExecutor 沿用空检查；同进程执行中的设备错误通过实际调用
        或 Future.result() 暴露，不在这里主动提交 GPU 探测任务。
        """
        raise NotImplementedError
    def shutdown(self) -> None:
        """通过 RPC 请求 Worker 关闭，具体清理范围由 Worker 决定。

        调用前应停止提交新任务。当前 UniProcExecutor 覆盖此方法，直接委托
        本地 Wrapper；Wrapper 再调用 GPU Worker 的 shutdown()。

        Side Effects:
            由 Worker 清理其运行资源引用；本方法不额外销毁全局进程组或 Graph
            pool，也不保证其他引用仍持有的存储立即释放。清理异常继续传播。
        """
        self.collective_rpc("shutdown")
    def sleep(self, level: int = 1) -> None:
        """保留资源休眠的 RPC 与状态更新流程，当前 GPU Worker 尚未实现。

        is_sleeping 已为 True 时只记录警告并返回；否则调用 Worker.sleep()，
        成功后才将两个资源标签写入 sleeping_tags，并将 is_sleeping 设为 True。
        当前 UniProc 的目标方法缺失，正常未休眠状态下调用会抛 NotImplementedError。

        Args:
            level: 透传给 Worker 的整数休眠等级，具体资源策略由 Worker 定义。

        Side Effects:
            Worker 调用成功后设置 sleeping_tags={"weights", "kv_cache"} 和
            is_sleeping=True。RPC 失败时不执行这两项状态回写。
        """
        # 重复休眠不再调用 Worker，避免把已有休眠状态当成一次新的资源释放。
        if self.is_sleeping:
            logger.warning("Executor is already sleeping.")
            return

        # time_before_sleep 是 RPC 前的单调时钟秒数，用于记录本次调用耗时。
        time_before_sleep = time.perf_counter()
        self.collective_rpc(
            "sleep",
            kwargs={"level": level},
        )
        # time_after_sleep 是成功返回后的时刻，是否同步 GPU 由 Worker 自己决定。
        time_after_sleep = time.perf_counter()

        # 只有 RPC 成功后，才把权重与 KV cache 登记为尚待恢复的资源。
        self.sleeping_tags = {"weights", "kv_cache"}
        # 存在尚待恢复的资源时，将 Executor 标记为休眠状态。
        self.is_sleeping = True

        logger.info(
            "It took %.6f seconds to fall asleep.",
            time_after_sleep - time_before_sleep,
        )
    def wake_up(
        self,
        tags: list[str] | None = None,
    ) -> None:
        """保留按资源标签唤醒 Worker 并更新休眠状态的流程。

        未休眠时只记录警告；提供 tags 时先验证其都位于 sleeping_tags 中，
        再透传给 Worker。当前 GPU Worker 尚未实现 wake_up()；只有 Worker
        调用成功后，Executor 才移除对应资源标签。

        Args:
            tags: 待恢复的资源名列表，调用方应提供无重复的已休眠标签。参数
                原样传给 Worker；本层对 None 或空列表采用清空全部标签的状态
                更新规则，对非空列表仅移除其中列出的标签。

        Side Effects:
            成功后更新 sleeping_tags；集合变空时设置 is_sleeping=False，仍有
            未恢复标签时保持休眠。未知标签只告警返回，RPC 失败时不更新集合。
        """
        # 未休眠时无需进入 Worker 唤醒流程，直接保留现有状态。
        if not self.is_sleeping:
            logger.warning("Executor is not sleeping.")
            return

        # 逐个检查本次要恢复的资源名，存在未知项时整次调用都不提交。
        if tags:
            # tag 是调用方要求恢复的单个资源名称，例如 weights 或 kv_cache。
            for tag in tags:
                if tag not in self.sleeping_tags:
                    logger.warning(
                        "Tag %s is not in sleeping tags %s.",
                        tag,
                        self.sleeping_tags,
                    )
                    return

        # 两个时刻仅统计唤醒 RPC 的耗时，单位为秒，不记录模型级编译时间。
        time_before_wakeup = time.perf_counter()
        self.collective_rpc(
            "wake_up",
            kwargs={"tags": tags},
        )
        # time_after_wakeup 是成功返回后的秒数，和调用前时刻相减用于耗时日志。
        time_after_wakeup = time.perf_counter()

        logger.info(
            "It took %.6f seconds to wake up tags %s.",
            time_after_wakeup - time_before_wakeup,
            tags if tags is not None else self.sleeping_tags,
        )

        # RPC 成功后再提交状态变化；非空列表只恢复指定资源。
        if tags:
            for tag in tags:
                self.sleeping_tags.remove(tag)
        else:
            # None 或空列表均清空剩余标签，表达本层的全部恢复状态。
            self.sleeping_tags.clear()

        # 所有已登记资源均恢复时解除休眠标记；部分恢复则继续保持 True。
        if not self.sleeping_tags:
            self.is_sleeping = False
    @classmethod
    def supports_async_scheduling(cls) -> bool:
        """声明是否支持调度与模型执行重叠，基类默认返回 False。

        Future 是调用结果的包装形式；是否能安全重叠后续调度，需要具体执行器
        与 Worker 共同支持，不能仅根据某个接口返回 Future 推导。

        Returns:
            supports_async_scheduling: 执行器与 Worker 支持安全重叠后续调度时为 True；基类默认 False。
        """
        return False
