"""定义 Worker 公共接口，并提供 Executor 使用的 Worker 包装层。

WorkerBase 保存共享配置、rank 和资源占位状态；具体硬件 Worker 实现设备初始化、
模型加载、启动预热及执行接口。WorkerWrapperBase 延迟构造该 Worker，按 rank
选择构造参数和 KV 分配方案，并将其余属性访问转发给真实 Worker。

本模块沿用 vLLM 的接口分工。当前 Executor 在同一进程内管理一个 GPU Worker，
Wrapper 自身不创建进程、不传输 RPC 消息；调用的组织与结果适配由 Executor 完成。
"""

from __future__ import annotations

from typing import Any,TypeVar,Callable,NamedTuple

import torch
from torch import nn

from nanovllm.kv_cache_interface import KVCacheSpec
from nanovllm.outputs import ModelRunnerOutput
from nanovllm.core.sched.output import SchedulerOutput
from nanovllm.config import NanovllmConfig, set_current_config
from nanovllm.kv_cache_interface import KVCacheConfig

# 将 apply_model() 的返回类型与调用方提供的回调返回类型关联起来。
_R = TypeVar("_R")

class CompilationTimes(NamedTuple):
    """返回 Worker 启动阶段累计的模型级编译时间，单位为秒。

    compile_or_warm_up_model() 使用此结果向调用方报告编译统计。当前只保留
    language_model，不包含多模态 encoder；数值来自配置中累计的编译时间，
    不统计整个预热方法的墙钟时间，也不自动累计局部 @torch.compile 的 JIT 时间。
    """

    # GPU Worker 传入 compilation_config.compilation_time；当前无模型级编译时为 0.0。
    language_model: float

class WorkerBase:
    """保存 Worker 公共状态，并约定硬件实现向 Executor 提供的执行接口。

    基类构造期只保存配置引用、校验 rank 并设置资源占位字段。具体 GPU Worker
    随后绑定设备、建立分布式环境和创建 GPUModelRunner；模型与 KV tensor 的
    分配由后续生命周期接口完成。

    需要硬件实现的方法以 NotImplementedError 占位；check_health() 和 shutdown()
    提供空默认实现。apply_model() 与 vocab_size 复用公共逻辑。以下接口说明描述
    具体 Worker 应遵守的调用约定，Executor 通过 WorkerWrapperBase 访问这些接口。

    正常启动先调用 init_device() 和 load_model()，再查询 KV 规格、测量显存并
    下发正式 KV 分配方案，最后执行 compile_or_warm_up_model()。GPU Worker 的
    显存测量和 KV 初始化入口由具体类提供，Wrapper 按同名方法转发。

    Attributes:
        vllm_config: 当前 Worker 使用的完整配置，与调用方共享对象引用。
        cache_config: KV 策略及运行时容量字段所在的配置对象。
        compilation_config: 模型构建、Graph 策略和编译统计所在的配置对象。
        device_config: 期望使用的设备描述；具体设备由 init_device() 绑定。
        model_config: 模型结构、精度、上下文上限及 enforce_eager 等配置。
        scheduler_config: 单轮 token budget 和请求槽位容量，预热也需遵守这些上限。
        local_rank: 本机可见设备编号，CUDA Worker 用它选择执行 GPU。
        rank: 默认分布式 world 中的全局 rank；当前只有 TP，单卡为 0。
        distributed_init_method: 建立默认 process group 的会合地址。
        is_driver_worker: 是否承担向 Engine 返回结果的 driver 职责，单卡为 True。
        device: 实际执行设备，构造完成时为 None，由具体 Worker 初始化。
        model_runner: 具体硬件的运行器，构造完成时为 None；无需继承 nn.Module。
    """

    def __init__(
        self,
        vllm_config: NanovllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        """保存配置引用、写入当前 rank，并初始化尚未分配资源的占位状态。

        此处不触发设备绑定、进程组创建或模型加载。配置别名均引用原有对象，
        例如具体 Worker 稍后回写 cache_config.num_gpu_blocks 时，调用方通过
        vllm_config.cache_config 也会看到该变化。

        Args:
            vllm_config: 当前 Worker 的完整配置；本方法不复制配置对象。
            local_rank: 本机可见 GPU 编号，设备合法性留给具体 init_device() 检查。
            rank: 当前 Worker 的全局 rank，必须落在配置 world_size 的范围内。
            distributed_init_method: 默认进程组的会合地址，例如本机 tcp:// 地址。
            is_driver_worker: 是否为 driver Worker；当前单卡 Executor 传入 True。

        Raises:
            ValueError: rank 不在 [0, parallel_config.world_size) 范围内。

        Side Effects:
            保存子配置别名，将 rank 写入共享的 vllm_config.parallel_config；
            设置 self.device=None 和 self.model_runner=None。
        """
        # 保留完整配置引用，供具体 Worker 查询未单独保存别名的子配置。
        self.vllm_config = vllm_config
        # 缓存配置既包含预算策略，也会在正式 KV 初始化时回填物理 block 总数。
        self.cache_config = vllm_config.cache_config
        # 编译配置同时保存 Graph 策略、静态 Attention 注册表和编译耗时。
        self.compilation_config = vllm_config.compilation_config
        # 期望设备可能是不带编号的 cuda；具体 CUDA device 在 init_device() 中解析。
        self.device_config = vllm_config.device_config
        # 模型配置提供 dtype、上下文上限与 enforce_eager 等执行约束。
        self.model_config = vllm_config.model_config
        # 预热与真实调度共用容量上限，避免 dummy 输入超出 Runner 的固定缓冲区。
        self.scheduler_config = vllm_config.scheduler_config
        # parallel_config 是共享并行配置；其 world_size 决定合法的全局 rank 范围。
        parallel_config = vllm_config.parallel_config
        # 在写入配置前拒绝越界 rank，防止模型层据错误的 rank 选择权重分片。
        if not 0 <= rank < parallel_config.world_size:
            raise ValueError(
                "Worker rank must be in [0, parallel_config.world_size), got "
                f"rank={rank}, world_size={parallel_config.world_size}."
            )
        # 将当前执行身份写回配置，后续模型构造与运行器读取同一份 rank 信息。
        parallel_config.rank = rank

        # local_rank 使用当前进程可见的设备编号，受 CUDA_VISIBLE_DEVICES 映射影响。
        self.local_rank = local_rank
        # rank 表示通信身份；它与本地设备编号承担不同职责。
        self.rank = rank
        # 保存 Executor 提供的会合地址，延迟到 init_device() 时创建进程组。
        self.distributed_init_method = distributed_init_method
        # 保存 driver 身份；该字段本身不执行结果过滤，返回协议由具体实现决定。
        self.is_driver_worker = is_driver_worker

        # None 表示尚未绑定实际执行设备；具体 Worker 在 init_device() 中赋值。
        self.device: torch.device | None = None
        # None 表示尚未创建运行器；GPUModelRunner 本身无需是 nn.Module。
        self.model_runner: Any = None

    def init_device(self) -> None:
        """建立设备与通信环境，并创建后续加载模型所需的运行器。

        由 Executor 在 Worker 构造后、load_model() 前调用。当前 GPU Worker
        依次绑定 CUDA device、检查 dtype、建立 world / TP group、采集加载前
        的显存基线，最后创建 GPUModelRunner。

        Side Effects:
            具体实现设置 device 和 model_runner，并创建相应设备与通信状态；
            模型权重由随后的 load_model() 单独加载。
        """
        raise NotImplementedError
    def load_model(self,*, load_dummy_weights: bool = False) -> None:
        """在设备初始化完成后，构造模型并加载权重。

        完成后应能查询 KVCacheSpec 并执行显存 profiling。正式 KV cache 的
        容量依赖后续显存规划，因此仍由 KV 初始化入口创建。

        Args:
            load_dummy_weights: 是否请求 dummy 权重加载，保留 vLLM 的同名接口；
                实际加载策略由具体 Worker / Runner 决定，正常启动使用 False。

        Side Effects:
            具体实现创建模型、加载权重、登记静态 Attention 层，并按配置包装
            模型执行入口；GPU Worker 同时记录模型加载的显存增量。
        """
        raise NotImplementedError
    def get_model(self) -> nn.Module:
        """返回已加载的原始语言模型，供模型检查或 apply_model() 使用。

        在 load_model() 完成后调用。具体 Worker 负责绕过 CUDA Graph 等执行
        Wrapper，使调用方可访问原模型的 module 和参数。

        Returns:
            原始 nn.Module 引用，不复制权重。跨 Worker 的模型操作可通过
            apply_model() 在模型所在 Worker 内执行。
        """
        raise NotImplementedError
    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """查询已加载模型的 KV 存储规格，供 Engine 生成正式分配方案。

        load_model() 构造的静态 Attention 层提供各自的 KVCacheSpec。Engine
        结合这些规格与可用显存字节数规划 KVCacheConfig，再下发给 Worker。

        Returns:
            Attention layer 稳定全名到 KVCacheSpec 的映射，例如 Qwen3 的
            "model.layers.0.self_attn.attn"。规格描述单层存储布局和页大小，
            查询本身不分配 KV tensor，也不决定最终物理 block 数。
        """
        raise NotImplementedError
    def apply_model(
        self,
        fn: Callable[[nn.Module], _R],
    ) -> _R:
        """将原始模型交给回调，并原样返回回调结果。

        通过 Executor 调用此接口时，fn 在模型所在的 Worker 内执行，无需将
        整个模型对象传回调用方。调用前应已完成 load_model()。

        Args:
            fn: 接收原始 nn.Module 引用的回调，返回类型由调用方决定。

        Returns:
            fn 的返回值，保留其类型与对象语义。

        Side Effects:
            由回调决定；fn 可以检查或修改原模型，修改直接作用于 Worker 持有
            的对象。本方法不复制模型，模型查询及回调异常均继续向上传播。
        """
        # 先由具体 Worker 解包模型，再将模型本体交给 fn，避免回调依赖运行器结构。
        return fn(self.get_model())
    def compile_or_warm_up_model(self) -> CompilationTimes:
        """在正式 KV 分配完成后准备执行内核、Graph 与采样路径。

        此时模型、正式 KV tensor 和 Attention metadata Builder 已就绪，
        InputBatch 尚无真实请求。具体 GPU Worker 编排内核预热、正式 Graph
        捕获与 sampler 预热；enforce_eager=True 时仍执行普通预热。

        Returns:
            CompilationTimes，单位为秒。当前 language_model 为模型级编译统计，
            无对应编译流程时返回 0.0，不代表本次预热或 Graph 捕获的总耗时。

        Side Effects:
            具体实现可触发 JIT、更新固定输入缓冲区、捕获 Graph，并调整进程级
            GC 策略。模拟采样可以消耗 RNG 状态，但不提交真实请求的生成结果。
        """
        raise NotImplementedError
    @property
    def vocab_size(self) -> int:
        """从模型配置查询全局词表大小，无需先加载模型权重。

        Returns:
            Hugging Face 配置中声明的正整数 vocab_size，不包含模型层为计算
            布局额外添加的 padding，也不表示某个 TP rank 的本地词表行数。

        Raises:
            ValueError: ModelConfig 未找到有效的正整数 vocab_size。
        """
        return self.vllm_config.model_config.get_vocab_size()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput | None:
        """消费一轮调度计划，并按返回值指示是否需要继续采样。

        正式 KV 初始化和预热完成后调用；返回 None 时，调用方必须先调用
        sample_tokens() 消费本轮状态，再提交下一轮 execute_model()。

        Args:
            scheduler_output: 本轮 Scheduler 快照，包含请求增删、KV block 变化
                及每请求执行 token 数。输入准备和模式分派由具体 Worker 完成。

        Returns:
            ModelRunnerOutput 表示该轮已有可返回的结果，None 表示需要调用
            sample_tokens()。当前 GPU Worker 对空调度返回空输出，对成功的
            forward 暂存 logits 并返回 None。

        Side Effects:
            具体实现同步 Worker 请求镜像、准备模型输入并执行 forward。有 token
            执行时写入 KV 并保存待采样状态；当前 Runner 会拒绝覆盖未消费的状态。
        """
        raise NotImplementedError
    def sample_tokens(self) -> ModelRunnerOutput:
        """消费 execute_model() 返回 None 时暂存的 logits，完成本轮采样。

        每次待采样状态只消费一次。当前 GPU Worker 执行 temperature 采样，
        过滤未完成 prefill 的候选，并通过同步 CPU 结果返回有效 token。

        Returns:
            ModelRunnerOutput，包含请求 ID、请求行映射和有效 sampled token IDs。

        Raises:
            RuntimeError: 具体 GPU Worker 没有待采样状态，例如未 forward 或重复
                消费同一轮结果。

        Side Effects:
            具体实现清空临时执行状态、回写 Worker 侧有效 token 镜像。Scheduler
            的权威请求进度仍由上层根据返回结果更新。
        """
        raise NotImplementedError
    def check_health(self) -> None:
        """提供可由具体 Worker 覆盖的健康检查入口。

        默认直接返回，当前单进程 GPU Worker 沿用该行为，设备错误由实际执行
        接口暴露。若后续需要主动探测通信或设备状态，可在具体实现中检查并抛出异常。
        """
        return

    def shutdown(self) -> None:
        """提供由具体 Worker 决定清理范围的关闭入口，基类默认直接返回。

        当前 GPU Worker 先恢复 Python GC 扫描范围，再委托 Runner 清理模型执行
        资源。进程组与其他进程级资源的销毁范围由具体实现及外层生命周期决定，
        此基类不要求每个 Worker 都销毁默认 process group 或全局 Graph pool。

        Side Effects:
            基类无副作用；具体 Worker 可以释放其资源引用，清理异常由调用方处理。
        """
        return


class WorkerWrapperBase:
    """为 Executor 管理一个延迟创建的真实 Worker，并适配 rank 对应的输入。

    Executor 先创建 Wrapper，再通过 init_worker() 构造 Worker。Wrapper 在
    init_worker()、init_device() 和 initialize_from_config() 的作用域内安装
    当前配置，并在正常或异常退出时恢复外层配置。

    其余属性通过 __getattr__() 访问真实 Worker。例如 load_model() 使用
    GPU Worker 自身提供的配置上下文；普通转发不会自动给所有方法增加上下文。
    当前 UniProcExecutor 与 Worker 位于同一进程，调用关系为：

    .. code-block:: text

        LLMEngine -> UniProcExecutor -> WorkerWrapperBase -> Worker

    rpc_rank 用于从 all_kwargs 选择构造参数，global_rank 用于从
    kv_cache_configs 选择 KV 分配方案。当前单卡时两者都为 0；保留两个字段，
    以表达 Executor 内部编号与默认分布式 world 编号各自的含义。

    Attributes:
        rpc_rank: Executor 内部 Worker 编号，对应 all_kwargs 的下标。
        global_rank: Worker 的全局 rank，对应 kv_cache_configs 的下标。
        vllm_config: 从构造参数取出的配置引用，初始为 None；init_worker() 在
            构造 Worker 前写入，供后续显式生命周期接口安装配置上下文。
        worker: 真实 Worker 引用，初始为 None；Worker 构造成功后才赋值。
    """
    def __init__(
        self,
        rpc_rank: int = 0,
        global_rank: int | None = None,
    ) -> None:
        """保存两个 rank，创建尚未绑定 Worker 和配置的包装对象。

        Args:
            rpc_rank: 当前 Executor 内部编号，调用方需保证能索引到对应构造参数。
            global_rank: 默认分布式 world 中的编号；为 None 时使用 rpc_rank。
                两者当前均由单卡 Executor 设为 0。

        Side Effects:
            设置 rank 字段及 worker、vllm_config 的 None 占位；不创建真实 Worker，
            不绑定设备，也不创建子进程或通信连接。
        """
        # rpc_rank 只用于 Executor 侧参数列表定位，不代替 Worker 的实际通信 rank。
        self.rpc_rank = rpc_rank
        # 显式 global_rank 优先；只有 None 才回退，合法的 global_rank=0 会被保留。
        self.global_rank = rpc_rank if global_rank is None else global_rank

        # None 表示尚未构造成功，shutdown() 据此允许初始化前或构造失败后的空清理。
        self.worker: WorkerBase | None = None
        # 构造函数不接收配置，init_worker() 从对应的 kwargs 中取出并保存引用。
        self.vllm_config: NanovllmConfig | None = None

    def shutdown(self) -> None:
        """在真实 Worker 已创建时，将关闭请求转交给它。

        worker 为 None 时直接结束，允许在 init_worker() 前或构造失败后调用。
        Wrapper 不重置成员引用，资源清理范围及重复关闭行为由具体 Worker 决定。

        Side Effects:
            调用已创建 Worker 的 shutdown()；Worker 的清理异常继续向上传播。
        """
        # 只有构造成功的 Worker 才有可委托的关闭入口。
        if self.worker is not None:
            self.worker.shutdown()
    
    def init_worker(
        self,
        all_kwargs: list[dict[str, Any]],
    ) -> None:
        """按 rpc_rank 选取构造参数，在当前配置作用域内创建 GPU Worker。

        这是正常启动时一次性的构造入口。单卡时 all_kwargs=[kwargs0]，
        rpc_rank=0 取出 kwargs0；Worker 构造完成后，其 device 和 model_runner
        仍为 None，后续由 init_device() 和 load_model() 逐步创建运行资源。

        Args:
            all_kwargs: 按 rpc rank 排列的 Worker 构造参数字典列表。选中的字典
                至少包含 vllm_config、local_rank、rank、distributed_init_method；
                is_driver_worker 可使用 Worker 构造函数的默认值。

        Raises:
            IndexError: rpc_rank 超出 all_kwargs 的索引范围。
            AssertionError: 选中的参数字典缺少 vllm_config，或其值为 None。

        Side Effects:
            先保存 self.vllm_config，再在构造成功后保存 self.worker；GPU Worker
            的基类构造还会回写配置中的 rank。构造异常向上传播，配置上下文仍恢复。
        """
        # kwargs 是当前 RPC rank 的构造参数浅副本；配置等字典值仍与调用方共享引用。
        # 保留外层字典副本，避免构造流程调整参数字典时影响 Executor 保存的原始项。
        kwargs = all_kwargs[self.rpc_rank].copy()

        # vllm_config 从参数中取得，既要传给 Worker，也要供 Wrapper 后续安装上下文。
        vllm_config:NanovllmConfig|None = kwargs.get("vllm_config")
        # 当前配置是 Worker 生命周期接口的前提；缺失时在构造 Worker 之前报错。
        assert vllm_config is not None, (
            "vllm_config is required to initialize the worker"
        )
        # 保存同一个配置引用，不从 kwargs 中移除，Worker(**kwargs) 仍接收该对象。
        self.vllm_config = vllm_config
        # 延迟导入 GPU 实现，避免 gpu_worker 继承 WorkerBase 时形成模块循环依赖。
        from nanovllm.worker.gpu_worker import Worker
        # 沿用 vLLM 的构造上下文，使 get_current_config() 可读取本实例配置；
        # 退出时恢复此前配置，当前 Worker 构造只保存公共状态，设备与权重稍后创建。
        with set_current_config(self.vllm_config):
            # 右侧构造完整成功后才写入 self.worker，构造失败时仍保留 None 占位。
            self.worker = Worker(**kwargs)

    def initialize_from_config(self, kv_cache_configs: list[KVCacheConfig]) -> None:
        """将按 global rank 排列的 KV 分配列表转换为当前 Worker 的单份方案。

        调用前应已创建 Worker、加载模型，并完成 Engine 侧的显存规划。单卡
        示例为 kv_cache_configs=[config0]、global_rank=0，实际执行
        worker.initialize_from_config(config0)，不把整个列表传给 GPU Worker。

        Args:
            kv_cache_configs: 按全局 rank 排列的 KVCacheConfig 列表，调用方需保证
                包含当前 global_rank。Wrapper 选择已有方案，不重新计算容量。

        Raises:
            IndexError: global_rank 超出 kv_cache_configs 的索引范围。
            AssertionError: init_worker() 尚未保存配置。

        Side Effects:
            具体 GPU Worker 回写 block 容量，并委托 Runner 分配和绑定正式 KV、
            初始化 Attention Builder 与 Graph keys；退出时恢复外层配置上下文。
        """
        # kv_cache_config 是当前全局 rank 对应的单份计划；此处不复制或修改该方案。
        kv_cache_config = kv_cache_configs[self.global_rank]
        # 配置在 init_worker() 中保存，未完成该前置步骤时不能建立初始化作用域。
        assert self.vllm_config is not None
        # 后端与 metadata Builder 的构造可能读取当前配置，初始化期间显式安装它。
        with set_current_config(self.vllm_config):
            # 分配细节由硬件 Worker 负责；Wrapper 只适配列表输入并维护上下文。
            self.worker.initialize_from_config(kv_cache_config)  # type: ignore

    def init_device(self)->None:
        """在当前配置作用域内，调用真实 Worker 的设备初始化入口。

        init_worker() 成功后、load_model() 前调用；Wrapper 不自行解析 GPU 编号
        或创建进程组，执行顺序由具体 GPU Worker 的 init_device() 保证。

        Raises:
            AssertionError: init_worker() 尚未保存配置。

        Side Effects:
            GPU Worker 绑定 CUDA device、初始化分布式环境、保存显存基线并创建
            Runner；模型权重仍由后续 load_model() 加载。退出时恢复外层配置。
        """
        # Worker 构造成功是前置约定，此断言只检查配置引用已由 init_worker() 保存。
        assert self.vllm_config is not None
        # 为设备和运行器初始化提供与 Worker 构造时相同的配置上下文。
        with set_current_config(self.vllm_config):
            self.worker.init_device()

    def __getattr__(self, attr: str):
        """在 Wrapper 上找不到属性时，返回真实 Worker 的同名属性。

        Python 先查找 Wrapper 自己的方法和字段，找不到才进入此处。因此
        init_device()、initialize_from_config() 使用 Wrapper 的显式适配，
        get_kv_cache_spec()、execute_model()、sample_tokens() 等访问由此转发。

        对方法名返回绑定到真实 Worker 的方法对象，调用方随后执行该方法；
        本函数不立即调用方法，不缓存属性，也不额外安装配置上下文。

        Args:
            attr: Wrapper 上未找到的属性名，由 Python 属性查找机制传入。

        Returns:
            真实 Worker 的同名属性值或绑定方法，保持原有对象引用。

        Raises:
            AttributeError: 底层对象不存在该属性；Worker 尚未创建时，通常表现为
                从 None 上查询 Worker 方法失败。
        """
        # 所有普通转发共用这一个入口；缺失属性的异常交由外层调用机制处理。
        return getattr(self.worker, attr)
