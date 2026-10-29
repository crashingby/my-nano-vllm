"""nano-vLLM 的完整推理配置及同步初始化期间的配置作用域。

调用方组装 NanovllmConfig 并直接传入 LLM / LLMEngine；Engine 到执行端共享
该对象。set_current_config() 为初始化过程中无法显式传参的模型层提供配置引用，
每轮 forward 的 Attention metadata 和 Graph key 则由 ForwardContext 管理。
"""

from __future__ import annotations

from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import field
from typing import TypeVar

from pydantic import ConfigDict

from nanovllm.config.cache import CacheConfig
from nanovllm.config.compilation import CUDAGraphMode, CompilationConfig
from nanovllm.config.device import DeviceConfig
from nanovllm.config.kernel import KernelConfig
from nanovllm.config.load import LoadConfig
from nanovllm.config.model import ModelConfig
from nanovllm.config.parallel import ParallelConfig
from nanovllm.config.scheduler import SchedulerConfig
from nanovllm.config.utils import config
from nanovllm.logger import init_logger
from nanovllm.model_executor.layers.quantization import QuantizationConfig

logger = init_logger(__name__)


# 聚合配置允许子配置持有 torch 和 Transformers 对象，并保留原实例引用。
@config(config=ConfigDict(arbitrary_types_allowed=True), slots=True)
class NanovllmConfig:
    """聚合一次 nano-vLLM Engine 实例所需的全部推理配置。

    调用方先创建各领域的子配置，再以 LLM(vllm_config) 或
    LLMEngine(vllm_config) 启动推理。Engine 不重新解析扁平参数，不复制或重建
    这些子配置；Executor、Worker 和 Runner 沿执行链共享同一总配置引用。

    本类构造时检查 Scheduler 容量，解析 Graph 默认模式和候选尺寸；后端能力
    限制由 Runner 在 Attention 初始化时解析。启动期间还会回写 EOS、实际 KV
    block 数和编译统计，并登记实际 Attention module，因此每个 Engine 应使用
    独立配置。省略的子配置由 default_factory 为本实例创建，不跨实例共享。

    Attributes:
        model_config: 模型路径、Hugging Face 结构配置、上下文长度及执行策略。
        scheduler_config: 单轮 token 预算及持久 batch 的 request 容量。
        eos: 初始 -1 表示尚未加载 tokenizer；Engine 加载后会用实际 EOS token ID
            覆盖该字段，供请求结束判断和 Scheduler 使用。
        parallel_config: TP size、派生 world_size 和 Worker 回写的 rank；当前
            支持同进程单 Worker 或单机多进程 TP。
        cache_config: KV 分配策略及正式初始化后的容量镜像，真实分配方案由
            Engine 根据 Worker 规格和显存预算生成。
        compilation_config: CUDA Graph 策略、捕获尺寸、编译统计及静态 module 表；
            当前 FA2 路径支持 FULL decode 与其他 batch 的普通执行。
        load_config: checkpoint 加载格式、加载进度显示及可选目标设备。
        device_config: 模型执行设备描述；具体 GPU 由 Worker 按 local_rank 绑定。
        kernel_config: 各模型层在构造时绑定的算子实现，例如 SiLU × Mul 的后端。
        quant_config: 可选的线性层权重方法配置，None 使用普通精度；当前没有
            内置量化算法或按名称解析配置的入口，须传入具体子类实例。
    """

    # 模型配置必须显式提供，因为其中的 model 路径没有合理默认值。
    model_config: ModelConfig
    # 未显式提供时创建本实例独立的 Scheduler 容量配置。
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    # tokenizer 加载前的占位值；Engine 随后覆盖它，并在输出处理时判断 EOS。
    eos: int = -1
    # 并行描述含 TP size、world_size 与 rank；是否能执行由 Executor 选择阶段检查。
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    # 保存用户分配策略，正式规划后也通过该子配置回写实际 block 总数。
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    # Graph 策略与静态 module 注册表；不保存请求级运行时状态。
    compilation_config: CompilationConfig = field(
        default_factory=CompilationConfig,
    )
    # 每个 Engine 持有独立加载配置，避免修改加载方式时影响其他实例。
    load_config: LoadConfig = field(default_factory=LoadConfig)
    # 每个 Engine 独立持有设备配置，配置构造不会探测或初始化 CUDA。
    device_config: DeviceConfig = field(default_factory=DeviceConfig)
    # 后端选择随 Engine 独立保存，模型层读取后绑定函数，不改全局 IR 默认策略。
    kernel_config: KernelConfig = field(default_factory=KernelConfig)
    # 当前仅向 Qwen 的线性层传递；embedding、LMHead 和 KV cache 仍保留普通精度。
    quant_config: QuantizationConfig | None = None

    def __post_init__(self) -> None:
        """检查模型分片与调度容量，并完成后端无关的 Graph 配置解析。

        先检查 Scheduler 的 token budget 和请求容量，再处理 enforce_eager 与
        默认 Graph 策略，最后推导尺寸。例如未显式指定策略且未强制 eager 时，
        这里选择 FULL_DECODE_ONLY；实际可用的完整捕获能力稍后由 Runner 检查。

        Side Effects:
            调用 ``SchedulerConfig`` 校验其 token/request 容量能否与当前模型上下文
            长度共同使用；随后直接修改持有的 CompilationConfig，回写模式、
            捕获尺寸和最大捕获尺寸。enforce_eager=True 时将这些值设为
            NONE、[] 和 0，即使调用方原先显式配置了 Graph 模式或尺寸。
            不探测 GPU、加载权重或执行 Graph 捕获。

        Raises:
            ValueError: 模型维度不能按 TP 分片、Scheduler 容量关系不成立，
                或显式最大尺寸与尺寸列表冲突。
            AssertionError: 启用 Graph 时尺寸上限小于 1，或显式尺寸列表为空。
        """
        # 模型结构与 TP 描述都已确定，先在父进程拒绝不支持的分片方案。
        self.model_config.verify_with_parallel_config(self.parallel_config)
        # ModelConfig 已先校验并收紧 max_model_len；Scheduler 仅处理批处理容量关系。
        self.scheduler_config.verify_with_model_config(
            self.model_config.max_model_len,
        )
        # eager 是用户明确的禁用策略，优先于默认或显式配置的 Graph 模式。
        if self.model_config.enforce_eager:
            self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        # 默认使用已接入的纯 decode FULL；后端能力解析随后决定最终可用模式。
        elif self.compilation_config.cudagraph_mode is None:
            self.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        # 模式先完成默认值解析，尺寸初始化才能区分启用与禁用状态。
        self._set_cudagraph_sizes()

    def _set_cudagraph_sizes(self) -> None:
        """结合 Scheduler 容量推导并规范化 CUDA Graph 捕获尺寸。

        沿用 vLLM 的同名流程。默认捕获上限为 min(max_num_seqs * 2, 512)，
        再受 max_num_batched_tokens 限制；当前无 spec decode，因此 decode
        query 长度固定为 1。候选尺寸为 1、2、4，随后在 256 之前按 8 递增，
        从 256 开始按 16 递增；预算本身落在上限以内时也加入候选列表。

        显式尺寸列表优先于默认规则：去重、排序并过滤超过 token 预算的尺寸。
        同时显式指定最大尺寸时，它必须等于过滤后的列表最大值。仅指定上限
        而未指定列表时，上限可以向下收紧到实际候选列表的最大值。
        列表应由调用方提供正整数；本方法不逐项校验类型和正值。显式列表非空
        但全部超过预算时，会过滤为空列表；未另外指定冲突上限时，有效最大尺寸
        随之置为 0。
        当前未接入 performance_mode 和 Sequence Parallel 对尺寸的调整分支。

        例如 max_num_seqs=10、max_num_batched_tokens=100，默认候选上限为 20，
        得到 [1, 2, 4, 8, 16]，最终最大捕获尺寸为 16。

        Side Effects:
            回写 cudagraph_capture_sizes 和 max_cudagraph_capture_size；禁用
            Graph 时分别设为 [] 和 0。最后调用 post_init_cudagraph_sizes()
            完成排序与最大值一致性检查；不生成 Graph keys 或捕获 CUDA Graph。
            FULL decode 的请求容量过滤稍后由 Dispatcher 初始化 keys 时执行。

        Raises:
            ValueError: 同时显式指定的最大尺寸与规范化列表最大值不一致。
            AssertionError: 启用 Graph 时候选上限小于 1，或原始显式尺寸列表为空；
                post_init_cudagraph_sizes() 也会检查非空列表与最大尺寸一致。
        """
        if (
            self.model_config is not None
            and not self.model_config.enforce_eager
            and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            # 先确定候选上限；显式列表会在后续分支覆盖默认候选规则。
            max_cudagraph_capture_size = (
                self.compilation_config.max_cudagraph_capture_size
            )
            if max_cudagraph_capture_size is None:
                max_cudagraph_capture_size = min(
                    self.scheduler_config.max_num_seqs * 2, 512
                )
            
            # token 输入 buffer 的容量也是所有候选捕获尺寸的硬上界。
            max_num_tokens = self.scheduler_config.max_num_batched_tokens
            max_cudagraph_capture_size = min(
                max_num_tokens, max_cudagraph_capture_size
            )
            assert max_cudagraph_capture_size >= 1, (
                "Maximum cudagraph size should be greater than or equal to 1 "
                "when using cuda graph."
            )
            # 显式尺寸仅按 Scheduler 预算过滤；不拿默认上限截断用户提供的列表。
            if self.compilation_config.cudagraph_capture_sizes is not None:
                assert len(self.compilation_config.cudagraph_capture_sizes) > 0, (
                    "cudagraph_capture_sizes should contain at least one element "
                    "when using cuda graph."
                )
                
                dedup_sizes = list(
                    set(self.compilation_config.cudagraph_capture_sizes)
                )
                cudagraph_capture_sizes = [
                    i for i in dedup_sizes if i <= max_num_tokens
                ]
                cudagraph_capture_sizes.sort()
            else:
                # 小尺寸单独保留，减少低并发 decode 的 padding 开销。
                cudagraph_capture_sizes = [
                    i for i in [1, 2, 4] if i <= max_cudagraph_capture_size
                ]
                if max_cudagraph_capture_size >= 8:
                    # 256 之前按 8 递增；range 上界排除 256，交给下一段处理。
                    cudagraph_capture_sizes += list(
                        range(8, min(max_cudagraph_capture_size + 1, 256), 8)
                    )
                if max_cudagraph_capture_size >= 256:
                    cudagraph_capture_sizes += list(
                        range(256, max_cudagraph_capture_size + 1, 16)
                    )
                # 预算落在候选上限内时，补捕获这个可能不在步长上的最大 batch。
                if (
                    max_num_tokens <= max_cudagraph_capture_size
                    and max_num_tokens not in cudagraph_capture_sizes
                ):
                    cudagraph_capture_sizes.append(max_num_tokens)
                cudagraph_capture_sizes = sorted(set(cudagraph_capture_sizes))

            # 有效上限由最终列表确定；过滤后为空时，沿用上游的 0 上限语义。
            valid_max_size = (
                cudagraph_capture_sizes[-1] if cudagraph_capture_sizes else 0
            )
            if (
                self.compilation_config.max_cudagraph_capture_size is not None
                and self.compilation_config.max_cudagraph_capture_size != valid_max_size
            ):
                # 两项均由用户指定时不猜测意图，直接报告配置冲突。
                if self.compilation_config.cudagraph_capture_sizes is not None:
                    raise ValueError(
                        "customized max_cudagraph_capture_size"
                        f"(={self.compilation_config.max_cudagraph_capture_size}) "
                        "should be consistent with the max value of "
                        f"cudagraph_capture_sizes(={valid_max_size})"
                    )
                logger.warning(
                    "Truncating max_cudagraph_capture_size to %d", valid_max_size
                )
            self.compilation_config.max_cudagraph_capture_size = valid_max_size

            # 去重和预算过滤会减少显式列表长度，记录最终覆盖结果供用户核对。
            if self.compilation_config.cudagraph_capture_sizes is not None and len(
                cudagraph_capture_sizes
            ) < len(self.compilation_config.cudagraph_capture_sizes):
                logger.warning(
                    "cudagraph_capture_sizes specified in compilation_config "
                    "%s is overridden by config %s",
                    self.compilation_config.cudagraph_capture_sizes,
                    cudagraph_capture_sizes,
                )
            self.compilation_config.cudagraph_capture_sizes = cudagraph_capture_sizes
        else:
            # 禁用时清除捕获计划，避免后续 Dispatcher 误用此前指定的尺寸。
            self.compilation_config.max_cudagraph_capture_size = 0
            self.compilation_config.cudagraph_capture_sizes = []

        # 两条分支均写入了具体列表，统一完成排序和最大尺寸的不变量检查。
        self.compilation_config.post_init_cudagraph_sizes()


# 同步初始化作用域使用进程级配置引用；请求级 forward 数据不保存在这里。
_current_vllm_config: NanovllmConfig | None = None


@contextmanager
def set_current_config(
    vllm_config: NanovllmConfig,
) -> Generator[None, None, None]:
    """在 Worker、模型或 Attention 初始化期间临时暴露完整配置。

    ``Attention`` 等底层 module 的构造函数无法逐层接收全部 Engine 配置时，可在
    此上下文内通过 :func:`get_current_config` 读取它，并向
    ``compilation_config.static_forward_context`` 注册自身。WorkerWrapper 也在
    Worker 构造、设备初始化和 KV 初始化期间安装这个作用域。
    退出时无条件恢复外层配置引用，支持嵌套和异常退出；作用域内对配置字段或
    注册表所做的修改会保留，不随全局引用恢复而回滚。

    Args:
        vllm_config: 当前初始化过程使用的总配置，保存原对象引用，不创建副本。

    Yields:
        空值。该 context manager 只提供隐式配置上下文。

    Side Effects:
        临时替换进程级 _current_vllm_config；正常和异常退出都会恢复先前引用。
    """
    global _current_vllm_config

    # 保存外层配置，使嵌套 context 或异常退出后仍能恢复先前构建环境。
    previous_vllm_config = _current_vllm_config
    _current_vllm_config = vllm_config
    try:
        yield
    finally:
        _current_vllm_config = previous_vllm_config


def get_current_config() -> NanovllmConfig:
    """取得最近一层 set_current_config() 安装的完整配置引用。

    Returns:
        vllm_config: 当前由 :func:`set_current_config` 安装的 ``NanovllmConfig``。

    Raises:
        AssertionError: 当前没有活动配置作用域时抛出。普通 forward 的运行时
            metadata 应读取 ForwardContext，不能假定本配置作用域始终存在。
    """
    if _current_vllm_config is None:
        raise AssertionError(
            "Current nano-vLLM vllm_config is not set. "
            "Call this function only while constructing a model inside "
            "set_current_config()."
        )
    return _current_vllm_config


def get_current_config_or_none() -> NanovllmConfig | None:
    """查询当前初始化作用域的配置；没有活动作用域时返回 None，不触发断言。

    Returns:
        vllm_config: 当前初始化作用域的完整配置引用；没有活动作用域时为 None。
    """
    return _current_vllm_config


T = TypeVar("T")


def get_layers_from_nanovllm_config(
    vllm_config: NanovllmConfig,
    layer_type: type[T],
    layer_names: Iterable[str] | None = None,
) -> dict[str, T]:
    """按类型和可选层名，从给定配置的静态注册表中筛选模型层。

    Runner 在查询 Attention 后端和 KV 规格时使用该接口。它显式读取参数中的
    总配置，不要求调用时存在 set_current_config() 作用域，也不读取请求 metadata。

    Args:
        vllm_config: 已完成模型构建、并已填充静态 module 注册表的总配置。
        layer_type: 要筛选的 Python module 类型，例如 ``Attention``。
        layer_names: 可选的层名子集。为 ``None`` 时遍历注册表中的所有层。

    Returns:
        layers: 新建的层名到 layer_type 实例映射，value 仍引用原始 module。名称不存在或
            类型不匹配的条目被跳过；不修改 static_forward_context，也不复制模型层。
    """
    # 该字典由 Attention 等 module 在模型构建期注册，Worker 只读地查询它。
    forward_context = vllm_config.compilation_config.static_forward_context
    # 不指定子集时保留注册顺序，便于得到稳定的层遍历顺序。
    if layer_names is None:
        layer_names = forward_context.keys()

    return {
        layer_name: layer
        for layer_name in layer_names
        if isinstance(layer := forward_context.get(layer_name), layer_type)
    }
