"""nano-vLLM 的总配置及模型构建期配置上下文。"""

from __future__ import annotations

from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TypeVar

from nanovllm.config.cache import CacheConfig
from nanovllm.config.compilation import CUDAGraphMode, CompilationConfig
from nanovllm.config.device import DeviceConfig
from nanovllm.config.load import LoadConfig
from nanovllm.config.model import ModelConfig
from nanovllm.config.parallel import ParallelConfig
from nanovllm.config.scheduler import SchedulerConfig
from nanovllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(slots=True)
class NanovllmConfig:
    """聚合一次 nano-vLLM Engine 实例所需的全部推理配置。

    ``ModelConfig`` 管理模型结构和执行策略，``SchedulerConfig`` 管理连续批处理
    容量，``LoadConfig`` 管理权重加载方式。并行策略、KV cache 策略与编译上下文
    分别由 ``ParallelConfig``、
    ``CacheConfig`` 与 ``CompilationConfig`` 持有。

    Attributes:
        model_config: 模型路径、Hugging Face 结构配置、上下文长度及执行策略。
        scheduler_config: 单轮 token 预算及持久 batch 的 request 容量。
        eos: tokenizer 初始化后由 Engine 写入的 EOS token ID。
        parallel_config: Tensor Parallel 等并行执行策略配置。
        cache_config: KV cache 的静态策略配置。
        compilation_config: CUDA Graph 策略、捕获尺寸及模型构建期的静态 module 表。
        load_config: checkpoint 加载格式、加载进度显示及可选目标设备。
        device_config: 模型执行设备描述；具体 GPU 由 Worker 按 local_rank 绑定。
    """

    # 模型配置必须显式提供，因为其中的 model 路径没有合理默认值。
    model_config: ModelConfig
    # Scheduler 容量拥有独立默认值，避免调用方每次创建它。
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    # Engine 在 tokenizer 加载后写入，Scheduler 用它判断请求自然结束。
    eos: int = -1
    # 并行策略拥有独立配置对象，当前只包含 TP size。
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    # KV cache 策略拥有独立配置对象，避免总配置继续堆积缓存细节。
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    # Graph 策略与静态 module 注册表；不保存请求级运行时状态。
    compilation_config: CompilationConfig = field(
        default_factory=CompilationConfig,
    )
    # 每个 Engine 持有独立加载配置，避免修改加载方式时影响其他实例。
    load_config: LoadConfig = field(default_factory=LoadConfig)
    # 每个 Engine 独立持有设备配置，配置构造不会探测或初始化 CUDA。
    device_config: DeviceConfig = field(default_factory=DeviceConfig)

    def __post_init__(self) -> None:
        """校验跨子配置约束，并解析 CUDA Graph 策略和捕获尺寸。

        Side Effects:
            调用 ``SchedulerConfig`` 校验其 token/request 容量能否与当前模型上下文
            长度共同使用；随后回写 Graph 配置模式、捕获尺寸和最大捕获尺寸。
            不探测 GPU、加载权重或执行 Graph 捕获。

        Raises:
            ValueError: 子配置之间的容量约束不成立时抛出。
        """
        # ModelConfig 已先校验并收紧 max_model_len；Scheduler 仅处理批处理容量关系。
        self.scheduler_config.verify_with_model_config(
            self.model_config.max_model_len,
        )
        # eager 是用户明确的禁用策略，优先于默认或显式配置的 Graph 模式。
        if self.model_config.enforce_eager:
            self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        # 当前先实现纯 decode FULL；后端能力解析随后由 Runner 决定最终可用模式。
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
        当前未接入 performance_mode 和 Sequence Parallel 对尺寸的调整分支。

        例如 max_num_seqs=10、max_num_batched_tokens=100，默认候选上限为 20，
        得到 [1, 2, 4, 8, 16]，最终最大捕获尺寸为 16。

        Side Effects:
            回写 cudagraph_capture_sizes 和 max_cudagraph_capture_size；禁用
            Graph 时分别设为 [] 和 0。最后调用 post_init_cudagraph_sizes()
            完成排序与不变量检查；不创建或捕获 CUDA Graph。

        Raises:
            ValueError: 启用 Graph 时上限非法、显式尺寸列表为空或包含非法尺寸，
                或同时指定的最大尺寸与规范化列表不一致。
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


# 模型构建发生在同步上下文中；当前 MVP 用进程内全局变量即可与 vLLM 的接口对齐。
_current_vllm_config: NanovllmConfig | None = None


@contextmanager
def set_current_config(
    vllm_config: NanovllmConfig,
) -> Generator[None, None, None]:
    """在模型构建期间临时暴露当前总配置。

    ``Attention`` 等底层 module 的构造函数无法逐层接收全部 Engine 配置时，可在
    此上下文内通过 :func:`get_current_config` 读取它，并向
    ``compilation_config.static_forward_context`` 注册自身。退出时会无条件恢复外层
    上下文，因此嵌套构建或构造异常都不会泄漏配置。

    Args:
        vllm_config: 即将用于构建一个模型实例的总配置。

    Yields:
        空值。该 context manager 只提供隐式配置上下文。
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
    """返回当前模型构建上下文中的总配置。

    Returns:
        当前由 :func:`set_current_config` 安装的 ``NanovllmConfig``。

    Raises:
        AssertionError: 在模型构建上下文之外调用时抛出，通常意味着某个层在模块
            导入阶段或 forward 阶段错误地依赖了静态构建配置。
    """
    if _current_vllm_config is None:
        raise AssertionError(
            "Current nano-vLLM vllm_config is not set. "
            "Call this function only while constructing a model inside "
            "set_current_config()."
        )
    return _current_vllm_config


def get_current_config_or_none() -> NanovllmConfig | None:
    """返回当前模型构建配置；没有活动上下文时返回 ``None``。"""
    return _current_vllm_config


T = TypeVar("T")


def get_layers_from_nanovllm_config(
    vllm_config: NanovllmConfig,
    layer_type: type[T],
    layer_names: Iterable[str] | None = None,
) -> dict[str, T]:
    """从静态 forward 上下文中筛选指定类型的模型层。

    Args:
        vllm_config: 已完成模型构建、并已填充静态 module 注册表的总配置。
        layer_type: 要筛选的 Python module 类型，例如 ``Attention``。
        layer_names: 可选的层名子集。为 ``None`` 时遍历注册表中的所有层。

    Returns:
        从完整层名到 ``layer_type`` 实例的映射。名称不存在或类型不匹配的条目不会
        出现在结果中。
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
