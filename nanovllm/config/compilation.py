"""定义 CUDA Graph 执行策略与模型构建期共享的静态上下文。

模式名称和辅助接口对齐 vLLM。这里仅描述配置，不创建 CUDA stream、捕获
Graph 或探测 GPU；具体支持范围由后续 Dispatcher、Attention 后端和 Runner
共同决定。当前计划先接入 FULL_DECODE_ONLY，piecewise 执行链尚未实现。
"""

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nanovllm.logger import init_logger

if TYPE_CHECKING:
    # 运行时在方法内部导入，避免 Attention 后端反向导入总配置产生循环依赖。
    from nanovllm.attention.backend import AttentionCGSupport

logger = init_logger(__name__)


class CUDAGraphMode(enum.Enum):
    """描述配置中的 Graph 策略，以及单次 forward 的具体运行模式。

    NONE、PIECEWISE、FULL 可以作为具体运行模式；两个组合模式只用于配置，
    由 Dispatcher 根据 batch 选择 decode_mode() 或 mixed_mode()。
    保留上游全部枚举名称不代表当前项目已经实现全部执行方式。

    例如 FULL_DECODE_ONLY.value 为 (2, 0)：decode_mode() 返回 FULL，
    mixed_mode() 返回 NONE。FULL_AND_PIECEWISE 则分别返回 FULL、PIECEWISE。
    """

    NONE = 0  # 关闭 CUDA Graph，按正常模型调用路径执行。
    PIECEWISE = 1  # 对可捕获的模型片段分别使用 Graph，当前执行链尚未接入。
    FULL = 2  # 捕获整个模型 forward；后端能力决定哪些 batch 可以使用。
    FULL_DECODE_ONLY = (FULL, NONE)  # decode 用完整 Graph，混合 batch 不使用 Graph。
    FULL_AND_PIECEWISE = (FULL, PIECEWISE)  # decode 用完整 Graph，混合 batch 用分段 Graph。

    def decode_mode(self) -> "CUDAGraphMode":
        """返回 uniform decode batch 对应的具体模式，不修改配置。

        组合模式取 tuple 第一项；普通模式直接返回自身。
        """
        return CUDAGraphMode(self.value[0]) if self.separate_routine() else self

    def mixed_mode(self) -> "CUDAGraphMode":
        """返回 prefill/混合 batch 对应的具体模式，组合模式取 tuple 第二项。"""
        return CUDAGraphMode(self.value[1]) if self.separate_routine() else self

    def has_mode(self, mode: "CUDAGraphMode") -> bool:
        """判断策略是否包含指定的具体运行模式。

        Args:
            mode: NONE、PIECEWISE 或 FULL；组合模式不能作为查询条件。
        """
        # 查询目标必须是单个模式，避免把两个执行策略当作具体运行方式比较。
        assert not mode.separate_routine()
        if self.separate_routine():
            # 组合 value 保存普通模式的整数值，因此比较 mode.value。
            return mode.value in self.value
        return self == mode

    def requires_piecewise_compilation(self) -> bool:
        """判断策略是否需要分段编译支持；完整 Graph 本身不要求分段编译。"""
        return self.has_mode(CUDAGraphMode.PIECEWISE)

    def max_cudagraph_mode(self) -> "CUDAGraphMode":
        """按 NONE < PIECEWISE < FULL 返回策略包含的最高模式。

        用于判断是否需要完整模型 Wrapper；不表示实际 batch 必然使用该模式。
        """
        return CUDAGraphMode(max(self.value)) if self.separate_routine() else self

    def has_full_cudagraphs(self) -> bool:
        """判断策略中是否有执行路径使用完整模型 Graph。"""
        return self.max_cudagraph_mode() == CUDAGraphMode.FULL

    def has_piecewise_cudagraphs(self) -> bool:
        """判断策略中是否有执行路径使用分段 Graph。"""
        return self.requires_piecewise_compilation()

    def separate_routine(self) -> bool:
        """判断 decode 和混合 batch 是否分别配置了运行模式。"""
        return isinstance(self.value, tuple)

    @classmethod
    def valid_runtime_modes(cls) -> frozenset["CUDAGraphMode"]:
        """返回上游定义的三个具体运行模式，排除仅用于配置的组合模式。

        这不是当前项目的后端能力声明；PIECEWISE 在此保留其枚举语义。
        """
        return frozenset({cls.NONE, cls.PIECEWISE, cls.FULL})

    def is_valid_runtime_mode(self) -> bool:
        """判断自身能否作为具体运行模式，而非需要继续分派的组合策略。"""
        return self in CUDAGraphMode.valid_runtime_modes()

    def __str__(self) -> str:
        """返回模式名称，供配置日志和运行模式统计使用。"""
        return self.name

    def __bool__(self) -> bool:
        """仅 NONE 为 False；其他枚举表示策略包含某种 CUDA Graph 执行方式。"""
        return self != CUDAGraphMode.NONE


@dataclass(slots=True)
class CompilationConfig:
    """保存 CUDA Graph 配置，以及模型构建期间登记的静态 module 表。

    与 vLLM 一样，部分字段初始为 None，表示等待总配置结合模型、Scheduler
    容量和后端能力完成解析。None 不等价于 CUDAGraphMode.NONE；Runner 使用
    捕获尺寸、Dispatcher 初始化 keys 之前，必须先完成配置解析。

    本类不执行捕获。static_forward_context 供 Worker 等模型外部组件按稳定
    层名查找 module，例如收集 Attention 的 KVCacheSpec；本轮 Attention
    metadata、Graph 运行模式和描述符由独立的 ForwardContext 携带。

    Attributes:
        cudagraph_mode: Graph 配置策略；None 表示尚未解析，NONE 表示明确禁用。
        cudagraph_num_of_warmups: 每个捕获形状在正式 capture 前的预热次数。
            具体循环由 Runner 执行；不包含其他 profiling/kernel warmup 次数。
        cudagraph_capture_sizes: 要捕获的 packed token 数列表；None 表示待推导。
            普通单 token decode 下，token 数也等于请求数。[] 是解析后的空列表，
            不应将它当成 None 自动推导默认尺寸。
        cudagraph_copy_inputs: 上游 piecewise 编译路径的输入复制选项；当前预留，
            不控制完整模型 Wrapper 的输入复制。
        max_cudagraph_capture_size: 最大捕获 token 数；None 表示待推导，存在捕获
            尺寸时应与规范化列表的最大值一致。超过范围的 batch 应回退普通执行。
        compilation_time: 模型级编译耗时统计，单位为秒。当前尚无模型级编译链，
            因而保持 0.0；不代表预热、Graph 捕获或局部函数 JIT 的总耗时。
        static_forward_context: 稳定完整层名到模型 module 的实例级映射；当前主要
            保存 Attention，供 Worker 查询和运行时上下文引用。
    """
    # 保留上游的 None 初始值与字段类型；总配置解析完成后再交给运行时组件。
    cudagraph_mode: CUDAGraphMode = None  # type: ignore[assignment]
    # 沿用上游默认值 0；后续预热流程必须保证首次 JIT 编译发生在 capture 外。
    cudagraph_num_of_warmups: int = 0
    # None 等待 NanovllmConfig 推导；显式列表随后统一排序和验证容量约束。
    cudagraph_capture_sizes: list[int] = None  # type: ignore[assignment]
    # 当前 Runner 自己维护固定地址 buffer；此预留字段尚无 piecewise 使用者。
    cudagraph_copy_inputs: bool = False
    # 这是 token 维度的捕获上限，不是最大上下文长度，也不是 KV cache block 数。
    max_cudagraph_capture_size: int = None  # type: ignore[assignment]
    # 由编译流程累计、供 Worker 返回 CompilationTimes；当前只有零值统计占位。
    compilation_time: float = field(default=0.0, init=False)
    # 每个配置独立持有注册表；init=False 防止调用方注入外部 module 映射。
    static_forward_context: dict[str, Any] = field(
        default_factory=dict,
        init=False,
    )

    def post_init_cudagraph_sizes(self) -> None:
        """在捕获尺寸解析完成后排序，并检查最大尺寸的不变量。

        对齐 vLLM 的同名接口，由 NanovllmConfig._set_cudagraph_sizes() 在
        推导默认尺寸、去重和按 Scheduler 容量过滤后调用。
        这里不推导默认值，也不替调用方修改最大捕获尺寸。
        上游还会在此解析 compile_sizes；当前项目尚未支持该字段，暂时省略。

        例如捕获尺寸为 [8, 1, 4]、最大尺寸为 8，调用后列表变为 [1, 4, 8]。
        Graph 禁用时，总配置应先将尺寸列表和最大尺寸分别解析为 [] 和 0。

        Side Effects:
            原地将 cudagraph_capture_sizes 按升序排列，供 Dispatcher 构建
            padding 映射和 Runner 遍历捕获尺寸使用。

        Raises:
            AssertionError: 捕获尺寸尚未解析，或非空列表的最大值与
                max_cudagraph_capture_size 不一致；表示配置解析流程未完成。
        """
        # None 是未解析状态；空列表则是允许进入收尾流程的已解析状态。
        assert self.cudagraph_capture_sizes is not None, (
            "cudagraph_capture_sizes must be resolved before "
            "post_init_cudagraph_sizes()."
        )
        # 去重由总配置负责；沿用上游在此仅原地排序的职责划分。
        self.cudagraph_capture_sizes.sort()
        # 非空列表排序后的末项必须等于总配置已确定的最大捕获 token 数。
        if self.cudagraph_capture_sizes:
            assert (
                self.cudagraph_capture_sizes[-1] == self.max_cudagraph_capture_size
            ), (
                "The largest cudagraph_capture_sizes entry "
                f"({self.cudagraph_capture_sizes[-1]}) must match "
                "max_cudagraph_capture_size "
                f"({self.max_cudagraph_capture_size})."
            )

    def resolve_cudagraph_mode_and_sizes(
        self,
        min_cg_support: "AttentionCGSupport",
        min_cg_attn_backend: str | None,
    ) -> CUDAGraphMode:
        """根据所有 Attention 后端的最低能力解析最终 Graph 配置策略。

        对齐 vLLM 的同名流程：Runner 确定后端后汇总能力，调用本方法，再用
        返回模式初始化 Dispatcher 的捕获 keys；在 metadata Builder 构造前调用。
        None/NONE 统一为 NONE；FULL 遇到仅支持 uniform batch 的后端时降级
        为 FULL_DECODE_ONLY；decode 完整捕获不受支持时回退 NONE。

        当前没有分段编译，显式 PIECEWISE/FULL_AND_PIECEWISE 会报错，自动
        回退也不会选择 PIECEWISE。当前只支持单 token decode，无 spec decode
        的尺寸对齐和 Mamba cache 检查，因此沿用接口名称但不修改捕获尺寸。
        默认尺寸推导和排序仍由总配置及 post_init_cudagraph_sizes() 完成。

        Args:
            min_cg_support: 所有 AttentionGroup 中 value 最小的完整捕获能力。
            min_cg_attn_backend: 对应后端名称，用于定位能力不足的后端；可为 None。

        Returns:
            最终配置策略，与回写的 cudagraph_mode 一致；组合策略仍需由
            Dispatcher 根据每轮 batch 选择具体运行模式。

        Side Effects:
            回写 cudagraph_mode，自动降级时记录 warning；捕获尺寸保持原值。

        Raises:
            ValueError: 请求尚未实现的分段模式，或 FULL 模式遇到完全不支持
                完整捕获的 Attention 后端；后一种情况沿用上游的报错顺序。
        """
        from nanovllm.attention.backend import AttentionCGSupport

        cudagraph_mode = self.cudagraph_mode
        # 沿用上游早返回语义：此处不推导默认模式，也不清空已有尺寸。
        if cudagraph_mode is None or cudagraph_mode == CUDAGraphMode.NONE:
            self.cudagraph_mode = CUDAGraphMode.NONE
            return CUDAGraphMode.NONE

        # 枚举保留上游名称，但当前不能把未实现的执行路径交给 Dispatcher。
        if cudagraph_mode.has_piecewise_cudagraphs():
            raise ValueError(
                f"CUDAGraphMode.{cudagraph_mode.name} requires piecewise "
                "compilation, which is not implemented yet."
            )

        # 先检查混合 batch 的 FULL；仅 ALWAYS 声明支持这条路径。
        if (
            cudagraph_mode.mixed_mode() == CUDAGraphMode.FULL
            and min_cg_support != AttentionCGSupport.ALWAYS
        ):
            msg = (
                f"CUDAGraphMode.{cudagraph_mode.name} is not supported "
                f"with {min_cg_attn_backend} backend (support: {min_cg_support})"
            )
            # 与上游一致，显式 FULL 遇到 NEVER 在混合 batch 检查处直接报错。
            if min_cg_support == AttentionCGSupport.NEVER:
                raise ValueError(msg + "; please use cudagraph_mode=NONE")

            # 尚无 Attention 分段编译路径，因此保留 decode FULL、关闭混合 Graph。
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
            logger.warning(msg + "; setting cudagraph_mode=FULL_DECODE_ONLY")

        # FULL_DECODE_ONLY 遇到 NEVER 时，沿用上游无分段编译情况下的 NONE 回退。
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and min_cg_support == AttentionCGSupport.NEVER
        ):
            logger.warning(
                "CUDAGraphMode.%s is not supported with %s backend "
                "(support: %s); setting cudagraph_mode=NONE because "
                "attention is not compiled piecewise",
                cudagraph_mode.name,
                min_cg_attn_backend,
                min_cg_support,
            )
            cudagraph_mode = CUDAGraphMode.NONE

        # 与上游一样，在自动降级后再次保护完整 Graph 的后端能力不变量。
        if (
            cudagraph_mode.has_full_cudagraphs()
            and min_cg_support == AttentionCGSupport.NEVER
        ):
            raise ValueError(
                f"CUDAGraphMode.{cudagraph_mode.name} is not supported "
                f"with {min_cg_attn_backend} backend (support: {min_cg_support})"
            )

        # 当前不涉及 spec decode 的尺寸对齐；这里只提交最终配置策略。
        self.cudagraph_mode = cudagraph_mode
        return cudagraph_mode
