from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol, TypeVar

import numpy as np
import torch
from nanovllm.config import NanovllmConfig
from nanovllm.config.cache import CacheDType

from nanovllm.kv_cache_interface import AttentionSpec
class AttentionType(str, Enum):
    """
    Attention type.
    Use string to be compatible with `torch.compile`.
    """

    DECODER = "decoder"
    """Decoder attention between previous layer Q/K/V."""
    ENCODER = "encoder"
    """Encoder attention between previous layer Q/K/V for encoder-decoder."""
    ENCODER_ONLY = "encoder_only"
    """Encoder attention between previous layer Q/K/V."""
    ENCODER_DECODER = "encoder_decoder"
    """Attention between dec. Q and enc. K/V for encoder-decoder."""

class MultipleOf:
    """表示 kernel page size 必须是指定整数的倍数。

    Attributes:
        base: kernel page size 的整除约束。例如 16 表示允许 16、32、64 等。
    """

    def __init__(self, base: int) -> None:
        """保存正整数倍数约束。

        Args:
            base: kernel page size 必须满足的整除单位。
        """
        self.base = base


class AttentionBackend(ABC):
    """
    描述一个 Attention 后端提供的实现类及其 KV cache 布局。

    这里保存的是类级别的能力与工厂接口，不保存某轮请求的状态。
    """
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list["CacheDType"]] = [
        "auto",
        "float16",
        "bfloat16",
    ]
    # backend 的 forward 内部是否自行把新 K/V 写入 paged KV cache。
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """返回用于识别后端的稳定名称。"""
        raise NotImplementedError
    @staticmethod
    @abstractmethod
    def get_impl_cls() -> type["AttentionImplBase"]:
        """返回实际执行 attention kernel 的实现类。"""

    @staticmethod
    @abstractmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        """返回将公共批次数据转换成后端元数据的 Builder 类。"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """返回一层 combined KV cache tensor 的逻辑形状。

        Args:
            num_blocks: 该层物理 page pool 中的 page 数量。
            block_size: 每个 kernel page 可容纳的 token 数。
            num_kv_heads: 当前 TP rank 的本地 KV head 数。
            head_size: 每个 KV head 的维度。
            cache_dtype_str: KV cache 数据类型名称；当前仅使用 auto。

        Returns:
            包含 num_blocks 维度的逻辑 tensor 形状。
        """
        raise NotImplementedError

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        """返回物理维度排列；未实现时按逻辑形状连续存储。

        Args:
            include_num_layers_dimension: 是否在逻辑形状前加入层维度。
                当前每层独立分配 KV cache，因此调用时应为 False。

        Returns:
            描述物理维度顺序的排列。
        """
        raise NotImplementedError

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        """返回 kernel page size 的约束；默认接受任意正整数倍数。"""
        return [MultipleOf(1)]

    @classmethod
    def get_kv_cache_block_dim(
        cls,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> int:
        """从后端返回的形状中找到 num_blocks 所在维度。

        Args:
            block_size: 每个 kernel page 的 token 数。
            num_kv_heads: 当前 TP rank 的本地 KV head 数。
            head_size: 每个 KV head 的维度。
            cache_dtype_str: KV cache 数据类型名称。

        Returns:
            num_blocks 在逻辑 KV cache 形状中的维度下标。
        """
        _S = 1234567
        shape = cls.get_kv_cache_shape(
            _S,
            block_size,
            num_kv_heads,
            head_size,
            cache_dtype_str=cache_dtype_str,
        )
        return shape.index(_S)
    @classmethod
    def full_cls_name(cls) -> tuple[str, str]:
        """返回可区分不同后端类的模块名和限定类名。"""
        return cls.__module__, cls.__qualname__

class AttentionMetadata:
    """后端专属元数据的类型标记。

    基类无需保存字段。具体后端可定义自己的 dataclass，放入 kernel
    真正需要的长度、block table 和 slot mapping。
    """
    pass


T = TypeVar("T", bound=AttentionMetadata)


@dataclass
class CommonAttentionMetadata:
    """保存 GPUModelRunner 为本轮 batch 准备的公共 Attention 输入。

    长度信息按 request 顺序排列；query 和 slot mapping 按本轮 packed
    token 顺序排列。Builder 读取这些字段，生成后端专属 metadata。

    Attributes:
        query_start_loc: GPU 上的 query 起始偏移前缀和，长度为 num_reqs + 1。
        query_start_loc_cpu: 同一组偏移的 CPU 视图，供需要主机长度的 Builder 使用。
        seq_lens: GPU 上每个请求执行本轮 token 后的总长度。
        num_reqs: 本轮传给 Attention 的请求数。
        num_actual_tokens: 本轮传给 Attention 的 packed token 数。沿用 vLLM
            命名；若以后加入 CUDA Graph padding，它也可能包含补齐 token。
        max_query_len: 本轮单个请求最多执行多少个 token。
        max_seq_len: 本轮单个请求执行后可能达到的最大序列长度。
        block_table_tensor: 每个请求的物理 kernel page ID 表。
        slot_mapping: 每个 packed token 对应的物理 KV slot。
        causal: 是否使用因果 mask；普通 decoder 默认为 True，当前仅支持 bool。
    """

    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    seq_lens: torch.Tensor
    num_reqs: int
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    block_table_tensor: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool = True  # 沿用 vLLM 默认值；不引入 FA4 的逐请求 tensor mask。


M = TypeVar("M")


class AttentionCGSupport(Enum):
    """声明 Attention 后端能随完整 CUDA Graph 捕获的 batch 范围。

    名称和值对齐 vLLM；Runner 对所有 AttentionGroup 取最小 value，作为
    CompilationConfig.resolve_cudagraph_mode_and_sizes() 的输入。
    例如一个组声明 ALWAYS、另一个声明 UNIFORM_BATCH，整模型只能按后者的
    范围使用 FULL。查询能力应通过 Builder.get_cudagraph_support() 完成。

    uniform 指各请求本轮 query 长度相同，不要求历史上下文长度相同。
    例如 query 长度 [1, 1, 1]、上下文长度 [128, 512, 42] 仍是 uniform；
    query 长度 [3, 1, 2] 则需要支持混合 batch 的 ALWAYS。
    该枚举不声明普通计算片段的 PIECEWISE 捕获能力，也不执行实际捕获。
    """

    ALWAYS = 3  # 支持包括 prefill/decode 混合 batch 在内的完整捕获。
    UNIFORM_BATCH = 2  # 本轮 query 长度相同；上游也用此等级表达多 token decode。
    UNIFORM_SINGLE_TOKEN_DECODE = 1  # 仅支持每个请求本轮计算一个 decode token。
    NEVER = 0  # 不支持完整模型捕获；不用于判断未来分段 Graph 的能力。


class AttentionMetadataBuilder(ABC, Generic[M]):
    """将公共批次数据构造成某个 Attention 后端的元数据。

    Builder 按 AttentionGroup 创建，可以被组内多个 Attention 层复用。
    它负责整理数据，不执行 Attention kernel。CUDA Graph 的基础接口分为：
    get_cudagraph_support() 查询该后端能否进入完整 Graph，
    build_for_cudagraph_capture() 为捕获期间的 dummy batch 构造后端 metadata。
    Runner 负责准备固定地址的输入和 padding，再把公共 metadata 交给 Builder。
    """
    # 未明确声明能力的后端默认不支持完整捕获，避免把普通 build 当作能力证明。
    # 外部调用 get_cudagraph_support()，允许子类按配置和 KV cache 规格解析能力。
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    @abstractmethod
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: NanovllmConfig,
        device: torch.device,
    ) -> None:
        """保存当前 AttentionGroup 的静态信息。

        Args:
            kv_cache_spec: 该组 Attention 层共用的 KV cache 规格。
            layer_names: 使用此 Builder 的完整 Attention 层名。
            vllm_config: 当前 Worker 的完整配置。
            device: 本 TP rank 的执行设备。
        """
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device

    @classmethod
    def get_cudagraph_support(
        cls: type["AttentionMetadataBuilder"],
        vllm_config: "NanovllmConfig",
        kv_cache_spec: "AttentionSpec",
    ) -> AttentionCGSupport:
        """返回此 Builder 对完整 CUDA Graph 的支持等级，不构造实例或 metadata。

        Runner 在确定 Attention 后端后，通过各 Builder 类查询能力并取最低等级，
        再交给 CompilationConfig 解析最终 Graph 模式。默认实现返回类属性；
        子类可以覆盖此方法，根据模型配置或 KV cache 规格返回更具体的能力。

        Args:
            vllm_config: 当前 Worker 的完整配置，供子类检查影响捕获的执行策略。
            kv_cache_spec: 当前 AttentionGroup 的缓存规格，供子类检查布局等限制。
                默认实现保留这两个上游参数，但不读取它们。

        Returns:
            AttentionCGSupport 等级；不表示某个 batch 已有可回放的 Graph 条目。
        """
        return cls._cudagraph_support

    @abstractmethod
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> M:
        """为当前 batch 创建后端专属元数据。

        Args:
            common_prefix_len: 用于 cascade attention 的公共前缀长度。
                当前没有 cascade attention，调用方传 0。
            common_attn_metadata: Runner 创建的本轮公共批次数据。
            fast_build: 是否优先降低构建开销；当前正常执行传 False。

        Returns:
            后端的 AttentionImpl.forward 能直接读取的元数据。
        """
        raise NotImplementedError

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> M:
        """为 CUDA Graph 捕获期间的 batch 构造后端专属 metadata。

        默认与 vLLM 一致，调用 self.build()，common_prefix_len 固定为 0，
        fast_build 沿用 build 的默认 False；不启用当前未支持的 cascade attention。
        子类需要捕获专用的字段或固定形状策略时，可覆盖此方法，但应调用
        self.build() 或 super().build_for_cudagraph_capture() 完成基础构建。

        例如调用方已将三个单 token decode 请求补齐到四个槽位，传入的公共
        metadata 应描述四个请求和四个 query token；此方法原样传给 build，
        不把四个槽位裁回三个，也不负责填充 query_start_loc 或 slot_mapping。

        Args:
            common_attn_metadata: 调用方为捕获准备的公共 metadata；tensor 地址、
                padding 和各字段一致性均需在调用前准备好。

        Returns:
            类型 M 的后端 metadata，供捕获期间的 AttentionImpl.forward 使用。
            默认实现遵循子类 build 的引用和分配行为，不保证复制输入 tensor。

        Side Effects:
            默认入口仅委托 build；后端的 buffer 分配等行为由具体 build 决定。
            本方法不查询捕获能力、不安装 ForwardContext，也不捕获或回放 Graph。
        """
        # 保留上游委托流程；能力判定由 Runner 在调用捕获入口前独立完成。
        return self.build(
            common_prefix_len=0,  # 捕获路径不启用公共前缀的 cascade 分支。
            common_attn_metadata=common_attn_metadata,  # 保留调用方准备的形状和 tensor。
        )


class AttentionLayer(Protocol):
    """供 AttentionImpl 识别单层 Attention 对象的最小类型约定。

    具体实现可以通过 layer 读取层级状态；当前普通 BF16 Attention
    暂时不需要 vLLM 用于量化的 q/k/v scale 字段。
    """
    #TODO:很难理解vllm源码里这里的成员
    kv_cache: torch.Tensor


class AttentionImplBase(ABC, Generic[T]):
    """保存普通 Attention 和未来其他实现共有的基本结构参数。

    vLLM 在这里还初始化 PCP/DCP 等上下文并行状态。当前仅保留
    num_heads、head_size、scale 三个基础约定。
    """

    num_heads: int
    head_size: int
    scale: float
    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        """在权重加载结束后执行实现特有的处理。

        Args:
            act_dtype: 模型计算使用的数据类型。

        当前普通 Attention 无需额外处理，因此默认实现为空。
        """


class AttentionImpl(AttentionImplBase[T], Generic[T]):
    """普通 decoder self-attention 的单层计算接口。"""


    kv_cache_dtype: str

    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        kv_cache_dtype: str = "auto",
    ) -> None:
        """创建某层的 Attention 计算实现。

        Args:
            num_heads: 当前 TP rank 的本地 Query head 数。
            head_size: 单个 head 的维度。
            scale: Attention score 的缩放系数。
            num_kv_heads: 当前 TP rank 的本地 KV head 数。
            kv_cache_dtype: KV cache 数据类型；当前使用 auto。
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: T,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """执行一层 Attention，并写入预分配的输出 tensor。

        Args:
            layer: 当前 Attention module，供实现读取层级状态。
            query: 本轮 packed token 的 Query。
            key: 本轮 packed token 的 Key。
            value: 本轮 packed token 的 Value。
            kv_cache: 当前层已绑定的 combined KV cache tensor。
            attn_metadata: Builder 为本轮生成的后端专属元数据。
            output: 调用方预分配的输出 tensor。

        Returns:
            本层的 Attention 结果，通常是写入后的 output。
        """
        raise NotImplementedError
