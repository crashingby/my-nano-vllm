"""声明 FlashAttention 后端，并执行普通 decoder 的分页 Attention。

Backend 提供能力和缓存布局，Builder 将 Runner 的公共 metadata 转为
后端 metadata，Impl 负责独立写入本轮 K/V，再读取分页缓存计算 Attention。
当前使用 FA2 和非量化 FP16/BF16 缓存，缓存写入由 Triton 算子完成。
Builder 按上游 FA2 分支声明 UNIFORM_BATCH；捕获 metadata 使用基类入口。
Graph 模式解析、固定输入准备和实际 capture/replay 分别由配置、Runner 与
Graph Wrapper 负责；本后端不创建 CUDA Graph。
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

from nanovllm._custom_ops import reshape_and_cache_flash
from nanovllm.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from nanovllm.attention.backends.fa_utils import (
    flash_attn_varlen_func,
    get_flash_attn_version,
)
from nanovllm.config import NanovllmConfig
from nanovllm.config.cache import CacheDType
from nanovllm.kv_cache_interface import AttentionSpec
from nanovllm.logger import init_logger

logger = init_logger(__name__)


class FlashAttentionBackend(AttentionBackend):
    """声明 FlashAttention 的计算实现、元数据构造器和 KV cache 布局。"""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]
    # Attention 包装层先调用 impl.do_kv_cache_update，再调用 impl.forward；
    # 因而 forward 读取缓存时，本轮新 K/V 已经与历史 K/V 一起就位。
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        """当前上游 FA2 CUDA 扩展要求 paged KV page 大小为 256 的倍数。"""
        return [MultipleOf(256)]

    @staticmethod
    def get_name() -> str:
        """返回用于后端选择和日志的稳定名称。"""
        return "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        """返回当前后端的单层 Attention 实现类。"""
        return FlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["FlashAttentionMetadataBuilder"]:
        """返回当前后端的批次元数据构造器类。"""
        return FlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """返回单层 combined KV cache 的 NHD 逻辑形状。

        Args:
            num_blocks: 当前层拥有的物理 kernel page 数。
            block_size: 每个 kernel page 保存的 token 数。
            num_kv_heads: 当前 TP rank 的本地 KV head 数。
            head_size: 每个 KV head 的维度。
            cache_dtype_str: 缓存数据类型名称；当前布局不依赖它。

        Returns:
            依次为 page、K/V、token、KV head、head 维度的形状。
        """
        # 布局规划必须符合实际加载的 FA2 扩展约束，不能沿用 vLLM 扩展的 16 对齐。
        if block_size <= 0 or block_size % 256 != 0:
            raise ValueError(
                "FlashAttention kernel block size must be a positive multiple of 256"
            )
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        """返回 NHD KV cache 的物理维度顺序。

        Args:
            include_num_layers_dimension: 是否在逻辑形状前加入层维度。
                当前每层独立分配，常规调用传 False。

        Returns:
            相对于逻辑形状的维度排列。
        """
        if include_num_layers_dimension:
            return (1, 0, 2, 3, 4, 5)
        return (0, 1, 2, 3, 4)


@dataclass
class FlashAttentionMetadata(AttentionMetadata):
    """保存一个混合 prefill/decode batch 的 Attention 输入。

    Attributes:
        num_actual_tokens: 本轮交给 Attention 的 packed Query 前缀长度。
            沿用公共 metadata；普通执行通常等于有效 token 数，完整 Graph
            捕获和回放时可包含 dummy 请求的 query token，保持捕获形状一致。
        max_query_len: 单个请求本轮最多执行的 token 数。
        query_start_loc: GPU int32 tensor，形状 [num_reqs + 1]，包含末尾边界。
        max_seq_len: 请求执行本轮 token 后的最大序列长度。
        seq_lens: GPU int32 tensor，形状 [num_reqs]，包含历史和本轮新 token。
        block_table: [num_reqs, max_blocks_per_req]，将请求的逻辑 page
            映射为缓存中的物理 page ID；有效长度由 seq_lens 决定。
        slot_mapping: 每个 packed token 对应的物理写入 slot，负值跳过写入。
            独立缓存更新入口实际接收 ForwardContext 传来的同批次映射。
        causal: 本轮是否使用因果 mask，从公共 metadata 传入，默认 True。

    例如 A 本轮执行 3 个 token、执行后总长为 5，B 执行 1 个 token、
    总长为 9：num_actual_tokens=4，query_start_loc=[0, 3, 4]，
    seq_lens=[5, 9]，max_query_len=3，max_seq_len=9。
    Query 只包含这 4 个新 token，K/V 则从缓存读取 A 的 5 个和 B 的 9 个。

    若三个单 token decode 请求补齐到四个请求槽位，Graph 路径可以使用
    num_actual_tokens=4、query_start_loc=[0, 1, 2, 3, 4]，并将末尾 dummy
    请求的 seq_lens 设为 0、slot_mapping 设为 -1；这些值由 Runner 准备。
    描述符和 metadata 的 padding 维度应匹配，回放前原地更新 tensor 内容。
    """

    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool = True  # FA2 为整批使用同一个 bool，未接入 FA4 的逐请求动态 mask。


class FlashAttentionMetadataBuilder(
    AttentionMetadataBuilder[FlashAttentionMetadata]
):
    """构造普通与捕获路径共用的 FA2 metadata，并声明完整 Graph 能力。

    vLLM 对 FA2 使用 UNIFORM_BATCH：max_query_len=1 时存在 packed-GQA
    专用执行路径，按 decode 捕获的 Graph 不能直接拿去执行混合 prefill/decode。
    当前同样保留此保守等级；FA3 和 XPU 的 ALWAYS 分支尚未支持，暂时省略。

    build_for_cudagraph_capture() 继承基类实现，调用本类 build(0, metadata)。
    FA2 没有 FA3 AOT scheduler 的固定调度 buffer，构造时无需额外分配它。
    """

    # 当前仅支持 CUDA FA2，直接采用上游 FA2 等级，避免类定义时探测 GPU。
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: "NanovllmConfig",
        kv_cache_spec: "AttentionSpec",
    ) -> AttentionCGSupport:
        """按 vLLM 的 FA2 分支返回完整捕获能力，不构造 metadata 或查询 GPU。

        Args:
            vllm_config: 当前完整配置；沿用上游签名，FA2 默认能力不读取此参数。
            kv_cache_spec: 当前 AttentionGroup 的缓存规格；默认能力不读取此参数。

        Returns:
            UNIFORM_BATCH。普通 decode 的每请求 query 长度为 1，属于这个范围；
            混合 prefill/decode 的 FULL 策略需由配置解析器降级。
            这是后端能力声明，不代表 Runner 已完成捕获或支持 spec decode。
        """
        return cls._cudagraph_support

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: NanovllmConfig,
        device: torch.device,
    ) -> None:
        """保存组内各层共享的静态构建信息和 Graph 配置快照。

        成员命名与 vLLM 一致。当前无 FA3 AOT schedule，因此没有分配
        scheduler_metadata buffer 或设置 max_num_splits；FA2 的普通 build
        直接引用 Runner 提供的 tensor，Graph 输入地址由 Runner 保持稳定。

        Args:
            kv_cache_spec: 该组 Attention 层的 KV cache 规格。
            layer_names: 使用此 Builder 的完整 Attention 层名。
            vllm_config: 当前 Worker 的完整配置。
            device: 当前 TP rank 的计算设备。
        """
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config  # 静态模型信息，不保存请求进度。
        self.parallel_config = vllm_config.parallel_config  # 保留上游并行配置入口。
        self.cache_config = vllm_config.cache_config  # 保留 KV cache 策略的共享引用。
        self.compilation_config = vllm_config.compilation_config  # Graph 策略和尺寸计划。
        self.kv_cache_dtype = kv_cache_spec.dtype  # 当前组实际的非量化缓存精度。
        self.block_size = kv_cache_spec.block_size  # 当前组 kernel page 的 token 容量。
        self.aot_schedule = False  # 当前只有 FA2，不启用 FA3 的预先调度接口。
        # 构造前总配置应已解析默认模式；此字段沿用上游，记录是否计划使用 FULL。
        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        # FA2 不按此上限分配 scheduler buffer；保留上游字段以供后续版本扩展。
        self.max_cudagraph_size = self.compilation_config.max_cudagraph_capture_size

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttentionMetadata:
        """从公共批次字段构造普通 decoder Attention 元数据。

        这里只创建 Python metadata 对象：tensor 字段直接引用公共元数据，
        不重新排列 token，不分配或复制 GPU tensor。公共缓冲区后续被改写时，
        此对象也会看到变化，因此它不是本轮数据的独立快照。
        捕获入口复用此函数；调用方应已准备好 padding 后的请求边界和 token 数。
        Python 标量 max_query_len/max_seq_len 和 causal 决定捕获时的执行参数，
        回放只更新 GPU tensor，不会重新执行此函数来选择另一套 launch 参数。

        Args:
            common_prefix_len: cascade attention 公共前缀长度；当前未使用。
            common_attn_metadata: Runner 为本轮 batch 创建的公共元数据。
            fast_build: 上游保留的快速构建选项；当前始终使用同一构建路径。

        Returns:
            可由组内所有 Attention 层复用的后端元数据。
        """
        # 普通 decoder 路径直接复用公共字段，无需 cascade 的前缀/后缀拆分。
        return FlashAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,  # 仅改字段名，保留原 tensor。
            slot_mapping=common_attn_metadata.slot_mapping,
            causal=common_attn_metadata.causal,  # 保留 Runner 提供的本轮 mask 语义。
        )


class FlashAttentionImpl(AttentionImpl[FlashAttentionMetadata]):
    """保存单层参数，将 KV 写入和 Attention 计算分别委托给底层算子。

    do_kv_cache_update 原地更新当前层缓存；forward 读取已更新的缓存，
    原地写入包装层预分配的 output。两者的调用顺序由 Attention 包装层保证。
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        kv_cache_dtype: str = "auto",
    ) -> None:
        """保存当前 TP rank 执行 Attention 所需的静态参数。

        Args:
            num_heads: 当前 TP rank 的本地 Query head 数。
            head_size: 每个 Attention head 的维度。
            scale: Attention score 的缩放系数。
            num_kv_heads: 当前 TP rank 的本地 KV head 数；默认等于 Query head 数。
            kv_cache_dtype: 缓存类型名称；auto 跟随实际缓存精度，也支持
                与缓存精度一致的 float16/bfloat16。
        """
        self.num_heads = num_heads  # 本 rank 的 Query head 数。
        self.head_size = head_size  # 当前 Q/K/V 使用相同的每 head 维度。
        self.scale = scale  # softmax 前缩放 QK 分数，通常为 head_size ** -0.5。
        # 未指定 KV head 数时使用 MHA；少于 Query head 数时使用 GQA/MQA。
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype  # 传给缓存写入算子，不在此处转换 tensor。
        # 根据设备能力和 head_size 选择版本；当前只支持 FA2，无法选择时为 None。
        self.vllm_flash_attn_version = get_flash_attn_version(
            head_size=head_size,
        )
        logger.info_once(
            "Using FlashAttention version %s",
            self.vllm_flash_attn_version,
        )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """用本轮 packed Query 和完整分页 K/V 计算 Attention，原地写入 output。

        真实推理前，包装层应已调用 do_kv_cache_update。这里不再次写入
        key/value，而是通过 block_table 读取缓存中的历史和本轮 K/V。
        metadata 为 None 时走 profiling 占位路径，不读取缓存或调用 FA。

        Args:
            layer: 持有当前层状态的 Attention module；保留上游接口，当前未读取。
            query: [T, num_heads, head_size]，按请求顺序拼接的本轮 Query。
            key: [T, num_kv_heads, head_size] 的新 K；已写入缓存，此处未使用。
            value: 与 key 同形状的新 V；已写入缓存，此处未使用。
            kv_cache: [num_blocks, 2, block_size, num_kv_heads, head_size]，
                第二维的 0/1 分别表示 K/V。
            attn_metadata: 本轮请求边界、总序列长度和分页映射；profiling 为 None。
            output: [T, num_heads, head_size]，包装层预分配的可写输出。

        Returns:
            原始 output 对象。仅更新前 num_actual_tokens 行；完整 Graph 路径
            该前缀可包含 dummy query，超出前缀的尾部保持原值。
            profiling 将整个 output 填零。

        Raises:
            AssertionError: 未选出当前设备和 head_size 支持的 FA 版本。

        Side Effects:
            在当前 CUDA stream 上写入 output，不修改 kv_cache 或 metadata。
        """
        # 与上游一致，先保证版本已选定；profiling 也需满足这个前置条件。
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )
        # profiling 没有真实请求边界，填零后返回，避免读取未绑定的缓存。
        if attn_metadata is None:
            return output.fill_(0)

        num_actual_tokens = attn_metadata.num_actual_tokens  # Graph 路径可包含 dummy Query。
        # unbind(1) 去掉长度为 2 的 K/V 轴，得到两个四维视图，不复制缓存。
        # 两个视图共享原存储，page stride 仍跨过 K 和 V；底层按真实 stride 访问。
        key_cache, value_cache = kv_cache.unbind(1)

        # FA2 无需 FA3/FA4 的 TMA singleton stride 规范化或 scheduler metadata。
        # 当前采用普通单段 varlen 路径，未接入 cascade、decode context parallel
        # 或 sliding window，因此无需对应的分支和额外参数。
        cu_seqlens_q = attn_metadata.query_start_loc  # packed Query 的请求边界前缀和。
        seqused_k = attn_metadata.seq_lens  # 每请求实际可读的 KV 长度，包含本轮新 token。
        max_seqlen_q = attn_metadata.max_query_len  # 单请求 Query 长度上界，供 kernel 调度。
        max_seqlen_k = attn_metadata.max_seq_len  # 捕获时需覆盖计划回放的 KV 长度范围。
        block_table = attn_metadata.block_table  # 请求逻辑 page 到物理缓存 page 的映射。
        causal = attn_metadata.causal  # 保留本轮 mask 设置，整批共用一个 bool。

        # Q 长度可以小于 KV 长度；causal=True 时使用右下对齐的因果 mask。
        # 例如 A 有 2 个历史 token、本轮 3 个 Query，总 KV 长度为 5：
        # 第一个 Query 对应序列位置 2，只可读位置 0..2，不能读本轮后两个 token。
        flash_attn_varlen_func(
            q=query[:num_actual_tokens],  # 按 metadata 指定的前缀计算，Graph 时含 dummy。
            k=key_cache,  # 提供分页缓存视图，由 block_table 定位每请求的 K。
            v=value_cache,  # 与 K 使用相同的物理 page 映射。
            out=output[:num_actual_tokens],  # 直接写入调用方的输出前缀，无需结果拷贝。
            cu_seqlens_q=cu_seqlens_q,  # Q 是 packed tensor，用前缀和分段。
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,  # K/V 是 paged tensor，用实际长度限定可读范围。
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=causal,
            block_table=block_table,
            fa_version=self.vllm_flash_attn_version,
        )
        # 返回完整原对象，保持上层统一的输出形状；执行前缀由 metadata 指定。
        return output

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """按 slot_mapping 将本轮新 K/V 原地写入当前层的分页缓存。

        forward_includes_kv_cache_update=False 时，由 Attention 包装层在
        forward 前调用。写入范围取决于 slot_mapping 的长度，无需先切片 K/V。
        完整 Graph 回放应保持 slot_mapping 的地址和长度不变，原地将 dummy
        或退出请求对应的 slot 设为 -1；Triton kernel 在 GPU 上读取新 slot，
        不需要根据本轮有效请求数重新选择 launch 形状。

        Args:
            layer: 当前 Attention module；保留上游接口，非量化路径无需读取它。
            key: [T, num_kv_heads, head_size] 的本轮新 K。
            value: 与 key 同形状、同精度的本轮新 V。
            kv_cache: [num_blocks, 2, block_size, num_kv_heads, head_size]，
                输入和缓存精度必须一致，当前支持 FP16/BF16。
            slot_mapping: [N] 的物理 slot，N <= T；第 i 项对应 key/value 的
                第 i 行，负值跳过，合法范围和有效 slot 的唯一性由调用方保证。

        Side Effects:
            Triton 算子在当前 CUDA stream 上原地修改映射到的缓存位置，
            未映射位置保持原值，不同步 GPU，也不更新请求进度或 block_table。

        例如 block_size=256、slot_mapping[i]=517，则物理 page 为 2、
        page 内偏移为 5：key_cache[2, 5] = key[i]，V 同理；-1 则不写入。
        """
        # 与 forward 使用相同的 K/V 共享视图；保留 stride，不调用 contiguous()。
        key_cache, value_cache = kv_cache.unbind(1)
        # 当前走非量化存储，不需要从 layer 读取 k_scale/v_scale。
        # 同名接口内部委托给 Triton，按 slot_mapping 散写本轮 K/V。
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
        )
