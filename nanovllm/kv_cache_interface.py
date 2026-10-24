"""Engine、Executor 与 Worker 共享的 KV cache 配置协议。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass,replace
from enum import Enum
import torch
from typing_extensions import Self

class KVCacheSpecKind(str, Enum):
    """保留 KV 规格的类型名称；当前分派按规格类判断，尚未使用此枚举。"""
    FULL_ATTENTION = "full_attention"
    MLA_ATTENTION = "mla_attention"


@dataclass(frozen=True)
class KVCacheSpec:
    """描述模型中某一层所需的 KV cache 格式。

    Attributes:
        block_size: 每个物理 KV page 保存的 token 数。
    """
    block_size:int
    @property
    def page_size_bytes(self)->int:
        """返回该层一个物理 KV page 占用的字节数。

        Returns:
            page_size_bytes: 该层一个物理 KV page 的存储大小，单位为字节。
        """
        raise NotImplementedError
    @property
    def storage_block_size(self) -> int:
        """返回物理存储 page 的 token 容量，当前与逻辑 block_size 一致。

        Returns:
            storage_block_size: 物理存储 page 的 token 容量，当前等于逻辑 block_size。
        """
        return self.block_size
    def max_memory_usage_bytes(self, max_model_len: int) -> int:
        """返回单个最大长度请求在该层最多占用的缓存字节数。

        Returns:
            max_memory_usage_bytes: 单个最大长度请求在该层最多使用的 KV 字节数。
        """
        raise NotImplementedError
    def copy_with_new_block_size(self, block_size: int) -> Self:
        """创建替换 block_size 的同类型规格，其他字段保持原值。

        Args:
            block_size: 新规格中一个 page 容纳的 token 数。

        Returns:
            kv_cache_spec: 新的不可变 dataclass 实例，不修改原规格，也不重新分配 KV tensor。
        """
        return replace(self, block_size=block_size)
    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """合并同一个 KV cache group 中所有层的规格。

        当前只支持完全相同的 FullAttentionSpec。若模型包含不同类型或不同
        page size 的 Attention，应在这里拒绝，直到实现 hybrid KV cache。

        Returns:
            merged_spec: 所有一致规格的深拷贝，不修改输入规格或分配 KV tensor。
        """
        if not specs:
            raise ValueError("Cannot merge an empty KV cache spec list")

        reference_spec = specs[0]
        if not all(spec == reference_spec for spec in specs[1:]):
            raise NotImplementedError(
                "nano-vLLM currently requires all layers in one KV cache "
                "group to have identical specs"
            )

        return deepcopy(reference_spec)

@dataclass(frozen=True,kw_only=True)
class AttentionSpec(KVCacheSpec):
    """普通 Attention KV cache 的公共规格。

    这里的 ``num_kv_heads`` 是当前 TP rank 本地拥有的 KV head 数，
    不是模型的全局 KV head 数。

    Attributes:
        num_kv_heads: 当前 Worker 本地的 KV head 数。
        head_size: 每个 KV head 的维度。
        dtype: KV cache tensor 的数据类型。
    """
    num_kv_heads:int
    head_size:int
    dtype:torch.dtype
    @property
    def page_size_bytes(self)->int:
        """计算该 Attention 层一个 KV page 的字节数。

        Returns:
            page_size_bytes: K、V 两份缓存及该层精度共同决定的单 page 字节数。
        """
        dtype_size = torch.empty((), dtype=self.dtype).element_size()
        # 2 分别代表 key cache 和 value cache。
        return (
            2
            * self.block_size
            * self.num_kv_heads
            * self.head_size
            * dtype_size
        )
    
@dataclass(frozen=True, kw_only=True)
class FullAttentionSpec(AttentionSpec):
    """标准全注意力层的 KV cache 规格。

    当前 nano-vLLM 只支持 Qwen3 的 full attention，因此暂时不加入
    sliding-window、MLA、Mamba、cross-attention 等规格。
    """

    def max_memory_usage_bytes(self, max_model_len: int) -> int:
        """计算一个最大长度请求在该层最多使用的 KV cache 字节数。

        Returns:
            max_memory_usage_bytes: 覆盖一个最大长度请求所需的全部物理 KV page 字节数。
        """
        # 最后一页即使没有填满也要完整分配，因此先向上取整得到 page 数。
        num_required_blocks = (
            max_model_len + self.block_size - 1
        ) // self.block_size
        return num_required_blocks * self.page_size_bytes

@dataclass
class KVCacheTensor:
    """描述一次原始 KV cache GPU 内存分配。

    本类是 Engine 的**分配计划**，不是已经创建好的 :class:`torch.Tensor`。
    Engine 在 profiling 后只知道“要预留多少字节、这段内存交给哪些 Attention
    layers 使用”；Worker 读取该计划后，才创建真实的 CUDA tensor，并按具体
    Attention backend 的 layout 将原始存储 reshape/view 成 K cache 和 V cache。

    一个 ``KVCacheTensor`` 代表一个完整的 *page pool*，不是一个 page，也不是
    一个 request 的 KV cache。设 :class:`KVCacheConfig` 的 ``num_blocks`` 为
    ``N``，本对象的 ``size`` 为 ``N * P``，那么它描述的 raw buffer 被切为：

    .. code-block:: text

        page 0 | page 1 | ... | page N - 1
          P B      P B             P B

    ``P = size // N`` 是该 tensor 的单个物理 page 字节数。Scheduler 输出的
    ``block_id=i`` 最终就是让对应 Attention layer 在这个 buffer 中访问第 ``i``
    个 page。page 内部如何排列 K/V、KV head、token offset 与 head dimension，
    由该 layer 的 ``KVCacheSpec`` 和 attention backend 决定；因此本类刻意只保存
    字节数，不保存固定的 tensor shape。

    当前 Qwen3 单 group MVP 中，每个 Attention layer 都有一个独立的
    ``KVCacheTensor``，其 ``shared_by`` 只有该 layer 自己。它的最终 view 在概念上
    类似：

    .. code-block:: text

        [num_blocks, K/V, block_size, num_kv_heads, head_size]

    旧版 ModelRunner 把所有层的这些独立 pools 合并为一个大 tensor：
    ``[K/V, num_layers, num_blocks, block_size, num_kv_heads, head_size]``。
    这是分配实现的不同，逻辑上仍等价于“每层拥有 N 个由 block ID 索引的 pages”。

    ``shared_by`` 不是 ``KVCacheGroupSpec.layer_names`` 的替代品。前者回答“这次
    raw GPU 分配要绑定给哪些 layer”，后者回答“哪些 layer 使用同一张逻辑 block
    table”。在 vLLM 的兼容多 group 布局中，来自不同 group、且 page layout 兼容的
    layers 可以绑定同一个 raw buffer；它们通过不同 group 的 block table 访问互斥
    的 ``block_id`` pages，不会覆盖彼此的 K/V。当前 nano-vLLM 尚未实现多 group
    KVCacheManager，因此当前运行时只支持 ``shared_by`` 长度为 1 的情形。

    Attributes:
        size: 此次 raw GPU 内存分配的总字节数。规划器通常按
            ``num_blocks * page_size_bytes`` 计算，故它包含 N 个等大小 page；
            ``page_size_bytes`` 已包含该 layer 一个 block 的 Key 和 Value 数据。
        shared_by: 需要绑定此 raw buffer 的完整 Attention layer 名称。Worker 对
            每个名称建立对应的 KV cache view。当前 Qwen3 中每个列表恰有一个 layer；
            多 group 支持完成后，列表可包含多个 page-layout 兼容但 block table
            相互独立的 layers。
    """

    # 一整段 raw GPU buffer 的总字节数；它内部包含 num_blocks 个等大的 pages。
    size: int
    # 从本次物理分配到 Attention layer 的绑定关系，不表达逻辑 block-table group。
    shared_by: list[str]

@dataclass
class KVCacheGroupSpec:
    """共享同一个 block table 的 Attention layers。

    当前 nano-vLLM 只有一个 KV cache group，包含模型的全部 Attention 层。
    保留这个结构不会要求 Scheduler 立即使用 tuple block IDs；单 group 下
    block ID 仍然可以保持为普通 ``int``。
    """

    layer_names: list[str]
    kv_cache_spec: KVCacheSpec

@dataclass
class KVCacheConfig:
    """描述一个 Worker 的完整 KV cache 分配方案。

    Engine 在模型权重加载、显存 profiling 完成后创建本对象。它不保存任何真实的
    K/V 值，而是同时交付三件彼此独立但必须一致的信息：

    1. ``num_blocks``：全局物理 block ID pool 的容量；
    2. ``kv_cache_tensors``：为实现这批 IDs，GPU 上要创建哪些 raw page pools；
    3. ``kv_cache_groups``：模型层如何把这些 IDs 解释为各自的 block tables。

    ``num_blocks`` 的单位是“全局可分配的物理 block ID”，不是 bytes、token、
    layer 数、请求数或 batch size。设 ``num_blocks=N``，合法的物理 ID 范围便是
    ``[0, N)``。KVCacheManager 从该 pool 取出 ID 并填入 request 的 block table；
    例如 ``[7, 19, 3]`` 表示该请求当前使用 ID 为 7、19、3 的三段 KV pages。

    对当前 Qwen3 的单 group 模型，ID ``7`` 同时选择每个 Attention layer 的
    第 7 个 page，因此它整体表示一段 ``block_size`` token 的模型 KV 状态：

    .. code-block:: text

        request block table: [7, 19, 3]
        layer 0 raw pool:     page 7, page 19, page 3
        layer 1 raw pool:     page 7, page 19, page 3
        ...

    所以单 group 下 ``N`` 表示物理总容量，其中 block 0 永久保留为空块；
    请求最多可同时使用 ``N - 1`` 个完整的逻辑 token blocks。
    每个逻辑 block 都会在 group 内每一层各占一个 page。若 ``block_size=256``，
    且不考虑其他请求竞争、prefix sharing 与 ``max_model_len``，总 token 槽位上界
    才可粗略看作 ``(N - 1) * 256``。

    在 vLLM 的兼容多 group 设计中，group 各自维护不同的 block table，但仍从同一
    个 ID pool 分配，因而它们竞争 ``N`` 个物理 ID，而非各自拥有 N 个独立 ID。
    该设计依赖规划器先把不同 group 的物理 page 大小统一；无法统一的通用异构
    layout 会被拒绝，或需要模型专用的规划器。当前 nano-vLLM 只支持一个 full
    attention group，尚未实现这一多 group 分配协议。

    ``kv_cache_tensors`` 是一维列表，因为它按“实际要分配几段 raw GPU 内存”组织，
    而不按 group 嵌套。列表下标没有 request、token 或模型 layer ID 的业务语义；
    Worker 必须遍历每个元素的 ``shared_by``，才能知道该 raw buffer 应绑定给哪些
    layers。当前 Qwen3 中通常有一个 Attention layer 对应一个列表元素，故列表长度
    约等于 Attention 层数，而它与 ``num_blocks`` 没有直接的数值关系。

    ``kv_cache_groups`` 则从调度器视角组织：同一个 group 内的 layers 共享一张
    block table，故对它们而言 ``block_id=7`` 指向同一段 token 范围；不同 groups
    有不同 block table。当前只有一个 group，所以 SchedulerOutput 可以只传
    ``list[int]``；支持多 group 后才需要按 group 传递多张 block table。

    Attributes:
        num_blocks: 全局物理 block ID pool 的容量。合法 ID 为
            ``0 <= block_id < num_blocks``；它表示 capacity，不表示当前已经分配
            的数量。TP 下每个 rank 最终必须使用相同的 ``num_blocks``，这样同一
            request 的 block table 才能在所有 rank 上被一致解释。
        kv_cache_tensors: 实际 raw GPU 内存池的分配描述列表。每个元素都包含
            ``num_blocks`` 个由 ID 索引的 pages；``size`` 定义总字节数，
            ``shared_by`` 定义 layer 绑定关系。该列表是物理分配计划，不是
            request block tables，也不按 group 嵌套。
        kv_cache_groups: 逻辑 block-table group 的描述列表。每个 group 保存其
            Attention layer 名称和 ``KVCacheSpec``；同组 layers 共用一张 table，
            不同组 layers 使用不同 table。该列表定义 Scheduler、KVCacheManager
            和 ModelRunner 如何共同解释 ``kv_cache_tensors`` 中的 pages。
    """

    # 全局物理 block ID 的容量；不是请求数量、token 数或当前使用量。
    num_blocks: int
    # 按实际 raw GPU 内存分配列出的 page pools；每项内部有 num_blocks 个 pages。
    kv_cache_tensors: list[KVCacheTensor]
    # 按共享 block table 关系划分的 Attention layer groups。
    kv_cache_groups: list[KVCacheGroupSpec]
