"""KV cache 物理 block 的轻量元数据类型。

本模块不保存实际的 K/V tensor；tensor 由模型执行层管理。这里的
``KVCacheBlock`` 只保存 Scheduler 和 KVCacheManager 做资源分配、引用计数和
prefix cache 查找所需要的 CPU 侧元数据。
"""
from __future__ import annotations
from dataclasses import dataclass
import os
from copy import deepcopy
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, NewType,Iterable
from nanovllm.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
)
from nanovllm.config import NanovllmConfig
if TYPE_CHECKING:
    from nanovllm.request import Request


# 逻辑完整 token block 的前缀相关 hash。
# 运行时它仍是 bytes；NewType 只提供静态类型上的区分。
BlockHash = NewType("BlockHash", bytes)
# 首个逻辑 block 的固定父 hash；Engine 创建 hasher 前必须初始化。
NONE_HASH: BlockHash

def init_none_hash(hash_fn:Callable[[Any],bytes])->None:
    """初始化首个逻辑 block 使用的父 hash。"""
    global NONE_HASH
    hash_seed = os.getenv("PYTHONHASHSEED")
    if hash_seed is None:
        NONE_HASH = BlockHash(os.urandom(32))
    else:
        NONE_HASH = BlockHash(hash_fn(hash_seed))


def hash_block_tokens(
    hash_function: Callable[[Any], bytes],
    parent_block_hash: BlockHash | None,
    curr_block_token_ids: Sequence[int],
    extra_keys: tuple[Any, ...] | None = None,
) -> BlockHash:
    """计算一个完整逻辑 block 的前缀相关 hash。

    prefix cache 不能只 hash 当前 block 的 token IDs。两个请求可能拥有相同的
    当前 token block、却拥有不同的更早前缀；此时它们对应的 attention K/V 不可
    互换。因此本函数把父 block hash、当前完整 block 的 token IDs，以及会改变
    K/V 语义的可选请求元数据共同组成 hash key。

    首个完整 block 没有真实父节点，使用由 Engine 初始化的 ``NONE_HASH`` 作为
    统一根节点。之后每个 block 都依赖前一个 block 的 hash，形成 prefix hash 链。
    任一位置的 token 或请求特定语义发生变化时，当前及后续 block 的 hash 都会
    随之变化，防止错误复用 KV cache。

    当前 nano-vLLM 的 Qwen 纯文本请求没有 LoRA、多模态输入、prompt embeddings
    或 cache salt，因此调用方传入 ``extra_keys=None``。未来接入这些功能时，必须
    将能改变 K/V 内容的请求特定信息放入 ``extra_keys``。

    Args:
        hash_function: Engine 根据 ``prefix_caching_hash_algo`` 选择的底层 hash
            函数。该函数接收一个完整的结构化 key，并返回 ``bytes`` digest。
        parent_block_hash: 前一完整 block 的 hash；首 block 为 ``None``。
        curr_block_token_ids: 当前完整逻辑 block 的 token IDs。调用方必须确保
            它已填满一个逻辑 block。
        extra_keys: 可选的请求特定语义信息，例如 LoRA 标识、多模态特征 hash、
            prompt embedding hash 或 cache salt。纯文本请求传入 ``None``。

    Returns:
        当前完整逻辑 block 的 ``BlockHash``。该值既是下一个 block 的父 hash，
        也是 KVCacheManager 查询 prefix cache 时使用的逻辑 key。
    """
    # 首 block 没有真实父 block；所有 Request 共享同一个根 hash 作为链的起点。
    if parent_block_hash is None:
        parent_block_hash = NONE_HASH

    # 固化为 tuple，确保 token 顺序和内容以不可变形式参与 hash 计算。
    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    # 该三元组的字段顺序是 prefix-cache key 协议的一部分，与 vLLM 保持一致。
    hash_key = (
        parent_block_hash,
        curr_block_token_ids_tuple,
        extra_keys,
    )

    # NewType 只增加静态类型语义；运行时保存的仍是底层算法返回的 bytes digest。
    return BlockHash(
        hash_function(hash_key)
    )

def get_request_block_hasher(
    block_size:int,
    caching_hash_fn: Callable[[Any], bytes],
)->Callable[[Request],list[BlockHash]]:
    """创建一个增量计算 Request 完整逻辑 block hash 的共享函数。

    Engine 在初始化时只调用一次本函数，得到的同一个 callable 会传给所有 Request。
    每个 Request 自己保存 ``block_hashes``；hasher 只根据该列表长度计算尚未处理的
    新完整 blocks，不会重复 hash 旧 prompt。

    Args:
        block_size: 一个逻辑 hash block 覆盖的 token 数。当前 MVP 中它与 KV cache
            的物理 block size 保持一致。

    Returns:
        一个接收 Request、返回新增完整 block hash 列表的函数。

    Raises:
        ValueError: block_size 不为正数时抛出。
    """
    if block_size <= 0:
        raise ValueError(
            f"block_size must be positive, got {block_size}"
        )
    def request_block_hasher(request:Request)->list[BlockHash]:
        """计算 request 尚未记录的完整 token blocks 的 hash。

        Args:
            request: token 序列已经更新的逻辑请求。

        Returns:
            新出现的完整逻辑 blocks 的 hash 列表。末尾不完整 block 不会被 hash。
        """
        # 已有 hash 数乘 block_size，即第一个尚未 hash 的 token 位置。
        start_token_idx = len(request.block_hashes) *block_size
        # 完整 token 序列包含 prompt 和已确认的 output tokens。 
        num_tokens=request.num_tokens
        # 当前剩余 token 不足一个完整 block 时，不产生任何新的 prefix-cache key。
        if start_token_idx + block_size > num_tokens:
            return []
        # 当前 block hash 依赖前一个完整 block hash，形成 prefix hash 链。
        prev_block_hash_value = (request.block_hashes[-1] if request.block_hashes else None)
        # 只收集本次新增 hash，最后由 Request.update_block_hashes() 追加。
        new_block_hashes : list[BlockHash] =[]
        while start_token_idx + block_size <= num_tokens:
            # 当前完整 block 的右开 token 下标。
            end_token_idx = start_token_idx + block_size
            # 当前逻辑 block 的完整 token 内容。
            block_tokens = request.all_token_ids[start_token_idx:end_token_idx]

            block_hash = hash_block_tokens(
                hash_function=caching_hash_fn,
                parent_block_hash=prev_block_hash_value,
                curr_block_token_ids=block_tokens,
                extra_keys=None
            )

            new_block_hashes.append(block_hash)
            start_token_idx += block_size
            prev_block_hash_value = block_hash
        return new_block_hashes
    return request_block_hasher

@dataclass(slots=True)
class KVCacheBlock:
    """描述一个可分配的物理 KV cache block。

    Attributes:
        block_id: block 在预分配物理 KV cache 池中的稳定编号。
        ref_cnt: 当前引用该物理 block 的逻辑请求数量。为零时 block 可回到空闲
            队列，但仍可保留 hash，作为可复用的 prefix cache。
        _block_hash: 完整逻辑 block 的前缀相关 hash。``None`` 表示该 block 尚
            未提交到 prefix cache，或已经被重新分配并清理旧缓存身份。
        prev_free_block: 空闲队列中的前驱，仅由 FreeKVCacheBlockQueue 维护。
        next_free_block: 空闲队列中的后继，分配出队后与前驱一起清空。
        is_null: 是否为永久保留的空块；该块不分配给请求，也不参与 prefix cache。
    """

    # 物理编号决定 block 在 GPU KV tensor 中对应的固定槽位。
    block_id: int
    # 引用计数保护共享 prefix cache，防止仍被请求使用的 block 被重新分配。
    ref_cnt: int = 0
    # _block_hash 只能在一个 block 首次被缓存时设置，重用时通过 reset_hash 清理。
    _block_hash: BlockHash  | None = None
    # 仅由 FreeKVCacheBlockQueue 维护，支持 O(1) 移除命中的空闲缓存块。
    prev_free_block: KVCacheBlock | None = None
    next_free_block: KVCacheBlock | None = None
    is_null: bool = False  # 对齐 vLLM，物理 block 0 被 KVCacheManager 标记为空块。


    @property
    def block_hash(self) -> BlockHash  | None:
        """返回 block 当前的 prefix cache hash。

        Returns:
            已缓存时返回 hash；尚未缓存或重用后已失效时返回 ``None``。
        """
        return self._block_hash

    @block_hash.setter
    def block_hash(self, block_hash: BlockHash ) -> None:
        """首次将一个完整逻辑 block 标记为可复用的 prefix cache。

        Args:
            block_hash: 由 token 内容和前缀 hash 共同计算得到的新 hash。

        Raises:
            AssertionError: 当前 block 已经关联其他 hash 时抛出。覆盖旧 hash 会让
                缓存 hash 索引与物理内容不一致，因此必须先完成 eviction。

        Side Effects:
            设置 ``_block_hash``。清理旧 hash 必须调用 :meth:`reset_hash`，而不是
            给该 property 赋 ``None``。
        """
        assert self.block_hash is None, (
            "The block already has a hash. This should not happen."
        )
        self._block_hash = block_hash

    def reset_hash(self) -> None:
        """在物理 block 被驱逐或重新分配前清理其 prefix cache 身份。

        调用方必须先从 ``cached_block_hash_to_block`` 删除旧映射；否则 hash
        仍会指向已经被新 token 覆写的物理内存。

        Side Effects:
            将 ``_block_hash`` 设为 ``None``，允许之后的完整 token block 再次写入
            新 hash。
        """
        self._block_hash = None

    def __repr__(self) -> str:
        """只显示相邻块编号，避免双向链表递归打印整个空闲队列。"""
        prev_block_id = (
            self.prev_free_block.block_id if self.prev_free_block else None
        )
        next_block_id = (
            self.next_free_block.block_id if self.next_free_block else None
        )
        return (
            f"KVCacheBlock(block_id={self.block_id}, ref_cnt={self.ref_cnt}, "
            f"_block_hash={self._block_hash!r}, "
            f"prev_free_block={prev_block_id}, next_free_block={next_block_id})"
        )


class FreeKVCacheBlockQueue:
    """按 vLLM 的双向链表维护空闲块及缓存驱逐顺序。

    队首优先分配。块上的前后指针使 prefix 命中时的中间移除保持 O(1)，
    队列本身不修改引用计数，也不拥有请求到物理块的映射。
    """

    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        """按物理编号串联空闲块，并设置永不分配的头尾哨兵。"""
        self.num_free_blocks: int = len(blocks)  # 仅统计真实空闲块。
        for i in range(self.num_free_blocks):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i + 1]
        self.fake_free_list_head = KVCacheBlock(block_id=-1)  # 不进入物理块池。
        self.fake_free_list_tail = KVCacheBlock(block_id=-1)
        if self.num_free_blocks > 0:
            self.fake_free_list_head.next_free_block = blocks[0]
            blocks[0].prev_free_block = self.fake_free_list_head
            self.fake_free_list_tail.prev_free_block = blocks[-1]
            blocks[-1].next_free_block = self.fake_free_list_tail
        else:
            self.fake_free_list_head.next_free_block = self.fake_free_list_tail
            self.fake_free_list_tail.prev_free_block = self.fake_free_list_head

    def popleft(self) -> KVCacheBlock:
        """移除并返回队首真实块，同时减少空闲块数量。"""
        first_block = self.fake_free_list_head.next_free_block
        if first_block is self.fake_free_list_tail or first_block is None:
            assert self.num_free_blocks == 0, "Free block count is out of sync"
            raise ValueError("No free blocks available")
        if first_block.next_free_block is None:
            raise RuntimeError("Invalid block without next_free_block")
        self.fake_free_list_head.next_free_block = first_block.next_free_block
        first_block.next_free_block.prev_free_block = self.fake_free_list_head
        first_block.prev_free_block = first_block.next_free_block = None
        self.num_free_blocks -= 1  # 已取出的块不再属于空闲链表。
        return first_block

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        """批量取出队首 n 个块，返回值保持原有驱逐顺序。"""
        if n == 0:
            return []
        assert self.num_free_blocks >= n  # 分配前由 Manager 检查整体容量。
        self.num_free_blocks -= n
        curr_block = self.fake_free_list_head.next_free_block
        ret: list[KVCacheBlock] = []
        for _ in range(n):
            assert curr_block is not None
            ret.append(curr_block)
            last_block = curr_block
            curr_block = curr_block.next_free_block
            last_block.prev_free_block = last_block.next_free_block = None
        if curr_block is not None:
            self.fake_free_list_head.next_free_block = curr_block
            curr_block.prev_free_block = self.fake_free_list_head
        return ret

    def remove(self, block: KVCacheBlock) -> None:
        """从链表中间移除一个命中的空闲缓存块，复杂度为 O(1)。"""
        if block.prev_free_block is None or block.next_free_block is None:
            raise RuntimeError(f"remove() called on an invalid block: {block}")
        block.prev_free_block.next_free_block = block.next_free_block
        block.next_free_block.prev_free_block = block.prev_free_block
        block.prev_free_block = block.next_free_block = None
        self.num_free_blocks -= 1

    def append(self, block: KVCacheBlock) -> None:
        """将一个缓存块放回队尾，延后它的驱逐优先级。"""
        last_block = self.fake_free_list_tail.prev_free_block
        if last_block is None:
            raise RuntimeError("prev_free_block of fake tail should exist")
        last_block.next_free_block = block
        block.prev_free_block = last_block
        block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = block
        self.num_free_blocks += 1

    def prepend_n(self, blocks: list[KVCacheBlock]) -> None:
        """将未缓存的空闲块按给定顺序放到队首，优先重新分配。"""
        if not blocks:
            return
        first_block = self.fake_free_list_head.next_free_block
        assert first_block is not None
        prev_block = self.fake_free_list_head
        for block in blocks:
            block.prev_free_block = prev_block
            prev_block.next_free_block = block
            prev_block = block
        prev_block.next_free_block = first_block
        first_block.prev_free_block = prev_block
        self.num_free_blocks += len(blocks)

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        """将已缓存的空闲块按给定顺序放回队尾。"""
        if not blocks:
            return
        last_block = self.fake_free_list_tail.prev_free_block
        assert last_block is not None
        for block in blocks:
            block.prev_free_block = last_block
            last_block.next_free_block = block
            last_block = block
        last_block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = last_block
        self.num_free_blocks += len(blocks)

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        """返回按驱逐顺序排列的空闲块快照，供状态检查使用。"""
        ret: list[KVCacheBlock] = []
        curr_block = self.fake_free_list_head.next_free_block
        if curr_block is None:
            raise RuntimeError("next_free_block of fake head should exist")
        while curr_block.next_free_block is not None:
            ret.append(curr_block)
            curr_block = curr_block.next_free_block
        return ret

def create_kv_cache_group_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
    grouped_layer_names: list[list[str]],
) -> list[KVCacheGroupSpec]:
    """根据 layer 分组结果构造 KV cache group 规格。

    ``kv_cache_spec`` 保存每一层独立上报的缓存格式；
    ``grouped_layer_names`` 则只描述哪些层共享同一份逻辑 block table。本函数把
    两种信息合并成 ModelRunner 和 KVCacheManager 都能消费的
    :class:`KVCacheGroupSpec`。

    同一个 group 中的 layers 必须拥有可合并的 ``KVCacheSpec``。当前 MVP 的
    ``KVCacheSpec.merge`` 只接受完全一致的 ``FullAttentionSpec``，因此 Qwen3
    的全部 decoder attention layers 可以组成唯一一个 group。

    Args:
        kv_cache_spec: 从完整 Attention layer name 到该层 ``KVCacheSpec`` 的
            映射。key 必须覆盖 ``grouped_layer_names`` 中出现的所有 layer。
        grouped_layer_names: 已完成分组的 layer names。外层 list 表示多个 KV
            cache groups，内层 list 表示共享一个 block table 的 layers。

    Returns:
        与 ``grouped_layer_names`` 顺序一致的 ``KVCacheGroupSpec`` 列表。

    Raises:
        KeyError: 分组中出现了 ``kv_cache_spec`` 未包含的 layer name 时抛出。
        ValueError: 某个 group 不包含任何 layer 时，由 spec 合并逻辑抛出。
        NotImplementedError: 同组 layers 的 KV cache 规格无法由当前 MVP 合并时
            抛出。
    """
    # kv_cache_groups 按输入的 group 顺序收集最终规格，顺序后续也会定义 group ID。
    kv_cache_groups: list[KVCacheGroupSpec] = []

    # 每个内层列表代表一组共享同一逻辑 block table 的 Attention layers。
    for layer_names_one_group in grouped_layer_names:
        # layer_specs 是当前 group 内各层独立上报的物理 KV page 规格。
        layer_specs = [
            kv_cache_spec[layer_name]
            for layer_name in layer_names_one_group
        ]

        # merged_layer_spec 是当前 group 对外使用的统一规格；MVP 要求各层完全相同。
        merged_layer_spec = layer_specs[0].merge(layer_specs)

        kv_cache_groups.append(
            KVCacheGroupSpec(
                layer_names=layer_names_one_group,
                kv_cache_spec=merged_layer_spec,
            )
        )

    return kv_cache_groups


def get_kv_cache_groups(
    vllm_config: NanovllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """将模型的 Attention layers 划分为 KV cache groups。

    KV cache group 表示一组共享同一份逻辑 block table 的 layers，并不表示这些
    layers 共享同一个 K/V tensor。当前 nano-vLLM 只支持规格完全相同的 full
    attention layers，因此会把 Qwen3 的全部 Attention layers 放入唯一 group。

    ``vllm_config`` 当前未参与单 group 分支的计算，但保留该参数以对齐 vLLM 接口。
    后续支持关闭 hybrid KV cache manager 或按配置调整 group 策略时，会在这里
    使用它。

    Args:
        vllm_config: nano-vLLM 全局配置。当前 MVP 仅为接口兼容而接收该参数。
        kv_cache_spec: 当前模型从 layer name 到独立 ``KVCacheSpec`` 的映射。

    Returns:
        当前模型的 KV cache groups。无 Attention layer 时返回空列表；Qwen3
        正常返回只包含一个 group 的列表。

    Raises:
        NotImplementedError: layers 无法合并为统一 ``FullAttentionSpec``，或模型
            使用当前 MVP 尚未支持的其他 KV cache 类型时抛出。
    """
    # attention-free 模型不需要 GPU KV cache，因此没有任何 group。
    if not kv_cache_spec:
        return []

    # layer_specs 保存模型所有 Attention layers 各自上报的缓存规格。
    layer_specs = list(kv_cache_spec.values())
    # merged_spec 验证这些层能否共享同一种 block table 语义和 page layout。
    merged_spec = layer_specs[0].merge(layer_specs)

    if not isinstance(merged_spec, FullAttentionSpec):
        raise NotImplementedError(
            "nano-vLLM currently supports FullAttentionSpec only"
        )

    return create_kv_cache_group_specs(
        kv_cache_spec=kv_cache_spec,
        grouped_layer_names=[list(kv_cache_spec)],
    )


def get_uniform_page_size(
    kv_cache_specs: Iterable[KVCacheSpec],
) -> int:
    """返回一组 KV cache specs 统一的物理 page 字节数。

    Args:
        kv_cache_specs: 待比较的 KV cache specs。调用方通常传入每个 group 的
            ``kv_cache_spec``。

    Returns:
        所有 specs 共有的 ``page_size_bytes``。

    Raises:
        AssertionError: 输入为空，或不同 specs 的 page 字节数不一致时抛出。
            当前 allocator 只能管理统一大小的物理 page。
    """
    # page_sizes 去重后应只剩一个值；多个值意味着 allocator 需要 hybrid 支持。
    page_sizes = {
        kv_cache_spec.page_size_bytes
        for kv_cache_spec in kv_cache_specs
    }

    assert len(page_sizes) == 1
    return page_sizes.pop()


def may_override_num_blocks(
    vllm_config: NanovllmConfig,
    num_blocks: int,
) -> int:
    """应用用户显式指定的 KV block 数覆盖值。

    Args:
        vllm_config: 全局配置，其中 ``cache_config.num_gpu_blocks_override`` 保存可选
            覆盖值。
        num_blocks: 根据显存 profiling 自动计算出的物理 block 数。

    Returns:
        存在 override 时返回用户值，否则返回自动计算出的 ``num_blocks``。
    """
    # override 主要用于测试和复现实验；正常推理应使用 profiling 的计算结果。
    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        num_blocks = vllm_config.cache_config.num_gpu_blocks_override

    return num_blocks


def get_num_blocks(
    vllm_config: NanovllmConfig,
    num_layers: int,
    available_memory: int,
    page_size: int,
) -> int:
    """把单个 Worker 的可用显存换算为物理 KV block 数。

    一个逻辑 block ID 会在 group 的每一层各占用一个物理 page，因此一个 block
    的总成本为 ``num_layers * page_size``。本函数先按该成本整除显存预算，再应用
    可选的 ``num_gpu_blocks_override``。

    Args:
        vllm_config: 全局配置，用于读取可选 block 数覆盖值。
        num_layers: 当前内存池需要分别保存 KV page 的 layer 数量。单 group
            Qwen3 中等于 decoder Attention layer 数。
        available_memory: 当前 Worker 可用于 KV cache 的显存字节数。
        page_size: 单个 Attention layer 的一个 K/V page 占用的字节数。

    Returns:
        当前 Worker 可分配的物理 KV block 数，结果不会小于零。

    Raises:
        ZeroDivisionError: ``num_layers`` 或 ``page_size`` 为零时抛出，表示上游
            group/spec 构造违反内部不变量。
    """
    # 自动 block 数使用整数除法，确保最终 tensor 总大小不超过显存预算。
    num_blocks = available_memory // page_size // num_layers
    # 防御异常的负显存预算，保证 block 数至少为零。
    num_blocks = max(int(num_blocks), 0)

    return may_override_num_blocks(vllm_config, num_blocks)


def check_enough_kv_cache_memory(
    vllm_config: NanovllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    available_memory: int,
) -> None:
    """检查单个 Worker 能否容纳至少一个最大长度请求。

    Args:
        vllm_config: 全局配置，其中 ``max_model_len`` 决定单请求最大上下文长度。
        kv_cache_spec: 当前 Worker 从 layer name 到 ``KVCacheSpec`` 的映射。
        available_memory: 当前 Worker 可用于 KV cache 的显存字节数。

    Returns:
        ``None``。显存满足最低要求时不修改任何输入对象。

    Raises:
        ValueError: 所有 layers 为一个 ``max_model_len`` 请求保存 KV，再加每层
            一页保留空块所需的总字节数超过 ``available_memory`` 时抛出。
    """
    # 一个最大长度请求之外，每层还需要一页永久保留的 null block。
    # 否则恰好够存请求的预算，在保留 block 0 后会少一个可分配的 block。
    required_memory = sum(
        layer_spec.max_memory_usage_bytes(vllm_config.model_config.max_model_len)
        + layer_spec.page_size_bytes
        for layer_spec in kv_cache_spec.values()
    )

    if required_memory > available_memory:
        raise ValueError(
            "No available memory for the KV cache blocks: "
            f"required={required_memory} bytes, "
            f"available={available_memory} bytes, "
            f"max_model_len={vllm_config.model_config.max_model_len}"
        )


def get_kv_cache_config_from_groups(
    vllm_config: NanovllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """根据 KV cache groups 为单个 Worker 生成最终 tensor 分配方案。

    Args:
        vllm_config: 全局配置，用于 block 数覆盖等缓存策略。
        kv_cache_groups: 当前 Worker 的 KV cache groups。每个 group 描述共享同一
            block table 的 layers 及其统一 spec。
        available_memory: 当前 Worker 可用于 KV cache 的显存字节数。

    Returns:
        当前 Worker 的 ``KVCacheConfig``，其中包含物理 block 数、每段 tensor 的
        字节大小和 layer 绑定关系。

    Raises:
        AssertionError: 不同 groups 的物理 page 大小不一致时抛出。
        ValueError: 根据显存预算或 override，无法同时提供一个空块和一个请求块。

    Notes:
        当前 Qwen3 只有一个 group。代码仍保留 vLLM 的 group-aware tensor layout，
        使每个 ``layer_index`` 能从每个 group 中选择对应 layer 并共享一段内存。
    """
    if not kv_cache_groups:
        # KVCacheManager 当前需要至少一个 null block。
        return KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=[],
        )

    # group_size 是所有 groups 中最大的 layer 数，也是需要建立的内存段数量。
    # 单 group Qwen3 中，它就是 decoder Attention layer 数量。
    group_size = max(
        len(group.layer_names)
        for group in kv_cache_groups
    )
    # page_size 是每个 layer 为一个逻辑 block 保存 K/V 时占用的统一字节数。
    page_size = get_uniform_page_size(
        group.kv_cache_spec
        for group in kv_cache_groups
    )
    # num_blocks 是当前 Worker 在全部 layers 上都能同时建立的逻辑 block 数量。
    num_blocks = get_num_blocks(
        vllm_config=vllm_config,
        num_layers=group_size,
        available_memory=available_memory,
        page_size=page_size,
    )

    if num_blocks <= 1:
        raise ValueError(
            "KV cache requires at least two physical blocks: "
            "one reserved null block and one allocatable block"
        )

    # kv_cache_tensors 按 layer_index 描述 Worker 后续应分配的每一段 GPU 存储。
    kv_cache_tensors: list[KVCacheTensor] = []

    # 该循环保留 vLLM 的 group-aware tensor layout。
    # 当前只有一个 group，因此每个 tensor 实际只对应一个 layer。
    for layer_index in range(group_size):
        # shared_by 收集各 group 在相同 layer_index 位置使用该物理 tensor 的 layers。
        shared_by = [
            group.layer_names[layer_index]
            for group in kv_cache_groups
            if layer_index < len(group.layer_names)
        ]

        kv_cache_tensors.append(
            KVCacheTensor(
                size=page_size * num_blocks,
                shared_by=shared_by,
            )
        )

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )


def get_kv_cache_configs(
    vllm_config: NanovllmConfig,
    kv_cache_specs: list[dict[str, KVCacheSpec]],
    available_memory: list[int],
) -> list[KVCacheConfig]:
    """为所有 Workers 生成具有统一 block ID 空间的 KV cache 配置。

    本函数是 Engine 侧 KV cache planning 的总入口。它先合并并校验各 Worker
    上报的 layer specs，再生成全局 group 布局，随后根据每个 Worker 的显存预算
    分别构造 ``KVCacheConfig``。最后将所有配置收缩到最小 block 数，使 TP ranks
    使用相同的合法 block ID 范围。

    Args:
        vllm_config: 全局配置，提供 ``max_model_len``、block override 等缓存策略。
        kv_cache_specs: 按 Worker rank 排列的 layer spec 字典。外层下标是 rank，
            每个字典内部是 ``layer_name -> KVCacheSpec``。
        available_memory: 按 Worker rank 排列的 KV cache 可用显存字节数。

    Returns:
        按 Worker rank 排列的 ``KVCacheConfig``。所有配置的 ``num_blocks`` 最终
        相同，但 tensor 和 layer 绑定信息分别属于对应 Worker。

    Raises:
        ValueError: Worker 数量不一致、没有 Worker、同名 layer 在不同 TP ranks
            上报告不同 spec，或某个 Worker 显存不足时抛出。
        NotImplementedError: Workers 拥有不同 layer 集合、需要 PP layer projection
            时抛出。
        RuntimeError: 收缩 block 数时发现 tensor size 与原 block 数不成整数比例，
            表示上游配置构造破坏了 page-size 不变量。

    Side Effects:
        只修改本函数新创建的 ``KVCacheConfig`` 对象以统一 ``num_blocks``；不会
        修改调用方传入的 ``kv_cache_specs`` 和 ``available_memory``。
    """
    # 两个外层列表都按 Worker rank 对齐，长度不同会导致错误的 rank 配置配对。
    if len(kv_cache_specs) != len(available_memory):
        raise ValueError(
            "kv_cache_specs and available_memory must have the same length"
        )
    if not kv_cache_specs:
        raise ValueError("No Worker KV cache specs were provided")

    # 合并 Worker specs。当前只有 TP，没有 PP，因此相同 layer 在各 rank
    # 上必须具有相同的本地 KV 规格。
    # merged_kv_cache_specs 是 Engine 看到的完整模型 layer-spec 权威映射。
    merged_kv_cache_specs: dict[str, KVCacheSpec] = {}

    # 外层循环依次消费每个 rank 上报的完整 layer spec 字典。
    for worker_specs in kv_cache_specs:
        # 内层循环合并该 Worker 的每一个 Attention layer。
        for layer_name, layer_spec in worker_specs.items():
            # existing_spec 用于检查同名 layer 在不同 TP ranks 上是否具有相同布局。
            existing_spec = merged_kv_cache_specs.get(layer_name)

            if existing_spec is None:
                merged_kv_cache_specs[layer_name] = layer_spec
            elif existing_spec != layer_spec:
                raise ValueError(
                    "The same layer reported different KV cache specs "
                    f"across TP workers: {layer_name}"
                )

    # 不支持 PP，因此每个 Worker 都必须包含完整且相同的 layer 集合。
    for worker_specs in kv_cache_specs:
        if worker_specs != merged_kv_cache_specs:
            raise NotImplementedError(
                "Different Worker layer sets require pipeline-parallel "
                "KV cache projection, which nano-vLLM does not support yet"
            )

    # global_kv_cache_groups 是全部 Workers 共同遵守的逻辑 block-table 分组方案。
    global_kv_cache_groups = get_kv_cache_groups(
        vllm_config=vllm_config,
        kv_cache_spec=merged_kv_cache_specs,
    )

    # kv_cache_configs 按 rank 收集依据各自显存预算生成的初始配置。
    kv_cache_configs: list[KVCacheConfig] = []

    # specs 与 memory 使用 zip 按相同 rank 配对；长度已经在函数入口校验。
    for worker_specs, worker_memory in zip(
        kv_cache_specs,
        available_memory,
    ):
        check_enough_kv_cache_memory(
            vllm_config=vllm_config,
            kv_cache_spec=worker_specs,
            available_memory=worker_memory,
        )

        kv_cache_configs.append(
            get_kv_cache_config_from_groups(
                vllm_config=vllm_config,
                kv_cache_groups=deepcopy(global_kv_cache_groups),
                available_memory=worker_memory,
            )
        )

    # TP ranks 必须共享相同的 block ID 范围，因此使用容量最小的 rank。
    # min_num_blocks 是全部 Worker 都能实际分配的统一物理 block 容量。
    min_num_blocks = min(
        kv_cache_config.num_blocks
        for kv_cache_config in kv_cache_configs
    )

    # 逐 rank 收缩配置；容量本来就是最小值的 Worker 不需要改 tensor size。
    for kv_cache_config in kv_cache_configs:
        # old_num_blocks 用于从 tensor 总字节数反推出单 block 的 page 字节数。
        old_num_blocks = kv_cache_config.num_blocks
        kv_cache_config.num_blocks = min_num_blocks

        if old_num_blocks == min_num_blocks:
            continue

        # tensor.size 与 num_blocks 成正比；统一 block 数后同步缩小 tensor。
        for tensor_config in kv_cache_config.kv_cache_tensors:
            if tensor_config.size % old_num_blocks != 0:
                raise RuntimeError(
                    "KV cache tensor size is not divisible by num_blocks"
                )

            # page_size 是该 tensor 为单个逻辑 block 预留的固定字节数。
            page_size = tensor_config.size // old_num_blocks
            tensor_config.size = page_size * min_num_blocks

    # override 可能比 profiling 预算更小；正式分配方案也必须容纳最大长度请求和空块。
    # 仅在正式规划入口检查，Graph profiling 使用的临时小 cache 不受此限制。
    for kv_cache_config, worker_specs in zip(kv_cache_configs, kv_cache_specs):
        allocated_memory = sum(tensor.size for tensor in kv_cache_config.kv_cache_tensors)
        check_enough_kv_cache_memory(
            vllm_config=vllm_config,
            kv_cache_spec=worker_specs,
            available_memory=allocated_memory,
        )

    return kv_cache_configs
