"""单组对象式 KV 管理，合并 vLLM Manager、SingleTypeManager 和 BlockPool。

只保留 full attention、本地前缀缓存和普通文本生成。所有物理资源状态仍由
一个 Manager 拥有，发送给 Worker 时才将 KVCacheBlock 转换成物理编号。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, overload

from nanovllm.core.kv_cache_utils import (
    BlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
)
from nanovllm.kv_cache_interface import KVCacheConfig
from nanovllm.metrics.stats import PrefixCacheStats
from nanovllm.request import Request


@dataclass
class KVCacheBlocks:
    """按 KV group 包装物理块；仅承载数据，不分配或增减资源引用。"""

    blocks: tuple[Sequence[KVCacheBlock], ...]  # 当前只使用 blocks[0]。

    @overload
    def get_block_ids(
        self, allow_none: Literal[False] = False,
    ) -> tuple[list[int], ...]:
        """默认返回各 KV group 的编号列表，空组保留为空列表。

        Returns:
            block_ids: 按 KV group 排列的物理 block ID 列表元组，空 group 保留为空列表。
        """
        ...

    @overload
    def get_block_ids(
        self, allow_none: Literal[True],
    ) -> tuple[list[int], ...] | None:
        """允许全空增量以 None 表示，使 Worker 跳过 block table 追加。

        Returns:
            block_ids: 按 KV group 排列的编号快照；全部为空且允许省略时为 None。
        """
        ...

    def get_block_ids(
        self, allow_none: bool = False,
    ) -> tuple[list[int], ...] | None:
        """创建编号快照，允许增量协议用 None 表示没有新增块。

        Returns:
            block_ids: 按 KV group 排列的编号快照；allow_none 且全部为空时为 None。
        """
        if allow_none and all(len(group) == 0 for group in self.blocks):
            return None
        return tuple([block.block_id for block in group] for group in self.blocks)


class BlockHashToBlockMap:
    """保留同一 hash 的多个物理副本，驱逐一个副本不删除其他副本。"""

    def __init__(self) -> None:
        """单副本直接保存对象，多副本才创建内部编号映射。"""
        self._cache: dict[BlockHash, KVCacheBlock | dict[int, KVCacheBlock]] = {}

    def get_one_block(self, key: BlockHash) -> KVCacheBlock | None:
        """查询任意一个匹配副本，不改变空闲队列或引用计数。

        Returns:
            block: 任意一个匹配 block_hash 的物理副本；未命中时为 None。
        """
        blocks = self._cache.get(key)
        if blocks is not None:
            if isinstance(blocks, KVCacheBlock):
                return blocks
            return next(iter(blocks.values()))
        return None

    def insert(self, key: BlockHash, block: KVCacheBlock) -> None:
        """登记新副本，保留已经发送给 Worker 的物理块编号。"""
        blocks = self._cache.get(key)
        if blocks is None:
            self._cache[key] = block
        elif isinstance(blocks, KVCacheBlock):
            self._cache[key] = {blocks.block_id: blocks, block.block_id: block}
        else:
            blocks[block.block_id] = block

    def pop(self, key: BlockHash, block_id: int) -> KVCacheBlock | None:
        """只驱逐指定物理副本，其余同 hash 块仍可命中。

        Returns:
            block: 移除的指定物理副本；不存在该 hash 或 block ID 时为 None。
        """
        blocks = self._cache.pop(key, None)
        if blocks is None:
            return None
        if isinstance(blocks, KVCacheBlock):
            if blocks.block_id == block_id:
                return blocks
            self._cache[key] = blocks  # 编号不匹配时恢复原索引。
            return None
        block = blocks.pop(block_id, None)
        if blocks:
            self._cache[key] = blocks
        return block

    def __len__(self) -> int:
        """返回前缀 hash 数量，同 hash 的物理副本不重复计数。

        Returns:
            num_hashes: 缓存的不同前缀 hash 数量，同 hash 的物理副本只计一次。
        """
        return len(self._cache)


class KVCacheManager:
    """统一维护请求块序列、前缀缓存、引用计数和空闲块队列。

    req_to_blocks 和 num_cached_block 对应上游 SingleTypeKVCacheManager；
    free_block_queue 和 cached_block_hash_to_block 对应上游 BlockPool。
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = True,
        log_stats: bool = False,
    ) -> None:
        """根据 Worker 已确定的单组配置创建物理块元数据。

        Args:
            kv_cache_config: 物理块总数和唯一 KV group 的规格。
            max_model_len: token slots 可达到的最大序列长度。
            enable_caching: 是否登记和查询完整块的本地前缀缓存。
            log_stats: 是否记录前缀查询和命中 token 数。

        Side Effects:
            创建 CPU 侧块对象、空闲队列和索引，并永久取走 block 0 作为空块。
            不分配或清零真实 GPU KV tensor，那部分由 Worker 按同一配置完成。
        """
        # 单 group 让每个物理编号在所有 Attention 层表示同一段逻辑 token。
        assert len(kv_cache_config.kv_cache_groups) == 1
        assert kv_cache_config.num_blocks > 0  # 永久保留的 block 0 必须存在。
        self.max_model_len: int = max_model_len
        self.enable_caching: bool = enable_caching
        self.log_stats: bool = log_stats
        self.prefix_cache_stats: PrefixCacheStats | None = (
            PrefixCacheStats() if log_stats else None
        )
        self.kv_cache_config: KVCacheConfig = kv_cache_config
        self.block_size: int = (  # token/block，调度器和 Worker 必须使用一致的单位。
            kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
        )
        self.num_gpu_blocks: int = kv_cache_config.num_blocks  # 包含不可分配的 block 0。
        # 一次建立固定身份的块对象；引用计数、hash 与空闲链表都操作这些对象。
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(self.num_gpu_blocks)
        ]
        self.free_block_queue: FreeKVCacheBlockQueue = FreeKVCacheBlockQueue(
            self.blocks
        )
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()
        # null block 永久离开空闲队列，只用于 padding，不参与正常缓存驱逐。
        self.null_block: KVCacheBlock = self.free_block_queue.popleft()
        self.null_block.is_null = True
        # 每个请求按逻辑 token 位置保存块引用，同一物理块可以由多个请求共享。
        self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)
        self.num_cached_block: dict[str, int] = {}  # 活跃请求已登记的完整块数。
        self.empty_kv_cache_blocks: KVCacheBlocks = KVCacheBlocks(((),))

    @property
    def usage(self) -> float:
        """返回去除永久 null block 后的物理块使用率。

        Returns:
            usage: 去除永久 null block 后的真实物理块使用率，范围为 [0, 1]。
        """
        return self.get_usage()

    def make_prefix_cache_stats(self) -> PrefixCacheStats | None:
        """取出并轮换前缀统计，关闭统计时不创建计数对象。

        Returns:
            stats: 本轮前缀缓存统计；关闭统计时为 None，取出后内部切换到新统计对象。
        """
        if not self.log_stats:
            return None
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """查询最长完整前缀，最后一个 token 留给模型重新计算 logits。

        查询不取得物理引用；allocate_slots() 容量检查成功后才 touch 命中块。

        Args:
            request: 保存真实 token 历史和完整块 hash 的待接纳/恢复请求。

        Returns:
            (computed_blocks, num_new_computed_tokens):
                computed_blocks: 连续命中的前缀 KVCacheBlocks 包装，未命中或关闭缓存时为空包装。
                num_new_computed_tokens: 命中块覆盖的 token 数；至少保留最后一个 token 重算 logits，未命中时为 0。

        Side Effects:
            开启统计时累加查询/命中数，不增加块引用计数或占用新物理块。
        """
        if not self.enable_caching:
            return self.empty_kv_cache_blocks, 0
        max_cache_hit_length = request.num_tokens - 1  # 即使全部历史命中也要产生最新 logits。
        computed_blocks: list[KVCacheBlock] = []
        max_num_blocks = max_cache_hit_length // self.block_size
        # Full attention 只能跳过从位置 0 开始的连续前缀，中间缺块后不再继续查找。
        for block_hash in request.block_hashes[:max_num_blocks]:
            block = self.get_cached_block(block_hash)
            if block is None:
                break  # Full attention 只复用从头开始连续命中的前缀。
            computed_blocks.append(block)
        num_new_computed_tokens = len(computed_blocks) * self.block_size
        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.record(
                num_tokens=request.num_tokens,
                num_hits=num_new_computed_tokens,
                preempted=request.num_preemptions > 0,
            )
        return self.create_kv_cache_blocks((computed_blocks,)), num_new_computed_tokens

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
    ) -> KVCacheBlocks | None:
        """按上游顺序检查容量、引用命中块、分配新块并登记完整前缀。

        Args:
            request: 即将执行本轮 forward 的请求。
            num_new_tokens: 本轮新增计算 token 数，不含前缀命中。
            num_new_computed_tokens: 本次新命中的前缀 token 数。
            new_computed_blocks: 与新命中 token 对应的物理块包装。

        Returns:
            new_blocks: 本次新增块的包装；容量不足返回 None，且不改变映射和引用。

        Side Effects:
            成功后更新请求块序列、引用计数和缓存 hash。与 vLLM 相同，完整
            前缀元数据在提交 forward 前登记，执行端必须保证 GPU 工作有序。
        """
        if num_new_tokens <= 0:
            raise ValueError("num_new_tokens must be greater than 0")
        new_computed_block_list = (
            new_computed_blocks.blocks[0]
            if new_computed_blocks is not None
            else self.empty_kv_cache_blocks.blocks[0]
        )
        # running 已有进度与本次新查出的命中前缀均不属于本轮新增计算量。
        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        total_computed_tokens = min(num_local_computed_tokens, self.max_model_len)
        # 本轮结束后的逻辑长度决定需要多少物理块，不按新增 token 数单独取整。
        num_tokens_need_slot = min(
            total_computed_tokens + num_new_tokens, self.max_model_len
        )
        req_blocks = self.req_to_blocks.get(request.request_id, [])
        num_required_blocks = (
            num_tokens_need_slot + self.block_size - 1
        ) // self.block_size
        num_new_blocks = max(
            num_required_blocks - len(req_blocks) - len(new_computed_block_list), 0
        )
        # 命中但 ref_cnt 为 0 的块还在空闲队列，touch 后也会消耗可用容量。
        num_evictable_blocks = sum(
            block.ref_cnt == 0 and not block.is_null
            for block in new_computed_block_list
        )
        if num_new_blocks + num_evictable_blocks > self.get_num_free_blocks():
            return None  # 容量检查必须早于 touch 和任何请求状态修改。

        req_blocks = self.req_to_blocks[request.request_id]
        if new_computed_block_list:
            assert not req_blocks  # 只有首次接纳或抢占恢复才附加缓存前缀。
            self.touch(new_computed_block_list)  # 先保护命中块，避免新分配将其驱逐。
            req_blocks.extend(new_computed_block_list)
            self.num_cached_block[request.request_id] = len(req_blocks)
        new_blocks = self.get_new_blocks(num_new_blocks)
        req_blocks.extend(new_blocks)  # 已有编号保持稳定，仅追加逻辑尾部。
        if self.enable_caching:
            num_tokens_to_cache = min(
                total_computed_tokens + num_new_tokens, request.num_tokens
            )
            self.cache_blocks(request, num_tokens_to_cache)
        return self.create_kv_cache_blocks((new_blocks,))

    def free(self, request: Request) -> None:
        """移除请求映射，并按逻辑尾部优先的顺序归还物理引用。"""
        self.free_blocks(reversed(self.pop_blocks_for_free(request)))

    def pop_blocks_for_free(self, request: Request) -> list[KVCacheBlock]:
        """移交物理块对象，暂不减少引用计数或归还空闲池。

        延迟回收调用方接管这些引用，写入完成后调用 free_blocks()。
        尚未分配就取消的请求返回空列表。

        Returns:
            req_blocks: 从请求映射移交的物理块列表，尚未分配时为空；引用计数暂不减少。
        """
        req_blocks = self.req_to_blocks.pop(request.request_id, [])
        self.num_cached_block.pop(request.request_id, None)
        return req_blocks

    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        """返回请求的物理块包装，不改变引用计数。

        Returns:
            blocks: 当前请求的 KVCacheBlocks 包装，不改变物理引用计数。
        """
        return self.create_kv_cache_blocks((self.req_to_blocks[request_id],))

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        """创建 Worker 使用的物理编号快照。

        Returns:
            block_ids: 请求各 KV group 的物理 block ID 列表元组。
        """
        return self.get_blocks(request_id).get_block_ids()

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """按调用方给定的逻辑进度增量登记完整块，避免重复写入 hash。

        Args:
            request: 已拥有当前物理 block table 和真实历史 hash 的请求。
            num_computed_tokens: 允许发布的前缀 token 数，向下取整到完整 block。

        Side Effects:
            为新增完整块写入 hash 索引并推进 num_cached_block；不执行设备同步。
            调用方负责提交、回收和块复用时的设备先后关系。
        """
        if not self.enable_caching:
            return
        num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
        num_full_blocks = num_computed_tokens // self.block_size  # 不发布尚未填满的尾块。
        if num_cached_blocks >= num_full_blocks:
            return
        self.cache_full_blocks(
            request=request,
            blocks=self.req_to_blocks[request.request_id],
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks,
        )
        self.num_cached_block[request.request_id] = num_full_blocks

    def create_kv_cache_blocks(
        self, blocks: tuple[Sequence[KVCacheBlock], ...],
    ) -> KVCacheBlocks:
        """包装块序列，空结果复用空包装；非空时保留原序列引用，不复制块。

        Returns:
            kv_cache_blocks: 保留原块序列引用的 KVCacheBlocks；没有块时复用空包装。
        """
        return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks

    def get_cached_block(self, block_hash: BlockHash) -> KVCacheBlock | None:
        """查询唯一 group 中任意一个匹配的物理副本。

        Returns:
            block: 唯一 KV group 内任意一个匹配的物理副本，未命中时为 None。
        """
        return self.cached_block_hash_to_block.get_one_block(block_hash)

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
    ) -> None:
        """写入新增完整块的 hash，保留同 hash 的其他物理副本。"""
        if num_cached_blocks >= num_full_blocks:
            return
        # 请求前部已登记，只有当前新增的完整后缀可以首次设置 block_hash。
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert len(request.block_hashes) >= num_full_blocks
        new_block_hashes = request.block_hashes[num_cached_blocks:]
        for i, block in enumerate(new_full_blocks):
            if block.is_null:
                continue
            assert block.block_hash is None  # 只处理尚未登记的逻辑块范围。
            block_hash = new_block_hashes[i]
            block.block_hash = block_hash
            self.cached_block_hash_to_block.insert(block_hash, block)

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """取出空闲块，先驱逐旧 hash，再取得物理块引用。

        Returns:
            ret: 按空闲队列顺序取出的物理块列表，已驱逐旧 hash 并取得新的引用。
        """
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")
        ret = self.free_block_queue.popleft_n(num_blocks)
        for block in ret:
            if self.enable_caching:
                self._maybe_evict_cached_block(block)
            assert block.ref_cnt == 0
            block.ref_cnt += 1
        return ret

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """只驱逐当前物理副本的 hash，不影响其他同 hash 块。

        Returns:
            evicted: 当前物理副本的旧 hash 被实际驱逐时为 True，否则为 False。
        """
        block_hash = block.block_hash
        if block_hash is None:
            return False
        if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:
            return False
        block.reset_hash()
        return True

    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        """取得缓存命中引用，必要时先从空闲链表中移除。"""
        for block in blocks:
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """按上游顺序优先复用无 hash 块，保留有 hash 块的缓存寿命。

        Args:
            ordered_blocks: 已按驱逐优先级排序的块，调用方负责反转请求链。
        """
        blocks_with_hash: list[KVCacheBlock] = []
        blocks_without_hash: list[KVCacheBlock] = []
        for block in ordered_blocks:
            block.ref_cnt -= 1
            if block.ref_cnt == 0 and not block.is_null:
                if block.block_hash is None:
                    blocks_without_hash.append(block)
                else:
                    blocks_with_hash.append(block)
        self.free_block_queue.prepend_n(blocks_without_hash)
        self.free_block_queue.append_n(blocks_with_hash)

    def get_num_free_blocks(self) -> int:
        """返回空闲真实物理块数量，不包含永久 null block。

        Returns:
            num_free_blocks: 空闲真实物理块数，不包含永久保留的 null block。
        """
        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
        """返回去除 null block 后的物理块使用率。

        Returns:
            usage: 去除 null block 后的物理块使用率；没有可分配真实块时为 0.0。
        """
        total_gpu_blocks = self.num_gpu_blocks - 1
        if not total_gpu_blocks:
            return 0.0
        return 1.0 - self.get_num_free_blocks() / total_gpu_blocks
