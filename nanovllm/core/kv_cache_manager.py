"""最小 KV cache manager。

``Request`` 保存逻辑 token 进度；``KVCacheManager`` 保存
``request_id -> block table`` 的物理映射；``KVCacheBlock`` 保存每个物理 block
的引用计数和 prefix cache 元数据。三者分离后，请求抢占只需释放物理映射，逻辑
请求对象本身仍可保留在 Scheduler 中等待恢复。
"""

from __future__ import annotations

from collections import deque


from nanovllm.core.kv_cache_utils import KVCacheBlock,BlockHash
from nanovllm.request import Request



class KVCacheManager:
    """管理物理 KV blocks、请求 block table 和本地 prefix cache。

    一个 block table 的第 ``i`` 项对应逻辑 token 区间
``[i * block_size, (i + 1) * block_size)``。多个 request 可以通过引用计数
共享已缓存的完整 prefix block；一旦某个空闲 block 要写入新内容，旧 hash 映射
必须被删除，避免 prefix cache 指向错误的物理内存。

    Attributes:
        block_size: 一个物理 block 容纳的 token 数，也是 prefix cache 的最小粒度。
        blocks: 按 ``block_id`` 索引的所有物理 block 元数据。
        free_block_ids: 引用计数为零、可被重新分配的 block ID 队列。队列中的
            block 可能仍保留 prefix hash。
        used_block_ids: 引用计数大于零、当前正在被至少一个请求使用的 block ID 集合。
        hash_to_block_id: 从前缀相关 block hash 到物理 block ID 的本地 prefix
            cache 索引。
        req_to_block_table: 每个活跃 request 的完整逻辑 block table。Request 被
            free 或抢占后，对应条目会被删除。
        null_block: ID 为 0 的永久保留空块，不属于空闲队列或请求占用集合。
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
    ) -> None:
        """创建一个由固定数量物理 blocks 组成的 KV cache 池。

        Args:
            num_blocks: 包含 null block 的物理总数，与 GPU KV tensor 容量一致；
                请求最多使用 num_blocks - 1 个 block。
            block_size: 每个 block 可保存的 token 数。prefix cache 只复用完整的
                block，因此它也决定缓存命中的最小粒度。
        """
        if num_blocks < 1:
            raise ValueError("num_blocks must include at least the reserved null block")
        self.num_gpu_blocks = num_blocks  # 物理总数包含 block 0，不能当作请求可用容量。
        # block_size 是全部 token->block 位置计算共享的基础单位。
        self.block_size: int = block_size
        # block_id 与列表下标相同，保证通过 ID 可以 O(1) 找到元数据。
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(block_id)
            for block_id in range(num_blocks)
        ]
        # 初始时所有 block 均未被引用，按 ID 顺序放入空闲队列。
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # 与 vLLM 的 BlockPool 一致，从空闲队列拿走第一块并永久保留。
        self.null_block = self.blocks[self.free_block_ids.popleft()]
        self.null_block.is_null = True
        # 此集合与 free_block_ids 对可分配块互补；null block 不计入请求占用率。
        self.used_block_ids: set[int] = set()
        # 只有完整 block 才能进入 prefix cache。
        self.hash_to_block_id: dict[BlockHash, int] = {}
        # 物理布局属于 KVCacheManager，而不是 Request。
        self.req_to_block_table: dict[str, list[int]] = {}

    def get_num_free_blocks(self) -> int:
        """返回可继续分配给请求的块数，已排除永久保留的 null block。"""
        return len(self.free_block_ids)

    @property
    def usage(self) -> float:
        """返回请求对可用块的占用比例；分母为物理总数减去一个 null block。"""
        total_gpu_blocks = self.num_gpu_blocks - 1
        if total_gpu_blocks == 0:
            return 0.0  # 仅有空块时没有可用请求容量，与上游空池统计一致。
        return 1.0 - self.get_num_free_blocks() / total_gpu_blocks

    @staticmethod
    def _num_blocks_for_tokens(
        num_tokens: int,
        block_size: int,
    ) -> int:
        """计算容纳给定 token 数所需的物理 block 数。

        Args:
            num_tokens: 需要保存 KV 的逻辑 token 数。
            block_size: 单个物理 block 的 token 容量。

        Returns:
            向上取整后的 block 数；当 ``num_tokens`` 为零时返回零。
        """
        # 加上 block_size - 1 后整除，等价于正整数的向上取整。
        return (num_tokens + block_size - 1) // block_size


    def get_block_table(self, request_id: str) -> list[int]:
        """返回一个活跃请求当前的完整物理 block table。

        Args:
            request_id: 要查询的请求 ID。

        Returns:
            从逻辑 block 下标到物理 block ID 的映射列表。返回的是内部列表；需要
            跨 Scheduler/Worker 边界传递时，调用方必须创建副本。

        Raises:
            KeyError: 请求当前没有已分配的物理 block table 时抛出。
        """
        return self.req_to_block_table[request_id]

    def _acquire_cached_block(self, block_id: int) -> None:
        """为新请求增加一个已命中 prefix cache block 的引用计数。

        Args:
            block_id: 命中的物理 prefix cache block ID。

        Side Effects:
            当 block 原来空闲时，把它从 ``free_block_ids`` 移入
            ``used_block_ids``；无论原来是否被使用，都会将 ``ref_cnt`` 加一。
        """
        # block 是待共享的物理缓存元数据，可能当前为空闲也可能已被其他请求引用。
        block = self.blocks[block_id]
        assert not block.is_null, "Reserved null block cannot enter the prefix cache"

        if block.ref_cnt == 0:
            # 空闲但可缓存的 block 一旦被复用，也需要从可重新分配池中取走。
            self.free_block_ids.remove(block_id)
            self.used_block_ids.add(block_id)

        # 每附加到一个新 request block table，就新增一份逻辑引用。
        block.ref_cnt += 1

    def get_computed_blocks(
        self,
        request: Request,
    ) -> tuple[list[int], int]:
        """查找 request 可复用的最长连续完整 prefix-cache blocks。

        Request 在创建和追加 output token 时已经增量维护 ``block_hashes``。本方法
        只消费这些逻辑 hash 查询物理 prefix cache，不再从 token 零开始重新 hash。

        Args:
            request: 处于 WAITING 或 PREEMPTED 状态的逻辑请求。

        Returns:
            一个二元组 ``(cached_block_ids, num_cached_tokens)``：
            - ``cached_block_ids`` 是连续命中的物理 block IDs。
            - ``num_cached_tokens`` 是命中 block 对应的已计算 token 数。

        Raises:
            RuntimeError: Request 缺少应有的完整 block hash 时抛出。
        """
         # 最后一个 token 必须由模型执行，才能得到采样下一个 token 所需的 logits。
        max_cache_hit_length = max(request.num_tokens - 1, 0)
        # 只有完整 block 能参与 prefix cache 命中。
        num_cacheable_blocks = (
            max_cache_hit_length // self.block_size
        )
        if len(request.block_hashes) < num_cacheable_blocks:
            raise RuntimeError(
                f"Request {request.request_id} has stale block hashes: "
                f"need {num_cacheable_blocks}, "
                f"got {len(request.block_hashes)}"
            )
        # cached_block_ids 按逻辑 block 顺序保存最长连续命中的物理 blocks。
        cached_block_ids: list[int] = []
        # 从 prompt 开头依次查询；prefix cache 一旦断开，后续 block 不能独立复用。
        for block_index in range(num_cacheable_blocks):
            # block_hash 是 Request 预先计算的当前完整逻辑 block key。
            block_hash = request.block_hashes[block_index]
            # block_id 是 prefix cache 中对应的候选物理 block。
            block_id = self.hash_to_block_id.get(block_hash)

            # 这个 block 没有缓存过，后缀也无法跳过计算。
            if block_id is None:
                break
            

            cached_block_ids.append(block_id)

        # 命中的每个物理 block 均对应恰好 block_size 个已计算 token。
        return cached_block_ids, len(cached_block_ids) * self.block_size

    def _allocate_new_block(self) -> int:
        """从空闲池取出一个 block，并清理它遗留的 prefix cache 索引。

        调用方必须先检查 ``free_block_ids`` 容量充足。空闲 block 可能仍保存旧
        prefix cache，因此在其物理 KV 槽位被新请求覆写前，必须先使旧 hash 失效。

        Returns:
            已标记为使用中、引用计数初始化为一的物理 block ID。

        Side Effects:
            从 ``free_block_ids`` 移除一个 ID，更新 ``used_block_ids``，删除可能
            存在的旧 hash 映射，并清理 block 的 hash 
        """
        # block_id 是从空闲队列头部取出的、即将承载新 KV 内容的物理编号。
        block_id = self.free_block_ids.popleft()
        # block 是该物理编号对应的 CPU 侧元数据。
        block = self.blocks[block_id]

        # 只有无人引用的 block 才能被重新写入新的 KV 内容。
        assert block.ref_cnt == 0
        if (
            block.block_hash is not None
            and self.hash_to_block_id.get(block.block_hash) == block_id
        ):
            # 新 token 会覆写这个物理槽位；保留旧映射会让 prefix cache 重新找到它，可能导致错误的复用
            del self.hash_to_block_id[block.block_hash]

        # 新分配给一个 request 后，物理 block 立刻拥有一份逻辑引用。
        block.ref_cnt = 1
        # reset_hash 专门表达“旧物理内容已失效”，不能通过 hash setter 覆写。
        block.reset_hash()

        # 从现在起 block 不再可重新分配，直到所有请求引用都被释放。
        self.used_block_ids.add(block_id)
        return block_id

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
    ) -> None:
        """将已成功写入物理 KV 的完整逻辑 blocks 提交到 prefix cache。

        Request 提供逻辑 hash，KVCacheManager 将其绑定到已经写入 KV tensor 的物理
        block。这使逻辑 token 身份与物理内存位置保持分离。

        Args:
            request: 已拥有物理 block table 的运行中请求。
            num_tokens: 已成功写入物理 KV cache 的 token 数。

        Raises:
            KeyError: request 没有物理 block table 时抛出。
            RuntimeError: 已完成物理 block 数超过 Request hash 数时抛出。
            AssertionError: 同一物理 block 绑定了不同逻辑 hash 时抛出。
        """
        # block_table 是 request 的逻辑 block 下标到物理 block ID 的映射。
        block_table = self.req_to_block_table[request.request_id]
        # num_full_blocks 限制循环只处理 KV 内容已经完整写入的 block。
        num_full_blocks = num_tokens // self.block_size
        if num_full_blocks > len(request.block_hashes):
            raise RuntimeError(
                f"Request {request.request_id} has stale block hashes: "
                f"need {num_full_blocks}, "
                f"got {len(request.block_hashes)}"
            )

        for block_index in range(num_full_blocks):
            # Request 已预先算好的逻辑 block key。
            block_hash = request.block_hashes[block_index]

            # block_id 找到存放当前逻辑位置 KV 的物理槽位。
            block_id = block_table[block_index]
            assert block_id != self.null_block.block_id, "Cannot cache the null block"
            # block 保存这个物理槽位的可复用缓存元数据。
            block = self.blocks[block_id]

            if block.block_hash is not None:
                # 同一物理 block 若已缓存，只能对应同一个逻辑前缀。
                assert block.block_hash == block_hash
                continue

            # 建立逻辑 hash 到物理 block 的 prefix-cache 索引。
            block.block_hash = block_hash
            self.hash_to_block_id[block_hash] = block_id

    def free(self, request: Request) -> None:
        """释放一个 request 对所有物理 KV blocks 的引用。

        抢占和请求结束都会调用本方法。它只释放 request 的 block table，不清理
        已完成完整 block 的 prefix hash；引用计数归零的 block 因而仍可作为空闲
        prefix cache 被后续请求重新引用。

        Args:
            request: 要释放物理 KV 资源的逻辑请求。

        Side Effects:
            删除 ``req_to_block_table`` 中的 request 条目，并减少其中每个 block 的
            ``ref_cnt``。引用归零的 block 会从 ``used_block_ids`` 移至
            ``free_block_ids``。
        """
        # block_table 是本请求曾拥有的所有物理 block；没有记录时按空表处理。
        block_table = self.req_to_block_table.pop(
            request.request_id,
            [],
        )

        # 反序释放，使逻辑尾部 block 更早回到空闲队列，符合原实现的复用顺序。
        for block_id in reversed(block_table):
            # block 是当前要减少一份请求引用的物理缓存元数据。
            block = self.blocks[block_id]
            assert not block.is_null, "Reserved null block cannot be freed by a request"
            block.ref_cnt -= 1
            # 引用计数不能为负；负数意味着重复释放或共享关系维护错误。
            assert block.ref_cnt >= 0

            # 最后一个引用消失后，block 可重用，但其完整 prefix hash 继续保留。
            if block.ref_cnt == 0:
                self.used_block_ids.remove(block_id)
                self.free_block_ids.append(block_id)

    def allocate_slots(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        cached_block_ids: list[int] | None = None,
    ) -> list[int] | None:
        """为一个请求本轮计算范围准备所需的物理 KV slots。

        对运行中请求，方法在既有 block table 尾部按需分配；对新请求或恢复请求，
        方法先附加已经命中的 prefix blocks，再为其余 token 分配新物理块。所有
        资源需求先统一检查，确保分配失败时不会留下部分修改的引用计数。会把前缀
        +分配的总 block table 加入 req_to_block_table 映射表。

        Args:
            request: 被调度的逻辑请求。
            num_computed_tokens: 本轮模型执行前已经有 KV 的 token 数，包含本地
                prefix cache 命中的 token。
            num_new_tokens: 本轮要交给模型 forward 的新增 token 数，必须大于零。
            cached_block_ids: 新/恢复请求命中的连续 prefix block IDs。运行中请求
                已有 block table，因此该参数必须为空。

        Returns:
            成功时返回本次新分配的物理 block IDs；若空闲池不足以同时附加缓存块
            和分配新块，则返回 ``None``，且不会改变 manager 状态。

        Raises:
            AssertionError: ``num_new_tokens`` 不为正，或运行中请求重复传入 prefix
                blocks 时抛出。
            RuntimeError: 既有 block table 已经长于本轮逻辑 token 范围时抛出。

        Side Effects:
            成功时更新 ``req_to_block_table``、相关 block 的 ``ref_cnt``，以及
            ``free_block_ids``/``used_block_ids`` 的资源归属。
        """
        # 0 token 不会产生 forward 进度，调用方必须在调度阶段过滤该情况。
        assert num_new_tokens > 0
        # request_id 是本次查询和写回 request block table 的唯一 key。
        request_id = request.request_id
        # existing_block_table 为运行中请求保留的物理映射；新请求尚不存在该条目。
        existing_block_table = self.req_to_block_table.get(
            request_id,
            [],
        )

        # 将可选参数标准化为空列表，后续容量计算与循环可统一处理。
        if cached_block_ids is None:
            cached_block_ids = []

        # 如果该请求已经运行，它的全部逻辑前缀必须已经写在既有 block table 中。
        if existing_block_table:
            # 那外部就不应该提供cached_block_ids
            assert not cached_block_ids, (
                "已有 block table 的 running request 不应重新附加 prefix blocks"
            )
            # block_table 直接复用内部列表，随后可在其尾部追加新物理 block。
            block_table = existing_block_table
        else:
            # 新/恢复请求从命中的物理 prefix blocks 开始构建自己的完整 block table。
            block_table = cached_block_ids.copy()

        # num_required_blocks 是本轮结束后容纳所有已计算 token 所需的总 block 数。
        num_required_blocks = self._num_blocks_for_tokens(
            num_tokens=num_computed_tokens + num_new_tokens,
            block_size=self.block_size,
        )
        # num_new_blocks 是总需求减去当前 table 已覆盖 prefix 后仍需新增的物理块数。
        num_new_blocks = num_required_blocks - len(block_table)
        if num_new_blocks < 0:
            raise RuntimeError(
                f"Request {request_id} has an invalid block table"
            )

        # 统计 prefix 命中 block 中当前无人引用、因此也要从空闲队列取走的数量。
        num_cached_blocks_to_acquire = sum(
            self.blocks[block_id].ref_cnt == 0
            for block_id in cached_block_ids
        )
        # num_required_free_blocks 是提交该请求前必须一次性预留的全部空闲块数量。
        num_required_free_blocks = (
            num_cached_blocks_to_acquire + num_new_blocks
        )
        # 资源不足时完全不修改状态，让 Scheduler 决定是否执行抢占。
        if len(self.free_block_ids) < num_required_free_blocks:
            return None

        # 新/恢复请求在资源确认后，才正式增加命中 prefix blocks 的引用计数。
        if not existing_block_table:
            for block_id in cached_block_ids:
                self._acquire_cached_block(block_id)

        # new_block_ids 只记录本轮真正新拿到的物理块，供未来 worker block diff 使用。
        new_block_ids: list[int] = []
        for _ in range(num_new_blocks):
            # block_id 是本次从空闲池获得并追加到逻辑尾部的新物理槽位。
            block_id = self._allocate_new_block()
            block_table.append(block_id)
            new_block_ids.append(block_id)

        # 至此资源分配完成，保存 request 的完整逻辑 block table。
        self.req_to_block_table[request_id] = block_table
        return new_block_ids
