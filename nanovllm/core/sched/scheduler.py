"""连续批处理 Scheduler 的 MVP 实现。

Scheduler 不维护显式的 prefill/decode 阶段。每次调度只让每个请求的
``num_computed_tokens`` 尽量追赶 ``num_tokens``，因此同一个 batch 可以同时包含
长 prompt 的 chunked prefill、普通 decode 和被抢占后需要重算的请求。
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable

from nanovllm.compilation.cuda_graph import CUDAGraphStat
from nanovllm.config import NanovllmConfig
from nanovllm.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from nanovllm.core.kv_cache_utils import KVCacheBlock
from nanovllm.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from nanovllm.core.sched.request_queue import SchedulingPolicy, create_request_queue
from nanovllm.core.sched.utils import check_stop, remove_all
from nanovllm.engine import EngineCoreOutput, EngineCoreOutputs
from nanovllm.kv_cache_interface import KVCacheConfig
from nanovllm.metrics.perf import PerfStats
from nanovllm.metrics.stats import SchedulerStats
from nanovllm.outputs import ModelRunnerOutput
from nanovllm.request import Request, RequestStatus


class Scheduler:
    """管理请求队列、KV cache 分配与单步 token budget。

    每次 :meth:`schedule` 先服务已经运行的请求，再使用剩余预算接纳等待请求。
    当资源不足时，当前 MVP 按 FCFS 语义从 ``running`` 队尾抢占；被抢占请求的
    逻辑 token 序列保留，但物理 KV table 被释放，之后可通过 prefix cache 恢复。

    Attributes:
        max_num_running_reqs: 同时处于 RUNNING 状态的请求数上限。
        max_num_scheduled_tokens: 单次 schedule 允许交给模型执行的总 token budget。
        kv_cache_manager: 物理 KV blocks、prefix cache 和 request block tables 的
            唯一所有者。
        requests: 从 request ID 到活跃 Request 的权威映射。
        waiting: 尚未获得运行资源或被抢占的请求队列，按配置选择 FCFS 或优先级。
        running: 已拥有 block table、可在后续 schedule 中推进的请求列表。
        prev_step_scheduled_req_ids: 上一轮实际进入模型 batch 的请求 IDs，用于决定
            CachedRequestData 是否需要补发完整 token 序列。
        finished_req_ids: 本轮之前已结束、下一次 SchedulerOutput 应通知 Worker
            清理本地状态的请求 IDs。
    """

    def __init__(
        self,
        vllm_config: NanovllmConfig,
        kv_cache_config: KVCacheConfig,
        block_size: int,
        hash_block_size: int | None = None,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        """根据已规划的 KV cache 配置创建 Scheduler。

        Args:
            vllm_config: 全局推理配置，提供请求数和单轮 token budget 等调度限制。
            kv_cache_config: 当前 Worker 已确定的 KV cache 分配方案。其中
                ``num_blocks`` 是包含保留空块的物理总数，请求可用容量为其减一。
            block_size: Engine 解析的 scheduler_block_size，单位为 token。
            hash_block_size: Request 哈希粒度，None 时沿用 block_size；当前两者须相等。
            include_finished_set: 是否向各 client 输出独立的完成请求集合。
            log_stats: 是否记录并返回调度和前缀缓存统计。
        """
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.kv_cache_config = kv_cache_config
        self.parallel_config = vllm_config.parallel_config
        # max_num_running_reqs 是请求数量上限，与单轮 token budget 是两个独立约束。
        self.max_num_running_reqs: int = self.scheduler_config.max_num_seqs
        # 该预算限制单个 forward batch 的总 token 数，使长 prompt 不会无限独占计算。
        self.max_num_scheduled_tokens: int = (
            self.scheduler_config.max_num_scheduled_tokens
            if self.scheduler_config.max_num_scheduled_tokens is not None
            else self.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.block_size: int = block_size  # 调度对齐粒度，沿用 vLLM Scheduler 的字段名。
        self.hash_block_size: int = (
            block_size if hash_block_size is None else hash_block_size
        )  # 与 Engine 为 Request 创建的 hasher 保持一致。
        # 普通自回归文本生成每步只采样一个 token。
        self.num_sampled_tokens_per_step = 1
        # 当前 KVCacheManager 只维护一张 block table，因此只接受单 group。
        if len(kv_cache_config.kv_cache_groups) != 1:
            raise NotImplementedError(
                "nano-vLLM Scheduler currently supports exactly one KV cache group"
            )
        # num_blocks 是规划器根据 profiling 得到的整数容量，必须和 Worker 实际
        # 分配的每层 KV tensor page 数一致。
        self.log_stats: bool = log_stats
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            scheduler_block_size=self.block_size,
            hash_block_size=self.hash_block_size,
            enable_caching=self.cache_config.enable_prefix_caching,
            log_stats=self.log_stats,
            watermark=self.scheduler_config.watermark,
        )

        # requests 是活跃逻辑请求的权威表，调度输出更新进度时通过它回查 Request。
        self.requests: dict[str, Request] = {}
        # 策略只决定等待队列排序；当前 running 的遍历和队尾抢占保持同一实现。
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)
        except ValueError as e:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e
        # waiting 保存未运行或已抢占请求，FCFS 下抢占请求从队首恢复。
        self.waiting = create_request_queue(self.policy)
        self.skipped_waiting = create_request_queue(self.policy)
        # running 保存有物理 KV 映射的请求，顺序也定义当前 MVP 的 FCFS 服务顺序。
        self.running: list[Request] = []
        self.finished_req_ids_dict: dict[int, set[str]] | None = (
            defaultdict(set) if include_finished_set else None
        )
        # 用于构造 CachedRequestData.all_token_ids。
        self.prev_step_scheduled_req_ids: set[str] = set()
        # 下一次 schedule 时通知 Worker 清理已经结束的 request。
        self.finished_req_ids: set[str] = set()

        self.defer_block_free: bool = False
        self.sched_step_seq: int = 0
        self.processed_step_seq: int = 0
        self.deferred_frees: deque[tuple[int, list[KVCacheBlock]]] = deque()

        self._inflight_prefills: set[Request] = set()
    def add_request(self, request: Request) -> None:
        """注册一个新请求并放入 FCFS 等待队列。

        Args:
            request: 由上层创建、状态必须为 ``RequestStatus.WAITING`` 的逻辑请求。

        Raises:
            ValueError: 同一个 ``request_id`` 已经注册时抛出。
            AssertionError: request 不是 WAITING 状态时抛出，防止同一 Request 被
                重复入队或错误复用。

        Side Effects:
            将 request 写入 ``requests`` 权威表，并追加到 ``waiting`` 队列末尾。
        """
        if request.request_id in self.requests:
            raise ValueError(
                f"Duplicate request_id:{request.request_id}"
            )

        # Scheduler 首次接手请求时，它不能已经占有物理资源或处于完成状态。
        assert request.status == RequestStatus.WAITING
        self.requests[request.request_id] = request
        # 请求进入对应的策略队列，不直接依赖底层 deque 或 heap。
        self.waiting.add_request(request)
    def finish_requests(
        self,
        request_ids: str | Iterable[str] | None,
        finished_status: RequestStatus,
    ) -> list[tuple[str, int]]:
        """按 vLLM 的两阶段顺序处理外部结束信号并释放请求。

        Args:
            request_ids: 单个 ID、ID 集合，或表示结束全部请求的 None。
            finished_status: 要设置的 FINISHED_* 状态。

        Returns:
            finished_requests: 本次实际结束的 (request_id, client_index)，忽略未知或已结束的请求。

        Side Effects:
            先批量移除队列成员，再修改状态并统一释放资源，保留队列策略类型。
        """
        # 外部结束入口只接受终止状态，避免把活跃请求从队列移走却仍标为 RUNNING。
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = self.requests.keys()

        # 先收集对象，不能在迭代 request_ids 的映射视图时同步删除 requests。
        running_requests_to_remove: set[Request] = set()
        waiting_requests_to_remove: list[Request] = []
        valid_requests: list[Request] = []
        # 未知 ID 和已结束请求视为幂等取消，不重复释放资源。
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                continue
            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            else:
                waiting_requests_to_remove.append(request)

        # 先批量移除成员，避免请求完成后遗留在 running 或策略队列中。
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)
            self.skipped_waiting.remove_requests(waiting_requests_to_remove)
        # 队列已统一清理，再标记终止并释放映射，避免残留可被再次调度的成员。
        for request in valid_requests:
            request.status = finished_status
            if finished_status == RequestStatus.FINISHED_ABORTED:
                request.stop_reason = "abort"
            self._free_request(request)
        return [(r.request_id, r.client_index) for r in valid_requests]
    def _free_request(self, request: Request) -> None:
        """按上游顺序清理 prefill 状态、登记完成通知并释放块引用。

        当前没有 KV connector，省略传输完成通知、延迟传输释放和 kv_xfer_params。
        """
        assert request.is_finished()
        self._inflight_prefills.discard(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)  # 下一轮通知 Worker 删除请求镜像。
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)
        self._free_blocks(request)

    def _free_blocks(self, request: Request) -> None:
        """先释放或移交物理引用，再删除 Scheduler 的逻辑请求。"""
        assert request.is_finished()
        self._free_request_blocks(request)
        del self.requests[request.request_id]

    def _free_request_blocks(self, request: Request) -> None:
        """按 vLLM 的 fence 判断释放块，仍有在途写入时移交块对象。"""
        if not self.defer_block_free or (
            request.last_sched_seq <= self.processed_step_seq
        ):
            self.kv_cache_manager.free(request)
            return
        blocks = self.kv_cache_manager.pop_blocks_for_free(request)
        if blocks:
            self.deferred_frees.append((self.sched_step_seq, blocks))

    def _drain_deferred_frees(self) -> None:
        """按上游队首 fence 顺序回收，逻辑尾部块优先归还物理池。"""
        while self.deferred_frees:
            fence, _ = self.deferred_frees[0]
            if fence > self.processed_step_seq:
                break
            _, blocks = self.deferred_frees.popleft()
            # 单 Manager 合并了 BlockPool，此处只去掉上游的 block_pool 转发层。
            self.kv_cache_manager.free_blocks(reversed(blocks))

    def is_finish(self) -> bool:
        """判断 Scheduler 是否已经没有任何待处理或运行中的请求。

        Returns:
            is_finished: 当 ``waiting`` 和 ``running`` 均为空时返回 ``True``。完成请求在上层
                清理后会从这两个容器移除，因此无需额外遍历 ``requests``。
        """
        return not self.waiting and not self.running

    def has_requests(self) -> bool:
        """判断 Engine 是否还需要执行调度或通知 Worker 清理请求。

        沿用 vLLM 的语义：除了 waiting/running 中尚未完成的请求，上一轮产生的
        finished_req_ids 也需要一次执行端调用。它与 is_finish() 的区别在于：
        所有用户请求已经完成时，仍可能有尚未送达 Worker 的清理通知。

        Returns:
            has_requests: 有活跃请求，或有等待下发的结束/取消通知时返回 True。
        """
        return not self.is_finish() or bool(self.finished_req_ids)

    def _preempt_request(self, request: Request) -> None:
        """抢占一个运行中请求，释放其物理 KV 资源并重新排队。

        当前实现不做 swap 或 CPU offload：被抢占请求的 block table 会完全释放，
        因而恢复时必须从 token 零重新计算，或通过 prefix cache 命中一段完整前缀。

        Args:
            request: 必须处于 ``RUNNING`` 状态、且已拥有 KV block table 的请求。

        Raises:
            AssertionError: request 不是 RUNNING 状态时抛出，避免对未分配资源的
                请求重复释放。

        Side Effects:
            释放 request 的 KV blocks，将状态改为 PREEMPTED，清零计算进度并把请求
            插入 ``waiting`` 队首，以便资源允许时优先恢复。
        """
        assert request.status == RequestStatus.RUNNING

        # 释放当前物理映射，或按已有 fence 协议移交块引用。
        self._free_request_blocks(request)
        self._inflight_prefills.discard(request)
        # PREEMPTED 区别于首次 WAITING，供 Worker 选择恢复而非首次建档的协议。
        request.status = RequestStatus.PREEMPTED
        # 所有物理 KV 均释放后，旧 num_computed_tokens 不再对应有效缓存。
        request.num_computed_tokens = 0
        # 在途总量保留；将尚未回收的工作标记为旧执行阶段的结果。
        # 使用赋值，避免重复计算已经包含在总量中的旧份额。
        request.num_stale_output_tokens = request.num_in_flight_tokens
        # 当前执行阶段已结束，旧生成位置不再作为新调度的目标。
        request.num_output_placeholders = 0
        # 抢占计数可用于后续可观测性和防止长期饥饿的策略。
        request.num_preemptions += 1
        # FCFS 放到队首重试；priority 队列仍按请求原有排序键重新入堆。
        self.waiting.prepend_request(request)

    def schedule(self) -> SchedulerOutput:
        """构造混合批次快照，并在返回前提交一次调度进度。

        先尝试推进 running，再在剩余 token budget 和 KV 容量内接纳 waiting。
        快照保存 forward 前的请求位置，_update_after_schedule() 随后推进权威进度；
        本方法只安排工作，不执行模型，也不读取 GPU token。

        Returns:
            scheduler_output: 新请求或已有请求的输入计划、每请求计算量、block 更新和清理通知。
                即使没有可执行 token，空计划仍可能携带 finished_req_ids。

        Side Effects:
            分配或释放 KV block，修改 waiting/running 和请求状态，并推进
            num_computed_tokens、num_in_flight_tokens；异步子类还增加输出占位。
            更新上一轮调度集合并轮换完成通知，不把返回计划中的标量自动改为新值。
        """
        # 没有活跃请求也可能还有 Worker 清理工作；零 token 输出必须保留完成通知。
        if self.is_finish():
            scheduler_output = SchedulerOutput.make_empty()
            scheduler_output.finished_req_ids = self.finished_req_ids
            self._update_after_schedule(scheduler_output)  # 同样轮换清理通知。
            return scheduler_output

        # token_budget 是本轮剩余可执行 token 数，所有请求共享这一全局预算。
        token_budget = self.max_num_scheduled_tokens
        # scheduled_running_reqs 收集本轮推进的既有 RUNNING 请求。
        scheduled_running_reqs: list[Request] = []
        # scheduled_new_reqs 收集首次从 WAITING 接纳、需在 Worker 首次建档的请求。
        scheduled_new_reqs: list[Request] = []
        # scheduled_resumed_reqs 收集由 PREEMPTED 重新获得资源的请求。
        scheduled_resumed_reqs: list[Request] = []
        # 请求到块对象包装；新建/恢复保存全表，普通 running 保存本轮新增块。
        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        # num_scheduled_tokens 是本轮调度决策的核心：request_id -> 推进 token 数。
        num_scheduled_tokens: dict[str, int] = {}
        # preempted_reqs 记录本轮因 KV 空间不足被迫释放物理资源的请求。
        preempted_reqs: list[Request] = []

        # req_index 指向尚未尝试推进的 running 请求，避免 list 遍历时修改队尾出错。
        req_index = 0

        # 先推进已有请求，优先保证连续 decode/chunked prefill 的服务公平性。
        while req_index < len(self.running) and token_budget > 0:
            # request 是当前 FCFS 顺序下待推进的运行中请求。
            request = self.running[req_index]
            # 保留上游的 placeholder 长度边界，避免结果尚未回收时多调度一步。
            if (
                request.num_output_placeholders > 0
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                req_index += 1
                continue
            # 去掉投机 token 后，目标位置由真实 token 和待确认输出占位组成。
            num_new_tokens = (
                request.num_tokens
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            # token budget 限制本请求本轮规模，支持长 prompt 的 chunked prefill。
            num_new_tokens = min(num_new_tokens, token_budget)
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len
                - request.num_computed_tokens
                - self.num_sampled_tokens_per_step,
            )
            # 没有新 token 时可能是 Engine 尚未写入新的采样结果，跳过并检查下一个。
            if num_new_tokens == 0:
                req_index += 1
                continue

            # 申请资源失败时，从运行队尾释放请求，直到当前请求可分配或自己被抢占。
            while True:
                # new_blocks 是本轮新获得的物理块；None 表示空闲池容量不足。
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request=request,
                    num_new_tokens=num_new_tokens,
                )

                # 分配成功后当前请求可安全进入本轮 batch。
                if new_blocks is not None:
                    break

                # 当前抢占统一选 running 队尾；waiting 的策略不改变受害者选择。
                # preempted_req 是队尾的低服务优先级请求；在 FCFS 下不会是已调度前缀。
                preempted_req = self.running.pop()
                self._preempt_request(request=preempted_req)
                preempted_reqs.append(preempted_req)

                # 当前请求自己也被移除时，说明释放全部 running 请求仍无法满足其需求。
                if preempted_req is request:
                    break

            # 最后一次分配仍失败时，不再尝试后续 running 请求，转入 waiting 接纳阶段。
            if new_blocks is None:
                break

            # 分配成功，记录该 RUNNING 请求本轮的逻辑 token 推进量。
            scheduled_running_reqs.append(request)
            # request_id 是输出映射使用的稳定 key，避免后续列表顺序耦合。
            request_id = request.request_id
            # 这是running request。只保存新分配的 block
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            # 从全局预算扣除已承诺给当前请求的计算量。
            token_budget -= num_new_tokens
            req_index += 1

        # 与上游一致：发生抢占的本轮不再接纳等待请求，恢复留到下一轮。
        if not preempted_reqs:
            # 暂存本轮尚不能恢复的请求；循环结束后重新归还 waiting。
            step_skipped_waiting = create_request_queue(self.policy)
            while self.waiting and token_budget > 0:
                # 并发请求数已满时，不能再把等待请求移动至 RUNNING。
                if len(self.running) == self.max_num_running_reqs:
                    break

                # request 只 peek 队首；FCFS 下它不可进入时不能跳过它调度后继请求。
                request = self.waiting.peek_request()
                # request_id 后续用于写入本轮 token 决策映射。
                request_id = request.request_id

                # waiting 队列只应保存首次等待或被抢占的请求，其他状态表示状态机错误。
                assert request.status in (
                    RequestStatus.WAITING,
                    RequestStatus.PREEMPTED,
                )
                if request.num_stale_output_tokens > 0:
                    # 旧结果还可能扩展真实 token 历史，此时不能重新采样相同位置。
                    # 暂时移走该请求，让其他已经就绪的请求继续竞争预算。
                    step_skipped_waiting.prepend_request(
                        self.waiting.pop_request()
                    )
                    continue
                # MVP 不支持远端 KV；waiting/preempted 都从本地 prefix cache 查询。
                # new_computed_blocks 是可共享的完整 prefix 物理块，尚未增加引用计数。
                new_computed_blocks, num_new_local_computed_tokens = (
                    self.kv_cache_manager.get_computed_blocks(request)
                )
                # num_computed_tokens 是本轮 forward 前可直接视为已有 KV 的总逻辑进度。
                num_computed_tokens = num_new_local_computed_tokens
                # num_new_tokens 是扣除 prefix cache 命中后仍需模型执行的 token 数。
                num_new_tokens = request.num_tokens - num_computed_tokens
                # 剩余全局预算决定本请求本轮最多可推进多少 token。
                num_new_tokens = min(
                    num_new_tokens,
                    token_budget,
                )

                # 正常非空请求会因“最后一个 token block 不命中”保证大于零；此处保守防御。
                if num_new_tokens == 0:
                    break

                # 与上游相同，先保护缓存命中，再分配新增物理块。
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request=request,
                    num_new_tokens=num_new_tokens,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    has_scheduled_reqs=bool(num_scheduled_tokens),
                )
                # 队首资源不足时保持 FCFS，不跳过它让后继请求插队。
                if new_blocks is None:
                    break

                # 容量检查和分配均成功后，才从 waiting 队列正式移除这个请求。
                request = self.waiting.pop_request()
                # 重新读取 request_id，使后续代码始终与出队对象绑定。
                request_id = request.request_id

                # New 与 PREEMPTED 的 worker 数据传输语义不同。
                if request.status == RequestStatus.WAITING:
                    if request.prefill_stats is not None:
                        request.prefill_stats.set(
                            request.num_prompt_tokens, num_computed_tokens
                        )
                    scheduled_new_reqs.append(request)
                else:
                    scheduled_resumed_reqs.append(request)


                # prefix cache hit 是本轮模型计算前就已确定的进度。
                request.num_computed_tokens = num_computed_tokens
                # 分配完成后请求拥有有效 block table，可以进入后续 running 调度循环。
                request.status = RequestStatus.RUNNING
                self.running.append(request)
                # 必须拿到完整 table
                req_to_new_blocks[request_id] = (
                    self.kv_cache_manager.get_blocks(request_id)
                )
                # 确认调度
                num_scheduled_tokens[request_id] = num_new_tokens
                # 扣除本请求已承诺的模型执行预算。
                token_budget -= num_new_tokens
            if step_skipped_waiting:
                # FCFS 恢复原有相对顺序；priority 队列按其排序规则重新插入。
                self.waiting.prepend_requests(step_skipped_waiting)
        # 构建调度输出

        # new_reqs_data 是首次发送给 Worker 的完整请求快照列表。
        new_reqs_data = [
            NewRequestData.from_request(
                request=req,
                block_ids=req_to_new_blocks[req.request_id].get_block_ids()[0],
            )
            # req 是本轮刚从 WAITING 接纳的请求，Worker 尚未保存其本地状态。
            for req in scheduled_new_reqs
        ]

        # cached_requests 合并普通运行与抢占恢复请求；两者都已存在于 Worker 协议中。
        cached_requests = (
            scheduled_running_reqs
            + scheduled_resumed_reqs
        )
        # resumed_req_ids 让 Worker 区分正常增量更新与需要重建的已抢占请求。
        resumed_req_ids = {
            request.request_id
            for request in scheduled_resumed_reqs
        }

        # all_token_ids 为跨轮未调度的已有请求补发完整 token 序列，避免 Worker 状态断档。
        all_token_ids = {
            request.request_id: request.all_token_ids.copy()
            for request in cached_requests
            if request.request_id not in self.prev_step_scheduled_req_ids
        }
        # 与 cached_requests / req_ids 一一对齐。
        cached_new_block_ids: list[list[int] | None] = []

        for request in cached_requests:
            block_ids = req_to_new_blocks[request.request_id].get_block_ids(
                allow_none=True
            )

            # 空列表表示普通 running request 本轮没有跨 block 边界；
            # Worker 无需更新其已有 block table，因此协议中使用 None。
            cached_new_block_ids.append(
                block_ids[0] if block_ids is not None else None
            )
        # cached_reqs_data 是 Worker 已建档请求的列式批次描述，所有列表按 req_ids 对齐。
        cached_reqs_data = CachedRequestData(
            # req_ids 定义其余按请求排列字段的稳定索引顺序。
            req_ids=[
                request.request_id
                for request in cached_requests
            ],
            resumed_req_ids=resumed_req_ids,
            # 当前没有 PP，因此没有只传递给后续 stage 的增量 token 数据。
            new_token_ids=[],
            all_token_ids=all_token_ids,
            # 每项与 req_ids 对齐：普通 running request 仅发送新增 blocks，
            # resumed request 则发送完整 table，供 Worker 覆盖失效的旧映射。
            new_block_ids=cached_new_block_ids,
            # 这些值必须是本轮 forward 之前的进度，故在 _update_after_schedule() 前读取。
            num_computed_tokens=[
                request.num_computed_tokens
                for request in cached_requests
            ],
            # 逻辑输出长度包含在途占位，供 Worker 对齐输出历史与输入位置。
            num_output_tokens=[
                request.num_output_tokens + request.num_output_placeholders
                for request in cached_requests
            ],
        )
        # scheduler_output 固化本轮决策；Engine/Worker 应只根据它构建模型输入。
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            # total_num_scheduled_tokens 是 batch 级总工作量，便于执行和统计使用。
            total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
            # 传递当前 set 对象，随后必须替换而不是 clear，避免破坏本输出快照。
            finished_req_ids=self.finished_req_ids,
            # 记录本轮抢占事件；Worker 从调度集合识别缺席请求，保留其请求镜像。
            preempted_req_ids={req.request_id for req in preempted_reqs},
        )

        # 非空批次才产生 GPU 写入 fence；快照此时已保存执行前进度。
        if self.defer_block_free and scheduler_output.total_num_scheduled_tokens > 0:
            self.sched_step_seq += 1
        self._update_after_schedule(scheduler_output)
        return scheduler_output

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        """保存计划后推进请求进度，并维护本轮与下一轮之间的通知边界。

        Args:
            scheduler_output: 已保存执行前标量和本轮输入数的计划。

        Side Effects:
            按本轮输入数增加 computed/in-flight 计数，判断是否仍有 prefill/recompute
            历史待处理。替换完成通知集合和上一轮调度 ID 集合，不原地清空旧快照。
            可选延迟释放策略开启时还记录最近一次 KV 写入编号。
        """
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            # 快照已经保存执行前进度，现在推进本轮承诺执行的位置。
            request.num_computed_tokens += num_scheduled_token
            # 包含 prefill 和 decode，不要求本轮产生有效采样输出。
            request.num_in_flight_tokens += num_scheduled_token
            if self.defer_block_free:
                # 记录最后一次提交的写入批次，供延迟释放判断使用。
                request.last_sched_seq = self.sched_step_seq
            # 此时 AsyncScheduler 尚未增加当前批次产生的输出占位。
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
            if not request.is_prefill_chunk:
                self._inflight_prefills.discard(request)
        # 更换集合，避免修改已经交给执行端的调度快照。
        self.finished_req_ids = set()  # 不能 clear，已返回的快照仍持有原集合。
        self.prev_step_scheduled_req_ids = set(num_scheduled_tokens)


    def _handle_stopped_request(self, request: Request) -> bool:
        """当前没有可恢复的流式输入，请求停止即结束生命周期。

        Returns:
            finished: 当前实现中停止请求直接结束生命周期，返回 True。
        """
        return True

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[int],
        is_stale: bool = False,
    ) -> tuple[list[int], bool]:
        """追加真实输出并检查停止条件，旧阶段结果不改写新阶段计算进度。

        Args:
            request: 收到输出时仍受 Scheduler 管理的请求。
            new_token_ids: 本批新增真实 token，遇到停止条件可能原地截短。
            is_stale: 是否属于抢占前的旧批次，旧结果仍可扩展真实历史。

        Returns:
            (new_token_ids, stopped):
                new_token_ids: 本次实际接受的输出 token 列表。
                stopped: 本次接受输出后触发停止条件时为 True。

        Side Effects:
            更新完整/输出 token 历史及 block hash，可能设置终止状态。非旧结果
            若在列表中途停止，则回退未接受后缀对应的计算进度；单 token 路径回退为零。
        """
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                if not is_stale:
                    # 停止后未交付的后缀不再计入当前执行阶段的计算进度。
                    # 当前单 token 采样时，这个回退量为零。
                    request.num_computed_tokens -= (
                        len(new_token_ids) - num_new
                    )

                del new_token_ids[num_new:]
                break
        return new_token_ids, stopped

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        """按原始计划回收一批结果，更新请求并按客户端聚合真实增量。

        Args:
            scheduler_output: 与本结果同批的原始执行计划，包含当时的输入数。
            model_runner_output: 已完成 CPU 回传的结果及本批请求行映射。

        Returns:
            engine_core_outputs: client_index 到 EngineCoreOutputs 的映射；无有效增量时可以为空，
                开启统计或完成通知时也可能只返回这些辅助字段。

        Side Effects:
            按输入量减少在途计数，区分抢占前旧结果，追加真实输出并判断停止。
            完成请求释放资源、离开队列，清理通知留给下次 Worker 更新。
            异步子类在输出 hook 中减少占位；此方法不会直接修补 Runner 的 CPU token 行。
            开启统计时消费当前窗口的缓存统计，因此结果必须按 Engine 队列顺序回收。
        """
        sampled_token_ids = model_runner_output.sampled_token_ids
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        cudagraph_stats = model_runner_output.cudagraph_stats

        # 按提交顺序消费已完成的非空结果，才能确认此前的 GPU 写入 fence。
        if self.defer_block_free and scheduler_output.total_num_scheduled_tokens > 0:
            self.processed_step_seq += 1
            self._drain_deferred_frees()

        perf_stats: PerfStats | None = None
        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)

            output_is_stale = False
            if request is not None:
                # 每个完成批次都回收计算量，纯 prefill 的空采样结果也必须处理。
                request.num_in_flight_tokens -= num_tokens_scheduled

                if request.num_stale_output_tokens > 0:
                    # Engine 按提交顺序回收，因此先返回的是抢占前的旧工作。
                    output_is_stale = True
                    request.num_stale_output_tokens -= num_tokens_scheduled
                    assert request.num_stale_output_tokens >= 0

            if request is None or request.is_finished():
                # 已取消或结束的请求不再追加迟到输出。
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )

            stopped = False
            new_token_ids = generated_token_ids
            status_before_stop = request.status

            if new_token_ids:
                # 旧结果仍可交付，但后续不能扣减新执行阶段的输出占位。
                new_token_ids, stopped = self._update_request_with_output(
                    request,
                    new_token_ids,
                    is_stale=output_is_stale,
                )
            finish_reason = None
            if stopped:
                finish_reason = request.get_finished_reason()
                finished = self._handle_stopped_request(request)
                if finished:
                    self._free_request(request)
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)
            if (
                new_token_ids
                or stopped
            ):
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=new_token_ids,
                        finish_reason=finish_reason,
                        stop_reason=request.stop_reason,
                        prefill_stats=request.take_prefill_stats(),
                    )
                )
        # 与上游一致，先收集停止请求，遍历结束后统一更新队列。
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # 已抢占请求也可能在迟到结果中结束，使用策略队列接口移除。
            self.waiting.remove_requests(stopped_preempted_reqs)
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }
        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # 将上次输出之后结束的请求归属到对应 client。
            for client_index, finished_set in finished_req_ids.items():
                # 即使本轮没有 token 输出，仍然传递该 client 的完成集合。
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set
                    )
            finished_req_ids.clear()

        if (
            stats := self.make_stats(
                cudagraph_stats, perf_stats
            )
        ) is not None:
            # 一轮统计只附加到一个 client，避免前端重复计数。
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # 纯 prefill chunk 也需要能够传递统计。
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs

    def make_stats(
        self,
        cudagraph_stats: CUDAGraphStat | None = None,
        perf_stats: PerfStats | None = None,
    ) -> SchedulerStats | None:
        """按当前队列与 KV 状态创建一次统计快照，关闭统计时返回 None。

        Args:
            cudagraph_stats: 可选的本批 Graph 容量观测结果。
            perf_stats: 可选的本批计算/访存估算，不在此主动采集。

        Returns:
            stats: 当前请求数量、KV 使用比例与前缀命中统计组成的 SchedulerStats。

        Side Effects:
            make_prefix_cache_stats() 转交并替换缓存统计窗口；重复调用会消费新的窗口，
            因此不能把该接口当作不改变状态的任意查询。
        """
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            num_skipped_waiting_reqs=len(self.skipped_waiting),
            kv_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            cudagraph_stats=cudagraph_stats,
            perf_stats=perf_stats,
        )
