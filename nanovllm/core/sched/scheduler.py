"""连续批处理 Scheduler 的 MVP 实现。

Scheduler 不维护显式的 prefill/decode 阶段。每次调度只让每个请求的
``num_computed_tokens`` 尽量追赶 ``num_tokens``，因此同一个 batch 可以同时包含
长 prompt 的 chunked prefill、普通 decode 和被抢占后需要重算的请求。
"""

from collections import deque

from nanovllm.config import NanovllmConfig
from nanovllm.core.kv_cache_manager import KVCacheManager
from nanovllm.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from nanovllm.kv_cache_interface import KVCacheConfig
from nanovllm.request import Request, RequestStatus


class Scheduler:
    """管理请求队列、KV cache 分配与单步 token budget。

    每次 :meth:`schedule` 先服务已经运行的请求，再使用剩余预算接纳等待请求。
    当资源不足时，当前 MVP 按 FCFS 语义从 ``running`` 队尾抢占；被抢占请求的
    逻辑 token 序列保留，但物理 KV table 被释放，之后可通过 prefix cache 恢复。

    Attributes:
        max_num_running_reqs: 同时处于 RUNNING 状态的请求数上限。
        max_num_scheduled_tokens: 单次 schedule 允许交给模型执行的总 token budget。
        eos: 模型 EOS token ID；后续完成判定逻辑会使用它。
        kv_cache_manager: 物理 KV blocks、prefix cache 和 request block tables 的
            唯一所有者。
        requests: 从 request ID 到活跃 Request 的权威映射。
        waiting: 尚未获得运行资源或被抢占的请求队列，当前保持 FCFS 顺序。
        running: 已拥有 block table、可在后续 schedule 中推进的请求列表。
        prev_step_scheduled_req_ids: 上一轮实际进入模型 batch 的请求 IDs，用于决定
            CachedRequestData 是否需要补发完整 token 序列。
        finished_req_ids: 本轮之前已结束、下一次 SchedulerOutput 应通知 Worker
            清理本地状态的请求 IDs。
    """

    def __init__(
        self, 
        vllm_config: NanovllmConfig,
        kv_cache_config:KVCacheConfig) -> None:
        """根据已规划的 KV cache 配置创建 Scheduler。

        Args:
            vllm_config: 全局推理配置，提供请求数和单轮 token budget 等调度限制。
            kv_cache_config: 当前 Worker 已确定的 KV cache 分配方案。其中
                ``num_blocks`` 是包含保留空块的物理总数，请求可用容量为其减一。
        """
        # max_num_running_reqs 是请求数量上限，与单轮 token budget 是两个独立约束。
        self.max_num_running_reqs: int = vllm_config.scheduler_config.max_num_seqs
        # 该预算限制单个 forward batch 的总 token 数，使长 prompt 不会无限独占计算。
        self.max_num_scheduled_tokens: int = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        # eos 暂由未来的 Engine 输出处理逻辑用于判断生成是否结束。
        self.eos: int = vllm_config.eos
        # 当前 KVCacheManager 只维护一张 block table，因此只接受单 group。
        if len(kv_cache_config.kv_cache_groups) != 1:
            raise NotImplementedError(
                "nano-vLLM Scheduler currently supports exactly one KV cache group"
            )
        # group 决定一张 block table 如何解释 token 区间。
        kv_cache_group = kv_cache_config.kv_cache_groups[0]
        block_size = kv_cache_group.kv_cache_spec.block_size
        # num_blocks 是规划器根据 profiling 得到的整数容量，必须和 Worker 实际
        # 分配的每层 KV tensor page 数一致。
        self.kv_cache_manager = KVCacheManager(
            num_blocks=kv_cache_config.num_blocks,
            block_size=block_size,
        )

        # requests 是活跃逻辑请求的权威表，调度输出更新进度时通过它回查 Request。
        self.requests: dict[str, Request] = {}
        # waiting 保存未运行或已抢占请求；deque 支持抢占请求 appendleft 优先恢复。
        self.waiting: deque[Request] = deque()
        # running 保存有物理 KV 映射的请求，顺序也定义当前 MVP 的 FCFS 服务顺序。
        self.running: list[Request] = []

        # 用于构造 CachedRequestData.all_token_ids。
        self.prev_step_scheduled_req_ids: set[str] = set()
        # 下一次 schedule 时通知 Worker 清理已经结束的 request。
        self.finished_req_ids: set[str] = set()

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
        # append 保留请求提交顺序，实现当前 MVP 的 FCFS 等待语义。
        self.waiting.append(request)
    def finish_requests(
        self,
        request_ids:list[str],
        finished_status:RequestStatus,
    )->None:
        """结束一组活跃请求，并释放其 Scheduler 侧资源。

        该方法是 Request 生命周期的统一出口。无论请求是正常生成 EOS、
        达到 max_tokens，还是被上层主动 abort，Engine 都应调用本方法，而不应
        直接修改 ``running``、``waiting`` 或 ``KVCacheManager``。

        Args:
            request_ids: 要结束的请求 ID 列表。不存在或已结束的 ID 会被忽略。
            finished_status: 目标完成状态，必须是任意 ``FINISHED_*`` 状态。

        Raises:
            AssertionError: ``finished_status`` 不是完成状态时抛出。

        Side Effects:
            - 从 ``running`` 或 ``waiting`` 中移除对应请求。
            - 释放 request 持有的物理 KV block table。
            - 更新 Request 状态，并从 ``requests`` 权威表删除。
            - 将 request ID 加入 ``finished_req_ids``，供下一次 SchedulerOutput
              通知 Worker 删除其本地 ``WorkerRequestState``。
        """
        # finish 接口只能接收 FINISHED_*，避免误把请求结束为 RUNNING 等中间状态。
        assert RequestStatus.is_finished(finished_status)
        # set 去重，防止调用方传入重复 ID 后重复释放同一个 block table。
        request_ids_to_finish=set(request_ids)
        if not request_ids:
            return
        # 先从权威表取出仍然有效的请求；不存在或已完成的请求无需重复处理。
        requests_to_finish:list[Request]=[]
        for request_id in request_ids_to_finish:
            request = self.requests.get(request_id)
            if request is None or request.is_finished():
                continue
            requests_to_finish.append(request)
        if not requests_to_finish:
            return
        # 只移除真正有效请求，不能用原始输入集合误删未知/已结束状态。
        valid_request_ids={
            request.request_id
            for request in requests_to_finish
        }
        # running 是 list，按 request ID 过滤后整体替换，避免遍历中原地 remove。
        self.running = [
            request
            for request in self.running
            if request.request_id not in valid_request_ids
        ]
        # waiting 是 deque；WAITING 与 PREEMPTED 请求都可能在这里。
        self.waiting = deque(
            request
            for request in self.waiting
            if request.request_id not in valid_request_ids
        )
        # 队列中移除后，再逐个释放物理资源并删除 Scheduler 权威状态。
        for request in requests_to_finish:
            # RUNNING 请求有 block table；PREEMPTED/WAITING 通常已无 table。
            # KVCacheManager.free() 对不存在的 table 按空表处理，因此可统一调用。
            self.kv_cache_manager.free(request)
            # EOS / 长度停止原因由 Engine 预先写入；abort 补一个可读原因。
            if (
                finished_status == RequestStatus.FINISHED_ABORTED
                and request.stop_reason is None
            ):
                request.stop_reason = "abort"
            # 先更新逻辑状态，再从活跃请求权威表删除。
            request.status=finished_status
            del self.requests[request.request_id]
            # 下一个 Worker batch 需要据此释放对应的 WorkerRequestState。
            self.finished_req_ids.add(request.request_id)
    def is_finish(self) -> bool:
        """判断 Scheduler 是否已经没有任何待处理或运行中的请求。

        Returns:
            当 ``waiting`` 和 ``running`` 均为空时返回 ``True``。完成请求在上层
            清理后会从这两个容器移除，因此无需额外遍历 ``requests``。
        """
        return not self.waiting and not self.running

    def has_requests(self) -> bool:
        """判断 Engine 是否还需要执行调度或通知 Worker 清理请求。

        沿用 vLLM 的语义：除了 waiting/running 中尚未完成的请求，上一轮产生的
        finished_req_ids 也需要一次执行端调用。它与 is_finish() 的区别在于：
        所有用户请求已经完成时，仍可能有尚未送达 Worker 的清理通知。

        Returns:
            有活跃请求，或有等待下发的结束/取消通知时返回 True。
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

        # free 删除 request block table 并减少每个物理 block 的引用计数。
        self.kv_cache_manager.free(request)
        # PREEMPTED 区别于首次 WAITING，供 Worker 选择恢复而非首次建档的协议。
        request.status = RequestStatus.PREEMPTED
        # 所有物理 KV 均释放后，旧 num_computed_tokens 不再对应有效缓存。
        request.num_computed_tokens = 0
        # 抢占计数可用于后续可观测性和防止长期饥饿的策略。
        request.num_preemptions += 1
        # appendleft 让刚被抢占的请求在当前 FCFS MVP 中优先等待恢复。
        self.waiting.appendleft(request)

    def schedule(self) -> SchedulerOutput:
        """调度一次可混合 prefill 与 decode 的连续 batch。

        本方法不执行模型 forward。它只决定每个请求本轮要推进多少 token、为其
        准备物理 KV blocks，并将执行前的 Worker 数据封装为 SchedulerOutput。输出
        固化后，Engine 先执行模型和采样，成功后再调用 commit_scheduled_tokens()
        推进本轮计算进度。首次接纳时可以先记录 prefix cache 命中的已有进度，
        但本轮新调度的 token 不在本方法内提前提交。

        Returns:
            描述本轮新请求、已有请求、token 数、完成通知和抢占事件的
            SchedulerOutput。若无活跃请求，返回零 token 输出，并保留尚待发送的
            finished_req_ids，供 Worker 清理已经结束或取消的请求。

        Side Effects:
            可能分配或释放 KV blocks、将请求在 waiting/running 间移动、修改
            prefix cache 命中的已有进度。新计算进度与跨轮记录在执行成功后提交。
        """
        # 没有活跃请求也可能还有 Worker 清理工作；零 token 输出必须保留完成通知。
        if self.is_finish():
            scheduler_output = SchedulerOutput.make_empty()
            # 保留 set 快照，成功送达后由 commit_scheduled_tokens() 替换 Scheduler 的集合。
            scheduler_output.finished_req_ids = self.finished_req_ids
            return scheduler_output

        # token_budget 是本轮剩余可执行 token 数，所有请求共享这一全局预算。
        token_budget = self.max_num_scheduled_tokens
        # scheduled_running_reqs 收集本轮推进的既有 RUNNING 请求。
        scheduled_running_reqs: list[Request] = []
        # scheduled_new_reqs 收集首次从 WAITING 接纳、需在 Worker 首次建档的请求。
        scheduled_new_reqs: list[Request] = []
        # scheduled_resumed_reqs 收集由 PREEMPTED 重新获得资源的请求。
        scheduled_resumed_reqs: list[Request] = []
        # req_to_new_blocks request_id -> 本轮发给 Worker 的 block table 更新数据。（新/恢复请求放完整table，普通running放新增的block）
        req_to_new_blocks: dict[str, list[int]] = {}
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
            # num_new_tokens 是本轮尚需写入 KV 的 token 数：目标长度减去已计算长度。
            num_new_tokens = request.num_tokens - request.num_computed_tokens
            # token budget 限制本请求本轮规模，支持长 prompt 的 chunked prefill。
            num_new_tokens = min(num_new_tokens, token_budget)

            # 没有新 token 时可能是 Engine 尚未写入新的采样结果，跳过并检查下一个。
            if num_new_tokens == 0:
                req_index += 1
                continue

            # 申请资源失败时，从运行队尾释放请求，直到当前请求可分配或自己被抢占。
            while True:
                # new_blocks 是本轮新获得的物理块；None 表示空闲池容量不足。
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request=request,
                    num_computed_tokens=request.num_computed_tokens,
                    num_new_tokens=num_new_tokens,
                )

                # 分配成功后当前请求可安全进入本轮 batch。
                if new_blocks is not None:
                    break

                # MVP 版本采用FCFS，从队尾抢占
                # TODO: 后面可以加策略的选择
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

        # 再接纳 waiting/preempted 请求，让剩余 token budget 形成混合连续 batch。
        while self.waiting and token_budget > 0:
            # 并发请求数已满时，不能再把等待请求移动至 RUNNING。
            if len(self.running) == self.max_num_running_reqs:
                break

            # request 只 peek 队首；FCFS 下它不可进入时不能跳过它调度后继请求。
            request = self.waiting[0]
            # request_id 后续用于写入本轮 token 决策映射。
            request_id = request.request_id

            # waiting 队列只应保存首次等待或被抢占的请求，其他状态表示状态机错误。
            assert request.status in (
                RequestStatus.WAITING,
                RequestStatus.PREEMPTED,
            )

            # MVP 不支持远端 KV；waiting/preempted 都从本地 prefix cache 查询。
            # new_computed_blocks 是可共享的完整 prefix 物理块，尚未增加引用计数。
            new_computed_blocks, num_new_local_computed_tokens = (
                self.kv_cache_manager.get_computed_blocks(request)
            )
            # 以后如果又远程KV，那可能会加上其他的token
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

            # new_block_ids 仅记录本轮新物理块；调用本身会原子地附加 prefix blocks。
            new_block_ids = self.kv_cache_manager.allocate_slots(
                request=request,
                num_computed_tokens=num_new_local_computed_tokens,
                num_new_tokens=num_new_tokens,
                cached_block_ids=new_computed_blocks,
            )
            # 队首资源不足时保持 FCFS，不跳过它让后继请求插队。
            if new_block_ids is None:
                break

            # 容量检查和分配均成功后，才从 waiting 队列正式移除这个请求。
            request = self.waiting.popleft()
            # 重新读取 request_id，使后续代码始终与出队对象绑定。
            request_id = request.request_id

            # New 与 PREEMPTED 的 worker 数据传输语义不同。
            if request.status == RequestStatus.WAITING:
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
                self.kv_cache_manager.get_block_table(request_id).copy()
            )
            # 确认调度
            num_scheduled_tokens[request_id] = num_new_tokens
            # 扣除本请求已承诺的模型执行预算。
            token_budget -= num_new_tokens

        # 构建调度输出

        # new_reqs_data 是首次发送给 Worker 的完整请求快照列表。
        new_reqs_data = [
            NewRequestData.from_request(
                request=req,
                block_ids=req_to_new_blocks[req.request_id].copy(),
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
            block_ids = req_to_new_blocks[request.request_id]

            # 空列表表示普通 running request 本轮没有跨 block 边界；
            # Worker 无需更新其已有 block table，因此协议中使用 None。
            cached_new_block_ids.append(
                block_ids.copy() if block_ids else None
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
            # 这些值必须是本轮 forward 之前的进度，故在 commit_scheduled_tokens() 前读取。
            num_computed_tokens=[
                request.num_computed_tokens
                for request in cached_requests
            ],
            # 输出长度让 Worker 知道 request.all_token_ids 中 prompt/output 的分界位置。
            num_output_tokens=[
                request.num_output_tokens
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
            # 将本轮抢占对象投影为稳定 ID 集合，供 Worker 丢弃本地请求状态。
            preempted_req_ids={req.request_id for req in preempted_reqs},
        )

        return scheduler_output

    def commit_scheduled_tokens(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """在模型执行与采样成功后，提交 Request 进度并轮换跨轮状态。

        调度输出中的进度必须保持在本轮 forward *之前*。当前同步 Engine 在取得
        模型与采样结果后调用本方法，再处理新生成 token，保证这些 token 不会被
        提前计入已写入 KV 的进度。零 token 输出也需要调用，以确认清理通知送达。

        Args:
            scheduler_output: 执行端已经成功消费的单步调度决定。

        Side Effects:
            增加每个被调度 Request 的 ``num_computed_tokens``，更新其 chunked prefill
            提示位，记录本轮已调度请求集合，并为下一轮替换完成通知集合。
        """
        # 逐请求 token 数是推进逻辑进度的唯一来源，避免从批次总数反推。
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        # 遍历本轮每个实际进入 batch 的 request 及其承诺执行的 token 数。
        for request_id, num_tokens in num_scheduled_tokens.items():
            # request 从权威表取回，确保修改的是 Scheduler 持有的同一对象。
            request = self.requests[request_id]
            # 本请求已计算的token数 加上 本次该请求调度的token数
            request.num_computed_tokens += num_tokens
            # 如果是一次性的prefill或者decode阶段，这两个数应该相等
            # 只有chunk prefill会prefill部分，导致数量偏小
            request.is_prefill_chunk = (
                request.num_computed_tokens < request.num_tokens
            )

        # 下一轮构造 all_token_ids 时，以本轮实际执行过的 request 集合为依据。
        self.prev_step_scheduled_req_ids = set(num_scheduled_tokens)
        # 不能 clear 原 set；SchedulerOutput 仍持有它。
        # 用新集合开始收集下一轮完成通知，保持已返回输出的快照稳定。
        self.finished_req_ids = set()
