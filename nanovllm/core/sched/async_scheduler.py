"""通过输出占位支持提前调度，并区分抢占前后的结果回收。"""

from nanovllm.core.sched.output import SchedulerOutput
from nanovllm.core.sched.scheduler import Scheduler
from nanovllm.request import Request, RequestStatus


class AsyncScheduler(Scheduler):
    """为异步执行维护生成占位，旧执行阶段的结果仅更新真实历史。"""

    def _update_after_schedule(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """先承诺本轮输入计算，再为能够产生输出的请求预留位置。

        Args:
            scheduler_output: 已保存执行前进度的本轮计划，不在此改写其字段。

        Side Effects:
            基类推进 Request 的 computed/in-flight 输入计数；本方法额外增加
            有效输出占位，使下一轮不必等真实 token 回到 CPU 即可计算输入量。
        """
        super()._update_after_schedule(scheduler_output)

        # 只遍历实际进入本批的请求，缺席请求不会凭空产生一个新输出。
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                # 中间 prefill chunk 不产生有效生成 token。
                continue

            request.num_output_placeholders += (
                self.num_sampled_tokens_per_step
            )

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[int],
        is_stale: bool = False,
    ) -> tuple[list[int], bool]:
        """追加真实结果，并回收当前执行阶段对应的输出占位。

        Args:
            request: 仍由 Scheduler 管理的权威请求。
            new_token_ids: 本批真实采样输出，基类停止判断可能原地截短该列表。
            is_stale: 是否属于抢占前的旧批次；旧占位已经清零，不能再次扣减。

        Returns:
            (new_token_ids, stopped):
                new_token_ids: 本次实际接受的输出 token 列表。
                stopped: 本次输出更新后请求触发停止条件时为 True。

        Side Effects:
            扩展真实 token 历史并可能改变请求终止状态。正常结果减少输出占位；
            回收前处于 RUNNING 时，按扣除占位的前缀长度尝试登记可缓存块。
        """
        # 基类可能把状态改为 FINISHED，缓存处理仍需知道回收前的状态。
        status_before_update = request.status

        # 先保存实际返回数量，停止检查可能截断传入列表。
        num_new_tokens = len(new_token_ids)
        new_token_ids, stopped = super()._update_request_with_output(
            request,
            new_token_ids,
            is_stale,
        )

        if not is_stale:
            # 旧执行阶段的占位已在抢占时清零，只有当前阶段结果需要扣减。
            request.num_output_placeholders -= num_new_tokens
            # 一份有效输出只能消费原先预留的位置，负值意味着批次回收不匹配。
            assert request.num_output_placeholders >= 0

        if status_before_update == RequestStatus.RUNNING:
            # 仅登记当前有效映射中、真实 token 已经确定的计算前缀。
            self.kv_cache_manager.cache_blocks(
                request,
                request.num_computed_tokens
                - request.num_output_placeholders,
            )

        return new_token_ids, stopped
