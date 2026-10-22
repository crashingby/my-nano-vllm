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
        """提交计算进度后，为能够产生输出的请求增加占位。"""
        super()._update_after_schedule(scheduler_output)

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
        """追加真实结果，并回收当前执行阶段对应的输出占位。"""
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
            assert request.num_output_placeholders >= 0

        if status_before_update == RequestStatus.RUNNING:
            # 仅登记当前有效映射中、真实 token 已经确定的计算前缀。
            self.kv_cache_manager.cache_blocks(
                request,
                request.num_computed_tokens
                - request.num_output_placeholders,
            )

        return new_token_ids, stopped