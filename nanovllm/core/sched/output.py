"""调度器向 Engine/Worker 暴露的单步调度结果。

沿用 vLLM 的 ``NewRequestData``、``CachedRequestData`` 和 ``SchedulerOutput``
划分：首次建档发送完整请求数据，已有请求发送增量更新，Worker 据此维护请求镜像
和 InputBatch。Engine 在异步队列中保留原始调度计划，供对应结果回收时匹配请求
与计算量；流水线并行的增量 token 字段目前保留为空。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from nanovllm.request import Request
from nanovllm.sampling_params import SamplingParams


@dataclass(slots=True)
class NewRequestData:
    """首次发送给 Worker 的请求建档数据与执行前进度。

    Attributes:
        req_id: 新请求的全局唯一标识。
        prompt_token_ids: 完整原始 prompt token 序列，按只读引用共享；Worker 据此
            创建本地请求状态，不只包含本轮计算的 chunk。
        sampling_params: Worker/采样器在后续生成中需要使用的采样配置。
        block_ids: 当前唯一 KV cache group 已分配的完整 block table。第 i 项是逻辑
            block i 对应的物理 block ID；当前单组 MVP 直接使用 ``list[int]``，
            不使用 vLLM 多 KV group 所需的外层 tuple。
        num_computed_tokens: 本轮执行前的逻辑进度，不含本轮调度量。首次接纳时来自
            prefix cache 命中的 token 数，因此可能大于零。
    """

    req_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams

    block_ids: list[int]
    num_computed_tokens: int

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: list[int],
    ) -> "NewRequestData":
        """从 Request 和 KVCacheManager 的快照构造首次调度数据。

        Args:
            request: 刚从 waiting 队列接纳、需要在 Worker 侧首次建档的请求。
            block_ids: 调用方准备好的完整 block table 列表，本方法直接持有其引用。

        Returns:
            new_request_data: 保存执行前进度的 NewRequestData。prompt、采样参数和传入页表均按引用
                传递；prompt 和采样参数在请求创建后按只读约定使用。
        """
        return cls(
            # req_id 让 Worker 能将后续 CachedRequestData 与这个新请求关联起来。
            req_id=request.request_id,
            # 共享创建后保持只读的完整 prompt，Worker 另行维护可变输入缓冲区。
            prompt_token_ids=request.prompt_token_ids,
            # SamplingParams 本身是请求创建后不变的共享配置对象。
            sampling_params=request.sampling_params,
            # Scheduler 已将物理块映射导出为编号列表，本对象直接持有传入的页表。
            block_ids=block_ids,
            # 该值必须是本轮 forward 前的进度，不能包含本次已调度 token。
            num_computed_tokens=request.num_computed_tokens,
        )


@dataclass
class CachedRequestData:
    """发送给已经在 Worker 侧建档请求的增量调度数据。

    ``new_block_ids``、``num_computed_tokens`` 和 ``num_output_tokens`` 均与
    ``req_ids`` 按下标对齐。恢复集合和完整历史按请求 ID 查询；当前未接入 PP，
    ``new_token_ids`` 固定为空，不参与这些列的长度对齐。

    Attributes:
        req_ids: 本轮被调度的已有请求 IDs，决定所有并行列表的顺序。
        resumed_req_ids: 本轮从 PREEMPTED 恢复、需要 Worker 重建部分状态的请求 IDs。
        new_token_ids: pipeline parallelism 使用的增量 token 数据；当前无 PP，保留
            空列表以对齐 vLLM 接口。
        all_token_ids: 本轮重新调度、但上一轮未调度的已有请求的完整真实历史副本，
            key 为请求 ID。值包含 prompt 和已确认输出，不含占位；异步重新入批时
            Worker 用它恢复真实输出历史。
        new_block_ids: 普通 RUNNING 请求只发送新增块，None 表示没有新增块；
            从 PREEMPTED 恢复的请求发送完整页表，Worker 用它覆盖失效的旧映射。
        num_computed_tokens: 每个请求在本轮执行前的逻辑进度，包含缓存命中或先前
            已承诺的计算，不含本轮调度量；异步时先前计算可能尚未完成。
        num_output_tokens: 执行前的逻辑输出长度，即已确认输出数加输出占位数。
            同步时占位数为零，异步时不能将该值视为 CPU 已取得的真实输出数。
    """

    req_ids: list[str]
    resumed_req_ids: set[str]

    # 当前没有 PP，保留空列表以对齐 vLLM 接口。
    new_token_ids: list[list[int]]
    # 为上一轮未调度的已有请求补发真实历史，支持异步重新建槽。
    all_token_ids: dict[str, list[int]]
    # 与 req_ids 按下标对齐：
    # - 普通 RUNNING request：仅发送本轮新增的 block IDs；None 表示没有新增块。
    # - PREEMPTED 后恢复的 request：发送完整 block table，Worker 用它覆盖旧映射。
    new_block_ids: list[list[int] | None]
    # 保存执行前位置；后续 Request 进度推进不会改写这一批的数值。
    num_computed_tokens: list[int]
    # 与 req_ids 对齐的逻辑输出长度，异步时包含尚未回收的输出占位。
    num_output_tokens: list[int]

    @property
    def num_reqs(self) -> int:
        """返回本批已有请求的数量。

        Returns:
            num_reqs: ``req_ids`` 的长度，也是所有按请求对齐的 list 应具有的长度。
        """
        return len(self.req_ids)

    @cached_property
    def _req_id_to_num_output_tokens(self) -> dict[str, int]:
        """延迟建立 request ID 到输出长度的查询表。

        该属性首次访问后缓存结果，因此 CachedRequestData 创建完成后不应再原地
        修改 ``req_ids`` 或 ``num_output_tokens``。

        Returns:
            req_id_to_num_output_tokens: 请求 ID 到执行前逻辑输出长度的字典，异步时长度包含输出占位。
        """
        # zip 按批次列式布局将 request ID 与对应输出长度配对。
        return dict(zip(self.req_ids, self.num_output_tokens))

    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        """构造一个不包含任何已有请求的合法空批次数据。

        Returns:
            cached_request_data: 所有集合、字典和列表都独立创建的空 CachedRequestData，可安全作为
                ``SchedulerOutput.make_empty`` 的组成部分。
        """
        return cls(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        )


@dataclass
class SchedulerOutput:
    """一次 ``Scheduler.schedule()`` 产生的执行前调度计划。

    ``num_scheduled_tokens`` 定义执行请求集合及每个请求本轮的输入 token 计算量。
    新请求与已有请求按 Worker 是否建档划分，两类都可能需要 prefill 或重算；
    Worker 根据执行前进度和调度量构建混合 batch。执行前计数按快照使用，类本身
    不强制冻结，也不对所有引用对象做深拷贝。

    Attributes:
        scheduled_new_reqs: 本轮首次进入 Worker 的请求完整数据。
        scheduled_cached_reqs: 本轮已有请求的增量数据，包含持续运行与抢占恢复请求。
        num_scheduled_tokens: 从 request ID 到本轮要执行 token 数的映射。
        total_num_scheduled_tokens: ``num_scheduled_tokens`` 所有值的总和，用于批次
            级统计和模型执行规划，不包含 CUDA Graph padding。
        finished_req_ids: 下发本计划前已结束的请求 IDs，通知 Worker 清理镜像和槽位。
        preempted_req_ids: 本轮因 KV 不足被抢占的请求 IDs，记录抢占事件。当前 Runner
            通过调度集合移出未执行请求，通过 resumed_req_ids 重建恢复请求，
            不直接读取这个事件字段。
    """

    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: CachedRequestData
    # req_id -> 当前 step 要推进的 token 数。
    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int
    # 已结束请求的 Worker 清理通知，零 token 计划也需要传递。
    finished_req_ids: set[str]
    # 记录本轮抢占事件；请求镜像保留，恢复时重新下发完整物理映射。
    preempted_req_ids: set[str]

    @classmethod
    def make_empty(cls) -> "SchedulerOutput":
        """构造一个没有调度、完成或抢占事件的合法空输出。

        Returns:
            scheduler_output: 所有集合与调度量均为空的计划，各可变容器独立创建。Scheduler 需要
                仅发送清理通知时，会再填入 finished_req_ids，Worker 仍需处理该计划。
        """
        return cls(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            finished_req_ids=set(),
            preempted_req_ids=set(),
        )
