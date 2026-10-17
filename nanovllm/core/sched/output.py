"""调度器向 Engine/Worker 暴露的单步调度结果。

这些对象刻意参考 vLLM 的 ``NewRequestData``、``CachedRequestData`` 和
``SchedulerOutput``。当前 nano-vLLM 还没有 Worker 侧 request cache，因此部分
字段暂时主要用于确立接口边界；未来接入 ModelRunner request cache、异步调度或
pipeline parallelism 时可以沿用这一数据流。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from nanovllm.request import Request
from nanovllm.sampling_params import SamplingParams


@dataclass(slots=True)
class NewRequestData:
    """首次发送给 Worker 的完整逻辑请求快照。

    Attributes:
        req_id: 新请求的全局唯一标识。
        prompt_token_ids: 原始 prompt token 序列；Worker 据此创建本地请求状态。
        sampling_params: Worker/采样器在后续生成中需要使用的采样配置。
        block_ids: 当前唯一 KV cache group 的完整 block table。第 i 项是逻辑
            block i 对应的物理 block ID；当前单组 MVP 直接使用 ``list[int]``，
            不使用 vLLM 多 KV group 所需的外层 tuple。
        num_computed_tokens: 本轮 forward 前已经可用的 token 数。新请求命中 prefix
            cache 时，这个值可能大于零。
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
            block_ids: 由 KVCacheManager 提供的完整 block table 快照

        Returns:
            与 ``request`` 当前逻辑状态一致的 NewRequestData。prompt 会复制一份，
            防止 Worker 后续修改内部输入缓冲区时影响 Request。
        """
        return cls(
            # req_id 让 Worker 能将后续 CachedRequestData 与这个新请求关联起来。
            req_id=request.request_id,
            # 复制 prompt，隔离 Scheduler 的逻辑请求状态和 Worker 的输入缓冲区。
            prompt_token_ids=request.prompt_token_ids,
            # SamplingParams 本身是请求创建后不变的共享配置对象。
            sampling_params=request.sampling_params,
            # block_ids 已由 Scheduler 复制内部 table，本对象直接持有该快照。
            block_ids=block_ids,
            # 该值必须是本轮 forward 前的进度，不能包含本次已调度 token。
            num_computed_tokens=request.num_computed_tokens,
        )


@dataclass
class CachedRequestData:
    """发送给已经在 Worker 侧建档请求的增量调度数据。

    除 ``all_token_ids`` 和 ``new_block_ids`` 以外，大多数 list 均按 ``req_ids``
    的相同下标对齐。消费者必须将它们视为一个批次的列式表示，而不是彼此独立的
    列表。

    Attributes:
        req_ids: 本轮被调度的已有请求 IDs，决定所有并行列表的顺序。
        resumed_req_ids: 本轮从 PREEMPTED 恢复、需要 Worker 重建部分状态的请求 IDs。
        new_token_ids: pipeline parallelism 使用的增量 token 数据；当前无 PP，保留
            空列表以对齐 vLLM 接口。
        all_token_ids: 上一轮未调度的已有请求需重新发送的完整 token 序列，key 为
            request ID。当前 MVP 使用它建立清晰的 Worker 同步语义。
        new_block_ids: 普通 RUNNING request：仅发送本轮新增的 block IDs；None 表示没有新增块。 
            PREEMPTED 后恢复的 request：发送完整 block table，Worker 用它覆盖旧映射。
        num_computed_tokens: 每个请求在本轮 forward 前已计算的 token 数。
        num_output_tokens: 每个请求在本轮 forward 前已经采样确认的输出 token 数。
    """

    req_ids: list[str]
    resumed_req_ids: set[str]

    # 当前没有 PP，保留空列表以对齐 vLLM 接口。
    new_token_ids: list[list[int]]
    # 上一轮running队列没有调度到的 request ，需要发送完整 token 序列。
    all_token_ids: dict[str, list[int]]
    # 与 req_ids 按下标对齐：
    # - 普通 RUNNING request：仅发送本轮新增的 block IDs；None 表示没有新增块。
    # - PREEMPTED 后恢复的 request：发送完整 block table，Worker 用它覆盖旧映射。
    new_block_ids: list[list[int] | None]
    num_computed_tokens: list[int]
    num_output_tokens: list[int]

    @property
    def num_reqs(self) -> int:
        """返回本批已有请求的数量。

        Returns:
            ``req_ids`` 的长度，也是所有按请求对齐的 list 应具有的长度。
        """
        return len(self.req_ids)

    @cached_property
    def _req_id_to_num_output_tokens(self) -> dict[str, int]:
        """延迟建立 request ID 到输出长度的查询表。

        该属性首次访问后缓存结果，因此 CachedRequestData 创建完成后不应再原地
        修改 ``req_ids`` 或 ``num_output_tokens``。

        Returns:
            将每个 ``req_id`` 映射到其本轮执行前输出 token 数的字典。
        """
        # zip 按批次列式布局将 request ID 与对应输出长度配对。
        return dict(zip(self.req_ids, self.num_output_tokens))

    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        """构造一个不包含任何已有请求的合法空批次数据。

        Returns:
            所有集合、字典和列表都独立创建的空 CachedRequestData，可安全作为
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
    """一次 ``Scheduler.schedule()`` 产生的不可变语义调度决定。

    ``num_scheduled_tokens`` 是本对象的核心真相：每个 request 本轮要执行多少
    token 都由它定义。P/D 在 vLLM 中不是逻辑状态；Worker 根据每个请求的进度和
    token 数自行形成混合连续 batch。

    Attributes:
        scheduled_new_reqs: 本轮首次进入 Worker 的请求完整数据。
        scheduled_cached_reqs: 本轮已存在于 Worker 中的请求增量数据。
        num_scheduled_tokens: 从 request ID 到本轮要执行 token 数的映射。
        total_num_scheduled_tokens: ``num_scheduled_tokens`` 所有值的总和，用于批次
            级统计和模型执行规划。
        finished_req_ids: 在本轮前已经结束、Worker 应清理本地状态的请求 IDs。
        preempted_req_ids: 本轮因 KV cache 资源不足而被抢占的请求 IDs。
    """

    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: CachedRequestData
    # req_id -> 当前 step 要推进的 token 数。
    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int
    # 已完成、需要通知未来 Worker cache 清理的请求。
    finished_req_ids: set[str]
    # 本轮因 KV cache 不足而被抢占的请求。
    preempted_req_ids: set[str]

    @classmethod
    def make_empty(cls) -> "SchedulerOutput":
        """构造一个没有调度、完成或抢占事件的合法空输出。

        Returns:
            可由 Engine 在 Scheduler 没有活跃请求时直接消费的 SchedulerOutput；
            其中各可变容器均为新对象，不会与其他调用共享。
        """
        return cls(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            finished_req_ids=set(),
            preempted_req_ids=set(),
        )
