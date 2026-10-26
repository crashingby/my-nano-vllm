"""nano-vLLM 的最小逻辑请求状态机。

当前一个 :class:`Request` 对应一个用户生成请求和一条生成分支。将来实现
beam search、best-of 或多分支采样时，可以在 Request 之下再引入真正的
Sequence；在当前 MVP 中，调度器只需要以 Request 为最小单位管理生命周期、
token 进度和 KV cache 所属关系。
"""

from __future__ import annotations

from enum import IntEnum, auto
from time import time
from typing import TYPE_CHECKING, Any,Callable
from nanovllm.sampling_params import SamplingParams
if TYPE_CHECKING:
    from nanovllm.core.kv_cache_utils import BlockHash
from nanovllm.engine import(
    FinishReason,
    PrefillStats
)

class RequestStatus(IntEnum):
    """描述一个 :class:`Request` 在调度器中的生命周期状态。

    Attributes:
        WAITING: 请求已提交，正在等待 Scheduler 分配 token budget 和 KV blocks。
        RUNNING: 请求已拥有 KV blocks，可以在后续 schedule 中继续推进。
        PREEMPTED: 请求曾因 KV cache 不足被抢占，逻辑 token 仍保留，但物理
            KV blocks 已释放，恢复时需要重新分配或命中 prefix cache。
        FINISHED_STOPPED: 模型生成了 EOS 或命中了用户指定的停止条件。
        FINISHED_LENGTH_CAPPED: 请求达到 ``SamplingParams.max_tokens`` 限制。
        FINISHED_ABORTED: 上层主动取消了请求。

    保持 vLLM 的关键不变量：所有 ``FINISHED_*`` 状态都排在 ``PREEMPTED``
    后面，因此 :meth:`is_finished` 可以统一按枚举值判断终止状态。
    """

    WAITING = auto()
    RUNNING = auto()
    PREEMPTED = auto()

    FINISHED_STOPPED = auto()
    FINISHED_LENGTH_CAPPED = auto()
    FINISHED_ABORTED = auto()

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        """判断一个请求状态是否表示生命周期已经结束。

        Args:
            status: 要判断的请求状态。

        Returns:
            is_finished: 当 ``status`` 是任意 ``FINISHED_*`` 状态时返回 ``True``；等待、
                运行和抢占状态均返回 ``False``。
        """
        return status > RequestStatus.PREEMPTED
    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> FinishReason | None:
        """将终止状态映射为输出协议的完成原因，未结束状态返回 None。

        Returns:
            finish_reason: 终止状态对应的 FinishReason；状态尚未结束时为 None。
        """
        return _FINISHED_REASON_MAP.get(status)


class Request:
    """表示一个逻辑文本生成请求及其调度进度。

    Request 只保存逻辑层信息，不直接拥有物理 KV cache block table。物理块的
    所有权和引用计数由 ``KVCacheManager`` 维护，以便多个请求安全地复用同一段
    prefix cache。

    Attributes:
        request_id: 调用方提供的全局唯一请求标识，也是 Scheduler 和
            KVCacheManager 中各映射的 key。
        priority: 优先级队列使用的排序键；默认 FCFS 策略按入队顺序处理。
        arrival_time: 请求进入系统的时间戳，用于同优先级请求的稳定排序。
        status: 当前请求生命周期状态。
        stop_reason: 请求完成或取消时由上层记录的停止原因；运行期间为 ``None``。
        sampling_params: 本请求专属的采样参数。
        max_tokens: 从 ``sampling_params`` 缓存的最大输出 token 数。
        ignore_eos: 从 ``sampling_params`` 缓存的 EOS 忽略开关。
        prompt_token_ids: 不可变语义的输入 prompt token 序列。
        _output_token_ids: 已经由模型确认生成的输出 token 序列。
        _all_token_ids: ``prompt_token_ids + _output_token_ids``，是 KV cache
            逻辑位置和模型输入使用的完整 token 序列。
        num_prompt_tokens: prompt 的 token 数，在请求创建后保持不变。
        num_computed_tokens: 已命中缓存或已承诺执行的逻辑进度。GPU 是否完成由
            执行结果确认，不能单独依据这个数字判断。
        num_output_placeholders: 已安排但尚未回收的有效输出位置数，不含 token ID。
        num_in_flight_tokens: 已安排但尚未回收的输入计算量，包含 prefill 和 decode。
        num_stale_output_tokens: 抢占时仍在途的旧输入计算量，回收后只更新真实历史；
            名称沿用输出协议，实际按各旧批次处理的输入 token 数扣减。
        last_sched_seq: 可选延迟释放策略记录的最近写入批次编号；该策略默认关闭。
        is_prefill_chunk: 本轮调度后是否仍有 prompt 或重算 token 未完成；它只
            是执行提示，不是 P/D 逻辑状态机。
        num_preemptions: 请求被抢占的累计次数，可供后续统计或策略使用。
        block_hashes: 已经形成完整 token block 的逻辑 hash 链。第 i 项对应第 i 个
            完整逻辑 block；它描述 token 内容，不描述物理 KV 内存位置。
        _block_hasher: Engine 创建、所有 Request 共享的增量 hash 函数；关闭前缀
            缓存时为 None，此时不维护 block_hashes。
    """

    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        priority: int = 0,
        client_index: int = 0,
        arrival_time: float | None = None,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None = None,
    ) -> None:
        """创建一个处于 ``WAITING`` 状态的文本生成请求。

        Args:
            request_id: 非空且在同一 Scheduler 内唯一的请求标识。
            prompt_token_ids: 已经 tokenizer 编码后的非空 prompt token 序列。
            sampling_params: 本请求的采样配置，例如最大生成长度和 EOS 策略。
            priority: 优先级调度的排序值；数值越小，排序越靠前。
            client_index: 结果按客户端分组时使用的路由编号，缺省为 0。
            arrival_time: 可选的入队时间戳。未提供时使用当前系统时间，测试中可
                显式传入以获得可复现的排序结果。
            block_hasher: 由 Engine 创建并在所有 Request 间共享的增量 hash 函数。
                函数接收 token 序列已经更新的当前 Request，只返回此前尚未记录的
                完整逻辑 block hashes；Request 会将结果追加到 ``block_hashes``。
                它不分配物理 KV block，也不访问 ``KVCacheManager``；这些物理缓存
                职责仍由调度与缓存管理层处理。未启用 prefix cache 时可传入
                ``None``。

        Raises:
            ValueError: ``request_id`` 为空，或 ``prompt_token_ids`` 为空时抛出。
        """
        if not request_id:
            raise ValueError("request_id must not be empty")
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")

        # request_id 是逻辑请求跨 Scheduler、KVCacheManager 和 Worker 的共同身份。
        self.request_id: str = request_id
        self.client_index: int = client_index  # Scheduler 按此编号聚合同一客户端的输出。
        # 默认 FCFS 不比较该字段，priority 策略通过 __lt__ 读取它。
        self.priority: int = priority
        # 未传入时间时在创建点采样，避免排序时多次读取系统时钟。
        self.arrival_time: float = (
            time() if arrival_time is None else arrival_time
        )
        # 新创建的请求尚未得到物理资源，因此总是从 WAITING 状态开始。
        self.status: RequestStatus = RequestStatus.WAITING
        # 在请求结束前没有停止原因；上层完成处理时再填写该字段。
        self.stop_reason: int | str | None = None

        # 保存完整采样配置，后续采样器需要读取 temperature 等参数。
        self.sampling_params: SamplingParams = sampling_params
        # 缓存热点字段，调度器无需每次通过 sampling_params 间接访问。
        self.max_tokens: int = sampling_params.max_tokens
        self.ignore_eos: bool = sampling_params.ignore_eos

        # prompt 永不修改；生成 token 只追加到私有 output/all 列表。
        self.prompt_token_ids: list[int] = prompt_token_ids
        # 只保存生成结果，不包含 prompt，供上层返回 completion 使用。
        self._output_token_ids: list[int] = []
        # KV cache 的逻辑位置基于完整序列，因此单独维护 prompt + output 的拼接表。
        self._all_token_ids: list[int] = prompt_token_ids.copy()

        # 尚未回收的有效采样输出数量；它不扩展真实 token 列表。
        self.num_output_placeholders = 0
        # 抢占时复制在途输入量，旧批次按原始输入数回收，不能扣减新阶段的占位。
        self.num_stale_output_tokens = 0
        # 已提交且尚未由 Scheduler 回收的输入数；GPU 完成不会自行减少该计数。
        self.num_in_flight_tokens = 0
        
        # 仅在 Scheduler 启用延迟释放时用于标记最后一批 KV 写入。
        self.last_sched_seq = 0
        # prompt 长度在构造后不变，常用于判断 prefill 与生成长度上限。
        self.num_prompt_tokens: int = len(self.prompt_token_ids)
        # schedule() 返回前推进的调度进度，可能包含尚未完成的 GPU 写入。
        self.num_computed_tokens: int = 0

        # 仅用于标记“本轮之后仍有 prompt/recompute token 没算完”。
        # 它不是 P/D 状态机，也不用于决定 Request 的逻辑阶段。
        self.is_prefill_chunk: bool = False

        # 记录抢占次数，后续可用于统计和调度策略。
        self.num_preemptions: int = 0

        self.prefill_stats: PrefillStats | None = PrefillStats()  # 首次输出时转交一次后清空。
        # 每项对应一个完整逻辑 token block 的 prefix hash。
        # 它与物理 block table 分离，抢占释放 KV 后仍可保留。
        self.block_hashes:list[BlockHash] =[]
        # Engine 注入的共享函数；prefix cache 未启用时允许为 None。
        self._block_hasher = block_hasher
        # Request 创建后立刻为完整 prompt blocks 建立逻辑 hash。
        self.update_block_hashes()

    @property
    def all_token_ids(self) -> list[int]:
        """返回 prompt 与已生成输出拼接后的完整 token 序列。

        Returns:
            all_token_ids: Request 当前的完整逻辑 token 序列。该列表由 Request 内部维护；调用方
                应将其视为只读，不能原地修改。
        """
        return self._all_token_ids

    @property
    def output_token_ids(self) -> list[int]:
        """返回已经确认生成的输出 token 序列。

        Returns:
            output_token_ids: 不包含 prompt 的输出 token 列表。调用方应将返回列表视为只读。
        """
        return self._output_token_ids

    @property
    def num_tokens(self) -> int:
        """返回当前完整序列的逻辑 token 数。

        Returns:
            num_tokens: ``len(prompt_token_ids) + len(output_token_ids)``。Scheduler 用这个
                值加 ``num_output_placeholders`` 再减 ``num_computed_tokens``，
                用于计算普通运行请求本轮需要推进的输入数。
        """
        return len(self._all_token_ids)

    @property
    def num_output_tokens(self) -> int:
        """返回已经生成的输出 token 数。

        Returns:
            num_output_tokens: 当前 completion 的长度，不包含 prompt token。
        """
        return len(self._output_token_ids)

    def is_finished(self) -> bool:
        """判断当前请求是否已经结束。

        Returns:
            is_finished: 当 ``self.status`` 为任意 ``FINISHED_*`` 状态时返回 ``True``。
        """
        return RequestStatus.is_finished(self.status)
    def get_finished_reason(self) -> FinishReason | None:
        """返回当前状态对应的完成原因，请求尚未结束时返回 None。

        Returns:
            finish_reason: 当前请求的完成原因；请求尚未结束时为 None。
        """
        return RequestStatus.get_finished_reason(self.status)
    def update_block_hashes(self)->None:
        """增量记录 Request 新出现的完整逻辑 token block hashes。

        hash 描述的是 token 序列身份；物理 KV block 的分配、引用计数与 prefix
        cache 索引仍由 KVCacheManager 管理。

        Side Effects:
            将此前未 hash 的完整 token blocks 追加到 ``self.block_hashes``。
        """
        if self._block_hasher is not None:
            # 共享 hasher 返回 new_block_hashes，只含新增完整 blocks，不重复 hash 已有 prompt。
            new_block_hashes = self._block_hasher(self)
            self.block_hashes.extend(new_block_hashes)
    def append_output_token_ids(
        self,
        token_ids: int | list[int],
    ) -> None:
        """追加模型本轮确认生成的一个或多个输出 token。

        此方法必须在模型采样结果确认后调用。它同时更新输出序列和完整序列，但
        不推进 ``num_computed_tokens``：新 token 在下一次被调度时，才由
        Scheduler 计入已承诺执行的进度。

        Args:
            token_ids: 单个采样 token，或同一轮确认的一组 token。投机解码接入后
                可以传入多个已验证 token。

        Side Effects:
            修改 ``_output_token_ids`` 与 ``_all_token_ids``，使下一轮 Scheduler
            能看到需要写入 KV cache 的新增 token。
        """
        # 单 token 路径使用 append，避免把一个 int 当作可迭代对象处理。
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            # 多 token 路径保持两个逻辑序列始终同步增长。
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)
        self.update_block_hashes()
    def take_prefill_stats(self) -> PrefillStats | None:
        """转交首轮 prefill 统计并清空引用，后续输出不再重复携带。

        Returns:
            prefill_stats: 首次调用时返回当前统计对象；已转交时返回 None。

        Side Effects:
            将 self.prefill_stats 设为 None，不复制统计对象。
        """
        if self.prefill_stats is None:
            return None
        prefill_stats = self.prefill_stats
        self.prefill_stats = None
        return prefill_stats
    def __lt__(self, other: "Request") -> bool:
        """为优先级队列定义请求排序。

        Args:
            other: 与当前请求比较的另一个 Request。

        Returns:
            is_less: 当前请求应排在 ``other`` 前面时返回 ``True``。排序优先级依次为
                ``priority``、``arrival_time``、``request_id`` 和对象身份。
        """
        # priority 越小越靠前，PriorityRequestQueue 通过 heapq 使用该比较规则。
        if self.priority != other.priority:
            return self.priority < other.priority
        # 相同优先级下，先到达的请求先被服务，维持 FCFS 语义。
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        # request_id 让同一时刻创建的请求仍具有跨进程可复现的顺序。
        if self.request_id != other.request_id:
            return self.request_id < other.request_id
        # 极端情况下用对象身份打破平局，保证严格弱序关系。
        return id(self) < id(other)


_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
}
