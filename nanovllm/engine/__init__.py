"""定义 Scheduler 向 Engine 客户端交付的增量 token、完成原因和统计结构。"""


import msgspec
from nanovllm.metrics.stats import PrefillStats,SchedulerStats
import enum
import time
FINISH_REASON_STRINGS = ("stop", "length", "abort", "error", "repetition")

class FinishReason(enum.IntEnum):
    """输出协议中的请求完成原因，使用整数减少序列化体积。

    STOP 表示 EOS 或停止 token，LENGTH 表示输出或上下文长度上限，ABORT 表示
    主动取消。ERROR 和 REPETITION 为协议保留值，当前请求状态机不会产生它们。
    字符串形式通过 FINISH_REASON_STRINGS 映射，不用于改变请求状态。
    """

    STOP = 0
    LENGTH = 1
    ABORT = 2
    ERROR = 3
    REPETITION = 4

    def __str__(self):
        """返回序列化整数对应的可读完成原因字符串。

        Returns:
            finish_reason: 当前枚举值对应的可读完成原因字符串，例如 stop、length 或 abort。
        """
        return FINISH_REASON_STRINGS[self.value]

class EngineCoreOutput(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):
    """一次结果回收中单个请求的增量输出，不是完整生成历史。

    Attributes:
        request_id: 用于回查客户端请求的逻辑 ID。
        new_token_ids: 本次新接受的真实输出，终止通知时可以为空。
        finish_reason: 非 None 表示请求已结束；None 表示后续仍可能继续输出。
        stop_reason: 命中的停止 token ID 等补充原因，普通长度停止可为 None。
        prefill_stats: 首次有效输出携带的 prefill 统计，后续不会重复发送。
    """
    request_id: str  # 逻辑请求 ID，不是当前 batch 的物理行号。
    new_token_ids: list[int]  # 已回收的真实增量，不含输出占位。
    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None
    prefill_stats: PrefillStats | None = None

    @property
    def finished(self) -> bool:
        """根据完成原因判断本条增量是否同时通知请求结束。

        Returns:
            finished: finish_reason 非 None 时为 True，表示本条增量同时通知请求结束。
        """
        return self.finish_reason is not None


class EngineCoreOutputs(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    # 按请求保存增量对象，array_like 压缩的是序列化布局。
    """同一客户端在一次 Engine 回收中得到的输出集合。

    Attributes:
        outputs: 多个请求的本次增量，默认独立空列表由 msgspec 创建。
        scheduler_stats: 本次可选调度快照，不代表每个请求各有一份。
        timestamp: 创建时的单调时钟秒数，供进程内延迟测量使用。
        finished_requests: 可选完成请求集合，由 Scheduler 的通知配置决定是否填入。
    """
    outputs: list[EngineCoreOutput] = []
    scheduler_stats: SchedulerStats | None = None
    timestamp: float = 0.0

    finished_requests: set[str] | None = None

    def __post_init__(self):
        """未显式指定时间戳时填入单调时钟，不覆盖调用方提供的非零值。"""
        # 单调时钟用于同一基准内计算耗时，不表示跨机器共享的墙上时间。
        if self.timestamp == 0.0:
            self.timestamp = time.monotonic()
