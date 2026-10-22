"""连续批处理调度器的静态配置。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

from pydantic import Field

from nanovllm.config.utils import config

SchedulerPolicy = Literal["fcfs", "priority"]
@config(slots=True)
class SchedulerConfig:
    """描述 Scheduler 每轮可提交给模型的连续批处理容量。

    当前 Scheduler 使用 FCFS 并支持文本生成。单轮调度预算可小于 Runner 的
    token buffer 容量；分块 prefill 策略和异步调度选项由具体执行链消费，配置
    构造成功本身不代表执行端已经支持对应功能。

    Attributes:
        max_num_batched_tokens: 单次 SchedulerOutput 最多调度的真实 token 总数。
            一个请求的 prefill 可以跨多个 step 分块提交；Runner 同时用它确定
            token 输入 buffer 的固定容量，总配置据此限制 Graph 捕获尺寸。
        max_num_seqs: 同时处于 running batch 的请求数上限，不限制 waiting 队列
            长度或一次 generate() 能提交的请求总数。Runner 用它分配 InputBatch
            行槽位、block table 和请求级 metadata，Graph 分派也受该容量约束。
        max_num_scheduled_tokens: 可选的单轮调度预算；None 表示使用 buffer 容量。
        long_prefill_token_threshold: 长 prefill 的 token 阈值，供调度策略使用。
        enable_chunked_prefill: 分块 prefill 的策略选项，默认开启。
        async_scheduling: 是否请求异步调度，默认关闭；需要执行链配套支持。
    """
    DEFAULT_MAX_NUM_BATCHED_TOKENS: ClassVar[int] = 2048
    DEFAULT_MAX_NUM_SEQS: ClassVar[int] = 128
    # Scheduler 的单轮真实 token 预算，也是 Runner token buffer 的容量上限。
    max_num_batched_tokens: int = Field(default=DEFAULT_MAX_NUM_BATCHED_TOKENS, ge=1)
    # 预算为零无法推进请求；上界需要与同一实例的 buffer 容量共同校验。
    max_num_scheduled_tokens: int | None = Field(default=None, ge=1)
    # 该限制既约束 Scheduler.running，也决定 InputBatch 的固定槽位数量。
    max_num_seqs: int = Field(default=DEFAULT_MAX_NUM_SEQS, ge=1)
    # 阈值的具体含义由 Scheduler 决定，零保留默认策略。
    long_prefill_token_threshold: int = 0
    # 保留现有分块 prefill 默认行为，尚未实现的策略由后续调度接线处理。
    enable_chunked_prefill: bool = True

    policy: SchedulerPolicy = "fcfs"
    # 现有 Engine 默认仍走同步路径，避免省略字段时意外请求异步能力。
    async_scheduling: bool = False

    def __post_init__(self) -> None:
        """在字段校验后检查调度预算不会超过 Runner 的 token buffer 容量。

        Raises:
            ValueError: 显式调度预算大于 max_num_batched_tokens。
        """
        # 正值约束由 Field 统一处理；此处只检查依赖两个字段的容量关系。
        if (
            self.max_num_scheduled_tokens is not None
            and self.max_num_scheduled_tokens > self.max_num_batched_tokens
        ):
            raise ValueError(
                "max_num_scheduled_tokens must not exceed "
                f"max_num_batched_tokens ({self.max_num_batched_tokens})"
            )

    def verify_with_model_config(self, max_model_len: int) -> None:
        """验证 batch 容量与模型上下文长度共同形成的约束。

        当前 Scheduler 始终允许 chunked prefill，因此 ``max_num_batched_tokens``
        可以小于 ``max_model_len``：长 prompt 会拆到多个 step 处理。每个活跃
        request 至少需要一个 token 预算，故 token 预算不能小于 request 上限。
        本方法由 NanovllmConfig 构造时调用，只检查关系，不修改已有容量值。

        Args:
            max_model_len: ``ModelConfig`` 已经根据模型位置编码收紧后的上下文上限。

        Raises:
            ValueError: 模型长度非法，或单轮 token 预算不足以给每个 request 至少
                分配一个 token 时抛出。
        """
        # NanovllmConfig 应在 ModelConfig 成功构造后调用本方法。
        if max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        # decode 阶段每个并发 request 至少调度一个 token，不能违反这个下界。
        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                "max_num_batched_tokens "
                f"({self.max_num_batched_tokens}) must be greater than or equal "
                f"to max_num_seqs ({self.max_num_seqs})"
            )
