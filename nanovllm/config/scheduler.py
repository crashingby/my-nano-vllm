"""连续批处理调度器的静态配置。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SchedulerConfig:
    """描述 Scheduler 每轮可提交给模型的连续批处理容量。

    当前 Scheduler 固定使用 FCFS、支持 chunked prefill，并且只支持文本生成。因而
    没有保留 vLLM 中与 priority policy、异步调度、多模态 encoder、投机解码和
    pipeline parallel 相关的配置项。

    Attributes:
        max_num_batched_tokens: 单次 ``SchedulerOutput`` 最多调度的 token 总数。
            一个请求的 prefill 可以跨多个 step 被分块提交。
        max_num_seqs: 一个 step 最多同时驻留在连续 batch 中的 request 数。
    """

    # 保留项目原有默认值，避免纯配置重构改变 benchmark 的默认执行规模。
    max_num_batched_tokens: int = 16384
    # 该限制既约束 Scheduler.running，也决定 InputBatch 的固定槽位数量。
    max_num_seqs: int = 512

    def __post_init__(self) -> None:
        """校验不依赖模型结构的 batch 容量约束。

        Raises:
            ValueError: token 预算或 request 容量不是正整数时抛出。
        """
        # 每轮没有 token 预算便无法形成合法的模型输入。
        if self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive")
        # 连续 batch 至少要能容纳一个 request。
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")

    def verify_with_model_config(self, max_model_len: int) -> None:
        """验证 batch 容量与模型上下文长度共同形成的约束。

        当前 Scheduler 始终允许 chunked prefill，因此 ``max_num_batched_tokens``
        可以小于 ``max_model_len``：长 prompt 会拆到多个 step 处理。每个活跃
        request 至少需要一个 token 预算，故 token 预算不能小于 request 上限。

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
