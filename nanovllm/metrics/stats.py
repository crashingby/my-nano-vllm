"""请求 prefill、前缀缓存和单轮 Scheduler 状态的统计数据结构。"""
from dataclasses import dataclass, field
from nanovllm.compilation.cuda_graph import CUDAGraphStat
from nanovllm.metrics.perf import PerfStats
@dataclass
class BaseCacheStats:
    """保存一个统计窗口的请求数、查询量和命中量。

    queries/hits 的单位由具体缓存定义；PrefixCacheStats 使用 token 数。
    reset 表示统计重置通知，不会在构造该对象时自动重置缓存。
    """

    reset: bool = False
    """是否携带缓存统计重置通知。"""

    requests: int = 0
    """本窗口首次接纳请求的查询次数。"""

    queries: int = 0
    """本窗口查询量，前缀缓存使用 token 数。"""

    hits: int = 0
    """本窗口命中量，与 queries 使用相同单位。"""

@dataclass
class PrefillStats:
    """记录首次接纳时的 prompt 总量、缓存命中量和待本地计算量。

    num_computed_tokens 是 prompt 总量减缓存命中，不是 Request 中会提前推进的
    同名调度进度。该结构没有计时，尚未完成的分块 prefill 也可先记录计划计算量。
    """
    # 当前只有本地缓存，以下三项单位均为 prompt token。
    num_prompt_tokens: int = 0
    num_computed_tokens: int = 0
    num_cached_tokens: int = 0

    def set(
        self,
        num_prompt_tokens: int,
        num_cached_tokens: int,
    ):
        """用 prompt 总量和缓存命中量覆盖本请求的 prefill 统计。

        Args:
            num_prompt_tokens: 初始 prompt 的总 token 数。
            num_cached_tokens: 可复用前缀 token 数，应介于 0 与 prompt 总量之间。

        Side Effects:
            原地写入三个计数，不累计，也不额外验证参数范围。
        """
        self.num_prompt_tokens = num_prompt_tokens
        self.num_computed_tokens = num_prompt_tokens - num_cached_tokens
        self.num_cached_tokens = num_cached_tokens

@dataclass
class PrefixCacheStats(BaseCacheStats):
    """区分首次接纳与抢占恢复的前缀缓存统计。

    继承字段记录首次接纳，preempted_* 记录恢复请求。两组 queries/hits 都使用
    token 数；读取窗口统计和重置标记由 KVCacheManager 控制。
    """

    preempted_requests: int = 0
    """抢占恢复请求的查询次数，与继承字段分开累计。"""

    preempted_queries: int = 0
    """抢占恢复请求查询的 token 总数。"""

    preempted_hits: int = 0
    """抢占恢复请求命中的 token 总数。"""

    def record(self, num_tokens: int, num_hits: int, preempted: bool) -> None:
        """把一次查询累加到首次接纳或抢占恢复对应的计数桶。

        Args:
            num_tokens: 本次查询覆盖的 token 数。
            num_hits: 其中命中缓存的 token 数。
            preempted: 是否为曾被抢占的请求，决定写入哪一组字段。

        Side Effects:
            对应请求数加一，并累加查询数和命中数；不改变物理缓存。
        """
        if preempted:
            # 重算恢复单独统计，避免与首次接纳的缓存命中率混合。
            self.preempted_requests += 1
            self.preempted_queries += num_tokens
            self.preempted_hits += num_hits
        else:
            # 首次接纳的查询写入继承字段。
            self.requests += 1
            self.queries += num_tokens
            self.hits += num_hits

@dataclass
class SchedulerStats:
    """一次结果回收后的 Scheduler 快照，计数与当前队列状态对应。

    kv_cache_usage 是去掉保留空块后的物理 block 使用比例，范围为 0 到 1。
    Graph 与性能估算字段为可选观测结果，不决定实际调度或执行模式。
    """
    num_running_reqs: int = 0  # RUNNING 队列长度，不等于本批实际执行请求数。
    num_waiting_reqs: int = 0  # waiting 队列长度，包含待接纳和待恢复请求。
    num_skipped_waiting_reqs: int = 0  # skipped_waiting 队列的当前长度。
    kv_cache_usage: float = 0.0  # 物理 block 使用比例，不是字节数。
    prefix_cache_stats: PrefixCacheStats = field(default_factory=PrefixCacheStats)
    cudagraph_stats: CUDAGraphStat | None = None

    perf_stats: PerfStats | None = None
