from dataclasses import dataclass, field
from nanovllm.compilation.cuda_graph import CUDAGraphStat
from nanovllm.metrics.perf import PerfStats
@dataclass
class BaseCacheStats:
    """Stores cache hit statistics."""

    reset: bool = False
    """Whether the cache was reset."""

    requests: int = 0
    """The number of requests in this update."""

    queries: int = 0
    """The number of queries in these requests."""

    hits: int = 0
    """The number of hits in these requests."""

@dataclass
class PrefillStats:
    """Breakdown of a scheduled prefill computation.

    Fields:
        num_prompt_tokens: Total number of tokens to be prefilled.
        num_computed_tokens: Tokens to be prefilled locally (actual compute work).
        num_cached_tokens: Tokens to be prefilled without actual compute work.
        
    """
    # 目前不区分内外缓存token
    num_prompt_tokens: int = 0
    num_computed_tokens: int = 0
    num_cached_tokens: int = 0

    def set(
        self,
        num_prompt_tokens: int,
        num_cached_tokens: int,
    ):
        self.num_prompt_tokens = num_prompt_tokens
        self.num_computed_tokens = num_prompt_tokens - num_cached_tokens
        self.num_cached_tokens = num_cached_tokens

@dataclass
class PrefixCacheStats(BaseCacheStats):
    """
    Stores prefix cache hit statistics.
    - `reset`: Whether `reset_prefix_cache` was invoked.
    - `queries`: Refers to the number of tokens that were queried.
    """

    preempted_requests: int = 0
    """The number of previously preempted requests in this update."""

    preempted_queries: int = 0
    """The `queries` number for preempted requests."""

    preempted_hits: int = 0
    """The `hits` number for preempted requests."""

    def record(self, num_tokens: int, num_hits: int, preempted: bool) -> None:
        """Aggregate request information into the stats."""
        if preempted:
            # Previously preempted request
            self.preempted_requests += 1
            self.preempted_queries += num_tokens
            self.preempted_hits += num_hits
        else:
            # New request
            self.requests += 1
            self.queries += num_tokens
            self.hits += num_hits

@dataclass
class SchedulerStats:
    num_running_reqs: int = 0
    num_waiting_reqs: int = 0  # length of the "waiting" request queue
    num_skipped_waiting_reqs: int = 0  # length of the "skipped waiting" queue
    kv_cache_usage: float = 0.0
    prefix_cache_stats: PrefixCacheStats = field(default_factory=PrefixCacheStats)
    cudagraph_stats: CUDAGraphStat | None = None

    perf_stats: PerfStats | None = None