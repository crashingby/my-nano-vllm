from dataclasses import dataclass


@dataclass
class DebugPerfStats:
    ## Stats for debugging the metrics calculation
    calc_duration: float = 0.0  # time spent calculating these stats
    num_prefill_requests: int = 0
    num_decode_requests: int = 0
    context_breakdown: dict[str, int] | None = None
    num_flops_per_gpu_breakdown: dict[str, int] | None = None
    num_read_bytes_per_gpu_breakdown: dict[str, int] | None = None
    num_write_bytes_per_gpu_breakdown: dict[str, int] | None = None
@dataclass
class PerfStats:
    num_flops_per_gpu: int = 0
    num_read_bytes_per_gpu: int = 0
    num_write_bytes_per_gpu: int = 0
    debug_stats: DebugPerfStats | None = None