"""性能估算的输出结构；字段是计数或字节数，不是 CUDA event 实测耗时。"""
from dataclasses import dataclass


@dataclass
class DebugPerfStats:
    """保存可选性能估算明细，供核对分类和单位。

    当前结构仅承载数据，字段不会自动采集。breakdown 按调用方提供的分类名聚合，
    不保证每轮都存在，也不代表已经接入 profiler。
    """
    calc_duration: float = 0.0  # 统计计算本身的耗时，单位由填充方约定。
    num_prefill_requests: int = 0  # 归入 prefill 的请求数。
    num_decode_requests: int = 0  # 归入 decode 的请求数。
    context_breakdown: dict[str, int] | None = None  # 按上下文分类的明细。
    num_flops_per_gpu_breakdown: dict[str, int] | None = None  # 单 GPU 浮点操作分类。
    num_read_bytes_per_gpu_breakdown: dict[str, int] | None = None  # 单 GPU 读取字节分类。
    num_write_bytes_per_gpu_breakdown: dict[str, int] | None = None  # 单 GPU 写入字节分类。
@dataclass
class PerfStats:
    """单批、单 GPU 的估算计算量与访存量，可附带调试分类明细。

    num_flops_per_gpu 单位为浮点操作次数，read/write 字段单位为字节。
    当前 Scheduler 允许该结构为 None，定义字段不意味着运行时已采集。
    """
    num_flops_per_gpu: int = 0
    num_read_bytes_per_gpu: int = 0
    num_write_bytes_per_gpu: int = 0
    debug_stats: DebugPerfStats | None = None
