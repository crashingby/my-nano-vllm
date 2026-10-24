"""记录本地 Engine 的逐 token 可见时间，按请求和 token 分别汇总延迟。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from statistics import fmean, median, pstdev
from typing import Any

from benchmarks.metrics import BenchmarkStats
from nanovllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class RequestLatency:
    """保存一条请求的到达、逐 token 可见时间和完成时间。

    所有时间戳单位为秒，原点是整次回放开始时的 perf_counter()。请求在调用
    add_request() 前到达；每轮 step_fn() 返回后，新增 token 使用同一观察时间。
    benchmark 持续追加时间戳，完成后再读取派生指标，不保存模型张量。
    """

    request_id: str  # 与提交给 Engine 的请求 ID 一致。
    prompt_len: int  # 包含 chat template 的逻辑输入 token 数。
    arrival_time: float  # 开始调用 add_request() 的时间，包含入队处理开销。
    token_timestamps: list[float] = field(default_factory=list)  # 仅记录实际生成 token。
    finish_time: float | None = None  # 观察到 Engine 完成结果时填写。

    def record_token(self, timestamp: float) -> None:
        """追加一个实际生成 token 的观察时间，检查生命周期及时间单调性。"""
        if self.finish_time is not None:
            raise RuntimeError(f"Request {self.request_id!r} has already finished")
        # 首 token 与到达时间比较，后续 token 与上一次观察时间比较。
        previous_time = self.token_timestamps[-1] if self.token_timestamps else self.arrival_time
        if not isfinite(timestamp) or timestamp < previous_time:
            raise ValueError(f"Non-monotonic token timestamp for {self.request_id!r}")
        self.token_timestamps.append(timestamp)  # 一个元素对应一个真实输出 token。

    def finish(self, timestamp: float, num_output_tokens: int) -> None:
        """核对完成结果的真实长度并关闭本条记录，异常时保留未完成状态。"""
        if self.finish_time is not None:
            raise RuntimeError(f"Duplicate completion for {self.request_id!r}")
        # 最后一个 token 也必须在记录中，不能因 Scheduler 删除请求而漏记。
        if len(self.token_timestamps) != num_output_tokens:
            raise RuntimeError(
                f"Token count mismatch for {self.request_id!r}: "
                f"observed={len(self.token_timestamps)}, completed={num_output_tokens}"
            )
        previous_time = self.token_timestamps[-1] if self.token_timestamps else self.arrival_time
        if not isfinite(timestamp) or timestamp < previous_time:
            raise ValueError(f"Non-monotonic completion timestamp for {self.request_id!r}")
        self.finish_time = timestamp  # 长度和时间校验成功后才提交完成状态。

    @property
    def ttft(self) -> float | None:
        """返回到达至首 token 的秒数；无输出时返回 None。

        Returns:
            ttft: 请求到达至首 token 可见的秒数，无输出时为 None。
        """
        return self.token_timestamps[0] - self.arrival_time if self.token_timestamps else None

    @property
    def e2el(self) -> float | None:
        """返回到达至完成的端到端秒数；尚未完成时返回 None。

        Returns:
            e2el: 请求到达至完成的秒数，尚未完成时为 None。
        """
        return self.finish_time - self.arrival_time if self.finish_time is not None else None

    @property
    def tpot(self) -> float | None:
        """返回请求级平均输出间隔秒数；少于两个 token 或尚未完成时返回 None。

        沿用 vLLM 的 (E2EL - TTFT) / (输出长度 - 1) 口径；当前 Engine
        在最后一个 token 所在 step 返回完成结果，因此分子也是首尾 token 间隔。

        Returns:
            tpot: 首尾 token 间隔除以输出间隔数的秒数；未完成或不足两个 token 时为 None。
        """
        if self.finish_time is None or len(self.token_timestamps) < 2:
            return None
        return (self.finish_time - self.token_timestamps[0]) / (len(self.token_timestamps) - 1)

    @property
    def itls(self) -> list[float]:
        """返回相邻输出 token 的间隔秒数，首 token 等待时间单独归入 TTFT。

        Returns:
            itls: 逐个相邻输出 token 的间隔秒数列表，不含首 token 等待。
        """
        return [
            current - previous
            for previous, current in zip(self.token_timestamps, self.token_timestamps[1:])
        ]

    def to_dict(self) -> dict[str, Any]:
        """导出原始秒数时间戳与毫秒指标；缺失的延迟在 JSON 中表现为 null。

        Returns:
            report: 包含原始秒数时间戳和毫秒派生指标的可序列化字典。
        """
        report = asdict(self)  # 保留回放起点下的时间戳，便于离线重算指标。
        report["num_output_tokens"] = len(self.token_timestamps)
        # 每请求指标与汇总报告统一用毫秒，原始 timestamp 字段继续使用秒。
        report["ttft_ms"] = self.ttft * 1000 if self.ttft is not None else None
        report["tpot_ms"] = self.tpot * 1000 if self.tpot is not None else None
        report["e2el_ms"] = self.e2el * 1000 if self.e2el is not None else None
        report["itls_ms"] = [interval * 1000 for interval in self.itls]
        return report


@dataclass(frozen=True)
class LatencyDistribution:
    """一组延迟样本的毫秒统计；空样本保留 count=0 和 None 指标。"""

    count: int  # TTFT/TPOT 按请求计数，ITL 按相邻 token 间隔计数。
    mean_ms: float | None = None  # 对当前集合中的每个样本等权求均值。
    median_ms: float | None = None  # 中位数，也是 P50。
    std_ms: float | None = None  # 总体标准差，单样本时为 0。
    min_ms: float | None = None  # 本次测量的最小值。
    max_ms: float | None = None  # 本次测量的最大值。
    p90_ms: float | None = None  # 使用排序后线性插值得到的 P90。
    p95_ms: float | None = None  # 使用相同规则得到的 P95。
    p99_ms: float | None = None  # 使用相同规则得到的 P99。

    @classmethod
    def from_seconds(cls, values: list[float]) -> LatencyDistribution:
        """将秒数样本转换成毫秒分布，分位数采用 NumPy 默认的线性插值规则。

        Returns:
            distribution: 从秒数样本计算的毫秒延迟分布；空输入对应 count=0。
        """
        if not values:
            return cls(count=0)  # 单 token 请求不会人为产生零 TPOT 或零 ITL。
        if any(not isfinite(value) or value < 0 for value in values):
            raise ValueError("Latency samples must be finite and non-negative")
        samples_ms = sorted(value * 1000 for value in values)  # 排序仅发生在测量结束后。
        return cls(
            count=len(samples_ms),
            mean_ms=fmean(samples_ms),
            median_ms=median(samples_ms),
            std_ms=pstdev(samples_ms),
            min_ms=samples_ms[0],
            max_ms=samples_ms[-1],
            p90_ms=_percentile(samples_ms, 90),
            p95_ms=_percentile(samples_ms, 95),
            p99_ms=_percentile(samples_ms, 99),
        )


def _percentile(sorted_values: list[float], percentile: float) -> float:
    """对非空有序样本做线性插值，输入和返回使用相同单位。

    Returns:
        percentile: 输入有序样本在指定百分位的线性插值结果，与输入单位相同。
    """
    position = (len(sorted_values) - 1) * percentile / 100  # 百分位映射到排序下标。
    lower = int(position)  # 左侧样本，单样本集合也能使用该计算路径。
    upper = min(lower + 1, len(sorted_values) - 1)  # 末尾位置直接使用最后一个样本。
    fraction = position - lower  # 左右样本之间的插值权重。
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


@dataclass
class LatencyBenchmarkStats(BenchmarkStats):
    """在公共吞吐计数上增加逐请求时间轴及 TTFT、TPOT、ITL 分布。

    request_latencies 按提交顺序保存。TPOT 对有效请求等权汇总，ITL 则对所有
    有效 token 间隔等权汇总，输出长度不同时二者的均值可以不同。
    """

    request_latencies: list[RequestLatency] = field(default_factory=list)  # 包含完成的记录。

    def summarize(self) -> dict[str, LatencyDistribution]:
        """汇总已完成请求的延迟分布，不修改原始记录，拒绝未完成的测量。

        Returns:
            distributions: ttft、tpot、itl 和 e2el 到毫秒分布对象的映射。
        """
        ttfts: list[float] = []  # 每个有输出的请求贡献一次首 token 等待时间。
        tpots: list[float] = []  # 每个至少输出两个 token 的请求贡献一个平均值。
        itls: list[float] = []  # 每个相邻 token 对贡献一次实际观察间隔。
        e2els: list[float] = []  # 每个完成请求贡献一次端到端延迟。
        for request in self.request_latencies:
            if request.finish_time is None:
                raise RuntimeError(f"Latency request {request.request_id!r} has not finished")
            if request.ttft is not None:
                ttfts.append(request.ttft)
            if request.tpot is not None:
                tpots.append(request.tpot)
            itls.extend(request.itls)  # chunked prefill 的空输出 step 自然计入等待间隔。
            e2els.append(request.finish_time - request.arrival_time)
        return {
            "ttft": LatencyDistribution.from_seconds(ttfts),
            "tpot": LatencyDistribution.from_seconds(tpots),
            "itl": LatencyDistribution.from_seconds(itls),
            "e2el": LatencyDistribution.from_seconds(e2els),
        }

    def log(self) -> None:
        """先输出实际工作量和 TPS，再输出各延迟分布与有效样本数。"""
        super().log()  # 公共吞吐计数与原来的系统吞吐 benchmark 使用同一口径。
        logger.info("TPS (output tokens/s): %.3f", self.output_tps)
        for name, distribution in self.summarize().items():
            if distribution.count == 0:
                logger.info("%s: samples=0, N/A", name.upper())
                continue
            logger.info(
                "%s (ms): samples=%d mean=%.3f median=%.3f std=%.3f "
                "min=%.3f max=%.3f p90=%.3f p95=%.3f p99=%.3f",
                name.upper(), distribution.count,
                distribution.mean_ms, distribution.median_ms, distribution.std_ms,
                distribution.min_ms, distribution.max_ms,
                distribution.p90_ms, distribution.p95_ms, distribution.p99_ms,
            )

    def to_dict(self) -> dict[str, Any]:
        """导出公共计数、毫秒分布和逐请求时间轴，供 JSON 报告或离线分析使用。

        Returns:
            report: 公共计数、毫秒延迟分布及逐请求时间轴的可序列化字典。
        """
        report = super().to_dict()  # 复用公共计数及输出 TPS 的计算规则。
        report["measurement"] = "local_engine_step_return"
        report["timestamp_unit"] = "seconds_since_replay_start"
        report["latency_distributions"] = {
            name: asdict(distribution) for name, distribution in self.summarize().items()
        }
        report["request_latencies"] = [request.to_dict() for request in self.request_latencies]
        return report
