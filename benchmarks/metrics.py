"""同步 Engine 回放的计数与吞吐指标，不依赖某一种数据集格式。"""

from dataclasses import asdict, dataclass
from typing import Any

from nanovllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class BenchmarkStats:
    """记录一次回放的实际工作量与计时区间。

    输入量按提交的逻辑 prompt 统计，输出量按 Engine 实际生成的 token 统计。
    scheduled_tokens 来自 step() 的非负返回值，不包含 Graph padding。
    elapsed 只覆盖请求提交与执行，不包含数据准备、模型初始化和最终清理。
    """

    num_prompts: int  # 预处理后实际接受的请求数，可能少于 CLI 请求数量。
    num_submitted: int = 0  # 已成功调用 add_request() 的数量。
    num_finished: int = 0  # 已收到完整 completion 的数量。
    num_input_tokens: int = 0  # 即使命中 prefix cache，也按逻辑输入长度计数。
    num_output_tokens: int = 0  # EOS 或长度结束后实际返回的生成 token 数。
    num_scheduled_tokens: int = 0  # 各轮真实计算量，可能包含抢占重算。
    num_engine_steps: int = 0  # 计时循环内调用 step() 的次数，不含最后清理调用。
    elapsed: float = 0.0  # 请求提交到全部完成的秒数，由回放层在结束时写入。

    @property
    def output_tps(self) -> float:
        """返回全部并发请求的输出 token 吞吐，单位为 token/s。

        elapsed 覆盖整次回放；此指标包含首 token，使用实际完成的输出长度。
        """
        # 计时尚未结束时拒绝计算，避免将无效测量包装为零吞吐。
        if self.elapsed <= 0:
            raise ValueError("elapsed must be positive before calculating TPS")
        return self.num_output_tokens / self.elapsed

    def to_dict(self) -> dict[str, Any]:
        """生成可序列化的计数与吞吐快照，不修改当前统计对象。"""
        report = asdict(self)  # dataclass 字段保留原有计数名称和秒数单位。
        report["output_tps"] = self.output_tps  # TPS 统一使用输出 token 口径。
        report["requests_per_second"] = self.num_finished / self.elapsed
        report["total_tokens_per_second"] = (
            self.num_input_tokens + self.num_output_tokens
        ) / self.elapsed
        return report

    def log(self) -> None:
        """输出本次计数与吞吐，不修改统计对象。

        Raises:
            ValueError: 回放尚未填写有效的计时结果。
        """
        if self.elapsed <= 0:
            raise ValueError("elapsed must be positive before reporting throughput")
        total_tokens = self.num_input_tokens + self.num_output_tokens
        logger.info("Benchmark finished.")
        logger.info("num_prompts: %d", self.num_prompts)
        logger.info("num_submitted: %d", self.num_submitted)
        logger.info("num_finished: %d", self.num_finished)
        logger.info("num_engine_steps: %d", self.num_engine_steps)
        logger.info("input_tokens: %d", self.num_input_tokens)
        logger.info("output_tokens: %d", self.num_output_tokens)
        logger.info("scheduled_tokens: %d", self.num_scheduled_tokens)
        logger.info("elapsed: %.3f s", self.elapsed)
        logger.info("requests/s: %.3f", self.num_finished / self.elapsed)
        logger.info("total_tokens/s: %.3f", total_tokens / self.elapsed)
        logger.info("output_tokens/s: %.3f", self.output_tps)
