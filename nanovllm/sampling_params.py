from dataclasses import dataclass
import math


@dataclass(slots=True)
class SamplingParams:
    """当前请求的生成参数；仅支持正 temperature 随机采样。"""

    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        """校验正 temperature 和生成长度，避免无效采样或越过输出长度约定。"""
        # 保留原有的最小 temperature 限制；当前没有 temperature=0 的 greedy 分支。
        if not math.isfinite(self.temperature) or self.temperature <= 1e-10:
            raise ValueError("temperature must be finite and greater than 1e-10")
        # 当前每个请求至少生成一个 token，不支持 max_tokens=None 的无限生成约定。
        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens < 1
        ):
            raise ValueError("max_tokens must be a positive integer")
