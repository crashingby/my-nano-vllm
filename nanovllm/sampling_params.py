from dataclasses import dataclass
import math


@dataclass(slots=True)
class SamplingParams:
    """当前请求的生成参数；仅支持正 temperature 随机采样。"""

    temperature: float = 1.0
    max_tokens: int = 16
    min_tokens: int = 0
    ignore_eos: bool = False
    stop_token_ids: list[int] | None = None
    _eos_token_id: int | None = None

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
        # min_tokens 不能超过输出预算，否则停止条件无法在预算内满足。
        if (
            isinstance(self.min_tokens, bool)
            or not isinstance(self.min_tokens, int)
            or not 0 <= self.min_tokens <= self.max_tokens
        ):
            raise ValueError("min_tokens must be an integer between 0 and max_tokens")

    @property
    def eos_token_id(self) -> int | None:
        """返回 Engine 注入的停止用 EOS；ignore_eos 请求通常由 Engine 设为 None。

        Returns:
            eos_token_id: Engine 注入的停止用 EOS ID；ignore_eos 请求通常为 None。
        """
        return self._eos_token_id
