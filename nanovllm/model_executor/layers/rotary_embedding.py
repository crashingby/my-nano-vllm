"""按完整序列位置为 Query/Key 应用 RoPE，复用预计算的 cos/sin 表。"""
from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """将最后一维的前后两半成对旋转，使用 FP32 计算后恢复输入 dtype。

    Args:
        x: 正常为 [num_tokens, num_heads, head_size] 的 Q 或 K。
        cos: 可向 x 的半维 head 通道广播的余弦值。
        sin: 同形的正弦值。

    Returns:
        output: 与 x 同形、同 dtype 的旋转结果，不原地覆盖输入。
    """
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    """缓存各位置的正弦余弦值，采用前后半维配对的全 head RoPE 布局。"""
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        """预计算 [最大位置数, 1, head_size] 的 cos/sin 缓存。

        Args:
            head_size: 一个 attention head 的通道数。
            rotary_dim: 参与旋转的通道数，当前必须等于 head_size。
            max_position_embeddings: 可索引的位置容量，不是本轮 token 数。
            base: RoPE 频率的指数基数。

        Side Effects:
            注册不写入 state_dict 的 cos_sin_cache buffer，随模块设备迁移。
        """
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        # 每对通道共享一个频率，随后与各逻辑位置做外积。
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # 中间单例轴用于向不同数量的 Q/K heads 广播。
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """按每个 token 的完整序列位置，分别旋转 Q 和 K。

        Args:
            positions: [num_tokens] 的位置索引，必须处于预计算表范围内。
            query: [num_tokens, num_query_heads, head_size]。
            key: [num_tokens, num_kv_heads, head_size]，head 数可以与 Query 不同。

        Returns:
            (query, key):
                query: 旋转后的 Query，位置表沿 head 轴广播。
                key: 旋转后的 Key，不改变 Value。
        """
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    """按构造参数获取最多缓存一个实例的 RoPE 模块。

    Args:
        head_size: 一个 head 的通道数。
        rotary_dim: 旋转通道数，当前必须等于 head_size。
        max_position: cos/sin 表的位置容量。
        base: RoPE 指数基数。

    Returns:
        rotary_emb: 参数命中时返回同一模块对象；改变参数会替换最近一次缓存。

    Side Effects:
        lru_cache 保存模块引用，不包含设备或 dtype 参数。Runner 关闭时会清理
        此缓存，调用方不应将共享实例当作各层互不影响的独立模块。
    """
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
