import torch
import torch.nn.functional as F

from nanovllm.ir import register_op


@register_op
def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """对最后一维为 2H 的输入计算 SiLU(gate) × up。"""
    gate, up = x.chunk(2, dim=-1)  # 将融合输入拆成两个 H 维视图。
    return F.silu(gate) * up