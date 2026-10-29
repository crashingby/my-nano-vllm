"""线性计算后端选择；构造时绑定实现，forward 不重复解析配置。"""

from collections.abc import Callable

import torch
from torch import nn
import torch.nn.functional as F


def default_unquantized_gemm(
    layer: nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """执行 PyTorch 普通精度线性变换，签名沿用 vLLM 的 GEMM 方法接口。

    Args:
        layer: 持有权重的层；当前实现无需额外读取层状态。
        x: 最后一维为本地输入特征的激活 tensor。
        weight: [本地输出特征, 本地输入特征] 权重。
        bias: 可选的本地输出偏置，是否传入由线性层决定。

    Returns:
        output: x @ weight.T + bias，设备和具体 GEMM 实现由 PyTorch 分派。
    """
    return F.linear(x, weight, bias)


def dispatch_unquantized_gemm(
    linear_backend: str = "auto",
) -> Callable[[nn.Module, torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]:
    """为普通精度方法选择 GEMM 函数，当前 auto 和 torch 均使用 PyTorch。

    auto 是选择策略名称，不执行性能搜索。后续添加 Triton/CuTe GEMM 时可在此
    按设备及实现能力扩展分派，而无需修改列并行或行并行的通信流程。

    Returns:
        gemm_impl: 接收 (layer, x, weight, bias) 的计算函数。

    Raises:
        ValueError: 显式选择了尚未实现的后端。
    """
    if linear_backend not in ("auto", "torch"):
        raise ValueError(f"Unsupported unquantized linear backend: {linear_backend!r}")
    return default_unquantized_gemm
