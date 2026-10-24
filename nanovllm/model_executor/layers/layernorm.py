"""使用 PyTorch 编译实现 RMSNorm，以及残差相加后归一化的合并路径。"""
import torch
from torch import nn


class RMSNorm(nn.Module):

    """对最后一维做均方根归一化，再乘逐通道可学习权重，不减去均值。"""
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        """创建 hidden_size 个初始为 1 的通道权重，并保存防止除零的 eps。"""
        super().__init__()
        self.eps = eps  # 加在均方内，避免零向量导致 rsqrt 溢出。
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """使用 FP32 统计最后一维均方，恢复输入精度后乘归一化权重。

        Args:
            x: 最后一维与 weight 长度一致的输入。

        Returns:
            x: 与输入形状相同的归一化结果。

        Side Effects:
            中间计算使用原地乘法；FP32 输入的 float() 可返回自身，因此可能修改 x。
            调用方不能依赖输入在执行后仍保留原值。
        """
        orig_dtype = x.dtype
        x = x.float()
        # 这是均方而不是去均值后的方差；保留末维以便逐通道广播。
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """先以 FP32 相加残差，再返回归一化结果和相加后的残差。

        Args:
            x: 当前层输出，最后一维与通道权重一致。
            residual: 与 x 形状可广播相加的残差，正常模型中两者同形。

        Returns:
            (x, residual):
                x: 归一化后的相加结果，与输入形状相同。
                residual: 输入 x 与输入 residual 相加后的未归一化结果，按原输入 dtype 保存。

        Side Effects:
            使用原地相加和归一化，FP32 输入可能与中间 tensor 共享存储；
            不保证原输入或返回残差在所有 dtype 下拥有独立存储。
        """
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        # 残差路径传递相加后的值，归一化输出另由下方计算。
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """无 residual 时返回归一化 tensor；提供 residual 时返回结果与残差二元组。

        Returns:
            默认返回单个 tensor；启用额外返回时返回二元组：
                x: 未提供 residual 时，返回归一化 tensor。
                residual: 提供 residual 时额外返回未归一化的残差和，此时整体为 (x, residual)。
        """
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
