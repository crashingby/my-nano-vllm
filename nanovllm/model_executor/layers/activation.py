"""为模型提供固定的 SiLU × Mul 层入口，实际计算由所选 IR 实现负责。"""

from collections.abc import Callable
from functools import lru_cache

import torch
from torch import nn

from nanovllm.config import KernelConfig, get_current_config_or_none
from nanovllm.config.kernel import SiluAndMulBackend
from nanovllm.ir.ops.activation import silu_and_mul


@lru_cache(maxsize=1)
def _get_compiled_native() -> Callable[[torch.Tensor], torch.Tensor]:
    """为 native 函数创建共享编译包装，首次执行时由 PyTorch 完成实际编译。

    各层共享同一个无状态 callable，保留原先 @torch.compile 的执行方式；
    包装只针对 native 计算函数，provider 选择在模型构造阶段完成。
    """
    native_impl = silu_and_mul.get_impl("native")  # torch 配置名映射到 IR 的 native。
    return torch.compile(native_impl.impl_fn)


class SiluAndMul(nn.Module):
    """构造时选择一个实现，逐轮 forward 直接调用已绑定的执行函数。

    Worker 加载模型时安装总配置，原有 SiluAndMul() 调用因此可自动读取
    kernel_config。单独实验时也可显式传 backend；配置作用域之外默认 torch。
    每个实例独立绑定实现，不受其他实例或 IR 默认优先级的后续修改影响。
    """

    def __init__(self, *, backend: SiluAndMulBackend | None = None) -> None:
        """读取或覆盖后端选择，按需导入 provider 并保存执行入口。

        Args:
            backend: 显式实现名，优先于当前总配置；None 从 kernel_config 读取。

        Side Effects:
            torch 路径取得共享编译包装；triton 路径导入模块以完成 IR 注册。
            cutedsl 路径按需导入 CuTe DSL 实现。此阶段不执行 kernel、不查询 GPU。
        """
        super().__init__()
        if backend is None:
            config = get_current_config_or_none()  # 普通 forward 不再读取构造上下文。
            kernel_config = config.kernel_config if config is not None else KernelConfig()
            backend = kernel_config.silu_and_mul_backend
        self.backend = backend  # 保存本实例实际采用的配置值，供 extra_repr 展示。
        self._forward_method: Callable[[torch.Tensor], torch.Tensor]
        if backend == "torch":
            self._forward_method = _get_compiled_native()
        elif backend == "triton":
            # 模块导入执行注册装饰器；torch 默认路径无需加载 Triton kernel 模块。
            import nanovllm.kernels.triton.activation  # noqa: F401

            self._forward_method = silu_and_mul.get_impl("triton")
        elif backend == "cutedsl":
            import nanovllm.kernels.cute.activation  # noqa: F401

            self._forward_method = silu_and_mul.get_impl("cutedsl")
        else:
            raise ValueError(f"Unknown SiluAndMul backend: {backend!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将 [..., 2H] 输入映射为独立的 [..., H] 输出，保持输入内容。

        torch 路径由编译器管理计算；triton/cutedsl 路径检查 CUDA、精度和连续
        布局后启动 kernel。指定实现不支持输入时抛错，便于确认实验采用的后端。
        """
        return self._forward_method(x)

    def extra_repr(self) -> str:
        """在模型结构输出中展示本层固定使用的后端。"""
        return f"backend={self.backend!r}"
