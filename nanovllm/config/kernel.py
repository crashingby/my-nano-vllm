"""指定模型层构造时绑定的算子实现，不保存请求级执行状态。"""

from typing import Literal

from nanovllm.config.utils import config

SiluAndMulBackend = Literal["torch", "triton", "cutedsl"]


@config
class KernelConfig:
    """选择算子后端，由模型层在构造时读取并固定本实例的执行入口。

    torch 使用 IR 的 native 函数并保留 torch.compile；triton 使用已注册的
    持久化 kernel；cutedsl 使用 CuTe DSL 的持久化 kernel，按需加载可选依赖。
    改变配置后需重新创建模型层；CUDA Graph 也随新模型重新预热和捕获。
    """

    silu_and_mul_backend: SiluAndMulBackend = "torch"  # 默认沿用已编译的 PyTorch 路径。

    def __post_init__(self) -> None:
        """校验实现名称，硬件与输入支持条件由所选实现负责。"""
        if self.silu_and_mul_backend not in ("torch", "triton", "cutedsl"):
            raise ValueError(f"Unknown SiluAndMul backend: {self.silu_and_mul_backend!r}")
