"""CPU/GPU 双缓冲运行时工具。"""

import numpy as np
import torch


class CpuGpuBuffer:
    """持有可变 CPU staging tensor 及其 device mirror。

    BlockTable 等 Worker 元数据先通过 CPU/NumPy view 低开销更新，再在每次模型
    执行前拷贝到 device。这里刻意在 inference mode 之外创建可变 tensor，确保之后的
    原地更新始终合法。

    Args:
        *size: CPU 与 GPU 两侧共用的 tensor 维度。
        dtype: tensor 元素类型。
        device: 接收 GPU mirror 的设备。
        pin_memory: CPU tensor 是否锁页，以支持异步 H2D 拷贝。
        with_numpy: 是否暴露 CPU tensor 的 NumPy view。``torch.bfloat16`` 不支持
            此选项。

    Attributes:
        cpu: 可变的 CPU staging tensor。
        gpu: 可变的 device mirror tensor。
        np: ``with_numpy`` 为真时，``cpu`` 对应的 NumPy view。
    """

    def __init__(
        self,
        *size: int | torch.SymInt,
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool,
        with_numpy: bool = True,
    ) -> None:
        # 这些 buffer 属于运行时状态；在关闭 inference mode 时创建，以允许后续原地写入。
        """分配同形 CPU staging 与设备镜像，可选暴露共享存储的 NumPy view。

        Args:
            *size: 两侧 tensor 的维度。
            dtype: 元素类型，with_numpy=True 时不能是 bfloat16。
            device: 镜像设备。
            pin_memory: 是否使用锁页 CPU 内存。
            with_numpy: 是否创建 cpu 的 NumPy view，而不是另复制一份数组。

        Side Effects:
            分配并清零两侧存储，主动关闭 inference mode 以允许后续原地改写。
            创建 NumPy view 失败时抛出异常，调用方负责保证 dtype 约束。
        """
        with torch.inference_mode(False):
            self.cpu = torch.zeros(
                *size,
                dtype=dtype,
                device="cpu",
                pin_memory=pin_memory,
            )
            self.gpu = torch.zeros_like(self.cpu, device=device)  # 独立存储，不与 CPU 自动同步。

        if with_numpy:
            # NumPy 无法直接表达 PyTorch bfloat16，不能静默改变精度后再暴露 view。
            if dtype == torch.bfloat16:
                raise ValueError(
                    "Bfloat16 torch tensor 无法直接转换为 NumPy array；请设置 "
                    "with_numpy=False"
                )
            self.np: np.ndarray = self.cpu.numpy()  # 原地写 np 会直接修改 CPU staging。

    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
        """将全部或前缀 CPU tensor 拷贝到 device。

        Args:
            n: 第一维需要拷贝的前缀长度。``None`` 表示拷贝完整 tensor。

        Returns:
            gpu_tensor: 更新后的 device tensor，或被拷贝的前缀 view。

        Side Effects:
            在当前流提交 non_blocking 拷贝。返回不代表设备已经读完 CPU 源；
            调用方复用 staging 前需通过流顺序或事件确保上一份数据已被消费。
        """
        if n is None:
            return self.gpu.copy_(self.cpu, non_blocking=True)
        return self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)

    def copy_to_cpu(self, n: int | None = None) -> torch.Tensor:
        """将全部或前缀 device tensor 拷贝回 CPU。

        使用 pinned memory 时该拷贝可能是非阻塞的；调用方读取 CPU 结果前必须同步
        产生这些数据的 device 工作。

        Args:
            n: 第一维需要拷贝的前缀长度。``None`` 表示拷贝完整 tensor。

        Returns:
            cpu_tensor: 更新后的 CPU tensor，或被拷贝的前缀 view。
        """
        if n is None:
            return self.cpu.copy_(self.gpu, non_blocking=True)
        return self.cpu[:n].copy_(self.gpu[:n], non_blocking=True)
