"""benchmark 共用的 NVTX 范围和生成随机数初始化工具。"""

from collections.abc import Generator
from contextlib import contextmanager
import random

import torch


@contextmanager
def nvtx_range(name: str) -> Generator[None, None, None]:
    """为当前 CPU 调用区间添加 NVTX 标记，异常退出时也配对结束。

    无 CUDA 时为空作用域，便于在 CPU 上检查数据准备与回放协议。
    此接口不启动 profiler，也不改变 GPU stream 或同步行为。
    """
    enabled = torch.cuda.is_available()  # 保存进入时的状态，确保 push/pop 成对。
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


def set_random_seed(seed: int) -> None:
    """重置 Python 和 PyTorch 的随机数状态；在 Engine 预热完成后调用。

    同步重置 CPU 和 CUDA 的 PyTorch 随机数；不启用确定性 kernel，也不保证
    不同 batch 调度或 Graph 模式下的采样序列相同。
    """
    random.seed(seed)
    torch.manual_seed(seed)
