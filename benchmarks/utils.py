"""benchmark 共用的 NVTX 范围和 Worker 随机状态初始化工具。"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from nanovllm.utils.torch_utils import set_random_seed as reset_random_seed

if TYPE_CHECKING:
    from nanovllm import LLM
    from nanovllm.worker.worker_base import WorkerWrapperBase


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


def _set_worker_random_seed(_worker: WorkerWrapperBase, seed: int) -> None:
    """作为 RPC callable 在 Worker 进程内重置随机状态，不返回结果。

    _worker 是 Executor 为 callable 自动绑定的 Wrapper，不由调用方传入。
    实际修改当前进程的 Python、NumPy 和 PyTorch RNG。
    """
    reset_random_seed(seed)


def set_random_seed(llm: LLM, seed: int) -> None:
    """等待所有 Worker 重置为同一个 seed，不返回结果。

    正式回放或诊断预热完成后调用；调用方应先排空在途批次。通过 callable
    RPC 修改模型所在进程的 RNG，兼容 uni 和 mp，不把父进程重置当作远程操作。
    此接口不保证不同调度顺序或 Graph 模式逐 token 一致。
    """
    llm.model_executor.collective_rpc(_set_worker_random_seed, args=(seed,))
