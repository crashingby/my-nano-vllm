# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""为共享内存等待提供按需构建的 CPU 扩展，兼容当前 Python 3.10～3.12。

MessageQueue 只在 VLLM_USE_SPINLOOP_EXT=1 时导入本模块。导入会编译或加载
PyTorch 扩展缓存中的产物；不使用 CUDA，不要求安装 vLLM 的二进制扩展。
构建失败转换为 ImportError，调用方可继续使用原有 Python/ZMQ 等待路径。
"""

from collections.abc import Callable
from pathlib import Path
import platform
from types import ModuleType
from typing import Any


def _load_extension() -> ModuleType:
    """使用当前解释器的完整 CPython API 构建并加载原生模块。

    Returns:
        extension: 导出 spinloop(buffer, callback, timeout) 的原生模块。

    Raises:
        ImportError: 编译器、Ninja 或动态库加载不可用。

    Side Effects:
        首次加载写入 PyTorch 扩展缓存，可用 TORCH_EXTENSIONS_DIR 指定目录。
        缓存和构建锁由 PyTorch 管理，允许多个 spawn Worker 共用编译产物。
    """
    try:
        from torch.utils.cpp_extension import load

        extra_cflags = ["-O2"]  # 只编译 host C++，不引入 CUDA Toolkit 依赖。
        if platform.machine().lower() in {"x86_64", "amd64", "i386", "i686"}:
            extra_cflags.append("-mmwaitx")  # 指令能否执行仍由原生模块在运行时检测。
        return load(
            name="nanovllm_spinloop",
            sources=[str(Path(__file__).resolve().parent / "csrc" / "spinloop.cpp")],
            extra_cflags=extra_cflags,
            with_cuda=False,
            is_python_module=True,
        )
    except Exception as error:
        raise ImportError(f"Cannot build or load the CPU spinloop extension: {error}") from error


_extension = _load_extension()  # 显式导入时加载；关闭队列开关不会导入本模块。


def spinloop(
    buffer: Any,
    callback: Callable[[], bool],
    timeout: float = 0.0,
) -> bool:
    """等待共享内存条件满足，期间按 CPU 能力使用等待指令并释放 GIL。

    Args:
        buffer: 支持连续 buffer 协议的共享元数据，等待期间保持有效。
        callback: 持有 GIL 时检查条件，返回 True 表示可以继续；异常直接传播。
        timeout: 等待秒数；沿用上游规则，不大于 1e-9 时不设置超时。

    Returns:
        ready: 条件满足时为 True，超过等待时限为 False；不修改共享元数据。

    Side Effects:
        在当前 CPU 线程等待，原生循环会反复调用 callback。
        暂停指令周围释放 GIL；CPU 调度及回调耗时可能使实际返回晚于时限。
    """
    return _extension.spinloop(buffer, callback, timeout=timeout)
