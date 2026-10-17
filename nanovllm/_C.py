"""按需编译并加载 LibTorch 算子，注册到 torch.ops._nanovllm_C。

当前扩展仅包含 host C++，无需 nvcc；PyTorch 的 JIT 缓存负责复用编译产物。
导入本模块不会编译或初始化 CUDA。Graph Wrapper 应在进入捕获前显式加载。
"""

import threading
from pathlib import Path


_extension_loaded: bool = False  # 只记录当前进程是否完成原生算子注册。
_extension_lock = threading.Lock()  # 防止同进程多个线程重复加载同一个动态库。


def load_extension() -> None:
    """首次调用时编译、加载原生扩展，之后直接返回。

    Side Effects:
        首次调用需要 C++ 编译器和 ninja，在 PyTorch 扩展缓存目录写入产物，
        并向 dispatcher 注册算子。可用 TORCH_EXTENSIONS_DIR 指定缓存位置。
        编译失败直接抛出原始异常；不要在 CUDA Graph 捕获内部首次调用。
    """
    global _extension_loaded
    if _extension_loaded:
        return  # replay 后的热路径不获取锁，也不重复调用构建系统。

    with _extension_lock:
        if _extension_loaded:
            return  # 等待另一个线程加载成功后，无需再次注册。

        from torch.utils.cpp_extension import load

        source_dir = Path(__file__).resolve().parent / "csrc"
        # 只注册 torch.ops，不创建 Python 扩展模块，因此不需要 pybind11。
        load(
            name="nanovllm_C",
            sources=[str(source_dir / "torch_bindings.cpp")],
            extra_cflags=["-O2"],
            with_cuda=False,  # 使用 LibTorch 的 CUDA Tensor API，无需编译 .cu 文件。
            is_python_module=False,
        )
        _extension_loaded = True  # 仅在编译和加载都成功后更新状态，失败后可重试。
