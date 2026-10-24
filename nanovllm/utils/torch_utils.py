"""共享 dtype 上下文、当前 CUDA stream 缓存和 Graph 输出的非 owning Tensor。

导入时包装 torch.cuda.set_stream，但不创建 stream 或初始化 CUDA。Worker 应
先绑定设备；current_stream() 首次使用时才在该线程的当前设备上创建 stream。
"""

import contextlib
import threading
from collections.abc import Generator
from typing import Any

import torch


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype) -> Generator[None, None, None]:
    """临时设置浮点 tensor 的默认精度，并在正常或异常退出时恢复。

    Args:
        dtype: 模型参数构造和权重后处理期间使用的默认浮点精度。

    Yields:
        无值；调用者在上下文中执行模型构造或加载。

    Side Effects:
        临时修改进程级 PyTorch 默认 dtype，不改变已有 tensor 的精度。
    """
    # 保存进入前的精度，支持嵌套调用与加载失败后的环境恢复。
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


# 沿用上游名称，保存原始入口；包装函数仍必须调用它才能真正切换 CUDA stream。
prev_set_stream = torch.cuda.set_stream
# CUDA 当前 stream 按线程管理，缓存也必须隔离，不能使用一个进程级 stream 变量。
_current_stream_tls = threading.local()


def _patched_set_stream(stream: torch.cuda.Stream) -> None:
    """记录本线程的 stream 对象，并转发给原始 PyTorch 设置接口。

    Args:
        stream: 调用方选择的有效 CUDA stream；保留对象引用，不创建副本。

    Side Effects:
        更新线程本地缓存，并由 prev_set_stream 切换真实的当前 stream。
        torch.cuda.stream 的进入和退出也调用此入口，因此嵌套和异常恢复会
        同步更新缓存。本方法自身不管理作用域，也不发起 Graph 捕获。
    """
    _current_stream_tls.value = stream  # 缓存原对象，后续查询无需重新构造 Stream 包装。
    prev_set_stream(stream)  # 沿用上游先记录后转发的顺序，实际设置由 PyTorch 完成。


# 安装 Python 层包装；仅替换函数引用，不调用 CUDA API，也不在导入时分配 stream。
torch.cuda.set_stream = _patched_set_stream


def current_stream() -> torch.cuda.Stream:
    """返回缓存的当前 CUDA stream，首次使用时创建并选中非默认 stream。

    对齐 vLLM 的 CUDA 分支，通过包装 torch.cuda.set_stream 跟踪 stream 切换，
    避免高频调用 torch.cuda.current_stream() 重新构造 Python Stream 对象。
    当前只支持 CUDA，省略 ROCm、CPU 占位对象及其他平台的查询分支。

    Returns:
        stream: 本线程缓存的 Stream 原对象。若已通过 torch.cuda.set_stream 或
            torch.cuda.stream 选择了 stream，直接返回该对象；否则懒创建一个
            当前设备上的非默认 stream，将它设为当前 stream 后返回。

    Side Effects:
        首次未缓存时分配 stream 并改变当前 stream，不能当作纯查询接口。
        之后读取缓存，不执行 stream 同步，不创建 CUDA Graph。

    Notes:
        Worker 应先设置 CUDA device，再使用本接口，符合当前单卡执行方式。
        与上游一样，假定 stream 切换经过被包装的 Python set_stream 入口；
        直接从 C/C++ 或私有 _cuda_setStream 切换不会自动更新此缓存。
    """
    if not hasattr(_current_stream_tls, "value") or _current_stream_tls.value is None:
        # 非默认 stream 可供 CUDA Graph 捕获；通过包装入口同时更新真实状态和缓存。
        torch.cuda.set_stream(torch.cuda.Stream())
    return _current_stream_tls.value


def weak_ref_tensor(tensor: Any) -> Any:
    """为非空 CUDA Tensor 创建不拥有原始显存分配的别名。

    与 vLLM 一致，非 Tensor 对象和空 Tensor 原样返回；非空 CPU Tensor
    不受支持。这里返回的仍是 Tensor，不是 Python weakref.ref。

    Args:
        tensor: 原始 Tensor，或需要原样透传的其他对象。

    Returns:
        weak_tensor: 与输入共享数据地址、shape、stride、dtype 和 device 的新 Tensor。
            不复制数据，也不持有原始 Tensor 或其 storage 的强引用。

    Notes:
        该别名自身不能保活原始分配；调用方必须保证数据地址仍然有效。
        Graph 输出依赖存活的 Graph pool，并应在下一次覆盖前消费。
        首次调用会加载原生扩展，Graph Wrapper 在捕获开始前完成此步骤。
    """
    if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
        from nanovllm._C import load_extension

        load_extension()  # 普通调用可按需加载；Graph 路径已在捕获前加载完成。
        return torch.ops._nanovllm_C.weak_ref_tensor(tensor)
    return tensor  # 空 Tensor 不需要为其分配创建非 owning 别名。


def weak_ref_tensors(
    tensors: torch.Tensor | list[Any] | tuple[Any, ...],
) -> torch.Tensor | list[Any] | tuple[Any, ...]:
    """按 vLLM 的单层容器规则转换 Tensor、list 或 tuple。

    Args:
        tensors: 单个 Tensor，或模型返回的 Tensor 列表、元组。
            容器中的非 Tensor 项原样透传，不递归处理嵌套容器。

    Returns:
        weak_tensors: 单个 Tensor 的非 owning 别名，或保持 list/tuple 类型的新容器。
            输入容器不被修改；各 Tensor 的数据不复制。

    Raises:
        ValueError: 顶层类型不属于 Tensor、list 或 tuple。

    Notes:
        暂不支持流水线并行，因此省略上游 IntermediateTensors 分支。
        返回值的数据有效期遵循 weak_ref_tensor 的地址生命周期要求。
    """
    if isinstance(tensors, torch.Tensor):
        return weak_ref_tensor(tensors)
    if isinstance(tensors, list):
        return [weak_ref_tensor(tensor) for tensor in tensors]
    if isinstance(tensors, tuple):
        return tuple(weak_ref_tensor(tensor) for tensor in tensors)
    raise ValueError("Invalid type for tensors")
