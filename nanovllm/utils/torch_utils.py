"""共享 dtype 上下文、当前 CUDA stream 缓存和 Graph 输出的非 owning Tensor。

导入时包装 torch.cuda.set_stream，但不创建 stream 或初始化 CUDA。Worker 应
先绑定设备；current_stream() 首次使用时才在该线程的当前设备上创建 stream。
"""

import contextlib
import threading
from collections.abc import Generator
from typing import Any
import numpy as np
import os
import random
import sys
import threading
import torch
from nanovllm.logger import init_logger

logger = init_logger(__name__)

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



def _cgroup_cpu_limit() -> float | None:
    """Effective CPU quota of this process's cgroup, None if unlimited.

    Resolves the process's own cgroup from /proc/self/cgroup and takes the
    tightest cpu.max (v2) or cfs quota (v1) along the hierarchy.
    """
    limit: float | None = None
    try:
        with open("/proc/self/cgroup") as f:
            entries = [line.strip().split(":", 2) for line in f]

        def visit(base: str, rel_path: str, read_quota) -> None:
            nonlocal limit
            path = rel_path
            while path:
                quota = read_quota(os.path.join(base, path.lstrip("/")))
                if quota is not None:
                    limit = quota if limit is None else min(limit, quota)
                path = path.rsplit("/", 1)[0]

        def read_v2(cg_dir: str) -> float | None:
            try:
                with open(os.path.join(cg_dir, "cpu.max")) as f:
                    quota, period = f.read().split()
                return None if quota == "max" else float(quota) / float(period)
            except (OSError, ValueError):
                return None

        def read_v1(cg_dir: str) -> float | None:
            try:
                with open(os.path.join(cg_dir, "cpu.cfs_quota_us")) as f:
                    quota = int(f.read())
                if quota <= 0:
                    return None
                with open(os.path.join(cg_dir, "cpu.cfs_period_us")) as f:
                    return quota / int(f.read())
            except (OSError, ValueError):
                return None

        for entry in entries:
            if len(entry) != 3:
                continue
            _, controllers, rel_path = entry
            if controllers == "":  # cgroup v2
                visit("/sys/fs/cgroup", rel_path, read_v2)
            elif "cpu" in controllers.split(","):  # cgroup v1
                visit("/sys/fs/cgroup/cpu", rel_path, read_v1)
    except OSError:
        pass
    return limit

def available_cpu_count() -> int:
    """CPUs actually usable by this process: scheduling affinity capped by
    the cgroup CPU quota (unlike `os.cpu_count()`, which is quota-blind)."""
    if sys.platform != "linux":
        return os.cpu_count() or 1
    count = len(os.sched_getaffinity(0))
    limit = _cgroup_cpu_limit()
    if limit is not None:
        count = min(count, int(limit))
    return max(1, count)

OMP_NUM_THREADS_SET_BY_VLLM = "VLLM_OMP_NUM_THREADS_SET_BY_VLLM"

def startup_omp_num_threads(num_local_procs: int) -> int:
    """Thread count for a worker process's startup work (weight loading).

    Weight loading does CPU-parallel work, so workers benefit from more than
    one thread, but only a bounded share of the CPUs this node's workers may
    actually use: torch's default is the host core count, which ignores both
    scheduling affinity and any cgroup CPU quota, and doesn't account for the
    other workers sharing the node.
    """
    return max(1, available_cpu_count() // max(1, num_local_procs))


def set_torch_threads_for_runtime() -> None:
    """Set torch intra-op threads to 1 for steady-state serving.

    Any multi-threaded torch CPU op in the engine hot loop leaves the OMP
    workers spin-waiting after each parallel region, stealing cycles from the
    step's serial code (and burning cgroup CPU quota in containers). No
    steady-state CPU op benefits from intra-op parallelism.
    Respects an externally-set OMP_NUM_THREADS.
    """
    if (
        omp_num_threads := os.environ.get("OMP_NUM_THREADS")
    ) is not None and os.environ.get(OMP_NUM_THREADS_SET_BY_VLLM) != "1":
        try:
            if int(omp_num_threads) > 1:
                logger.warning_once(
                    "OMP_NUM_THREADS=%s is set; leaving Torch threads at %d "
                    "for serving. Multi-threaded torch CPU ops during serving "
                    "can degrade performance through spin-wait contention and "
                    "cgroup CPU-quota throttling.",
                    omp_num_threads,
                    torch.get_num_threads(),
                )
        except ValueError:
            pass
        return
    if torch.get_num_threads() != 1:
        logger.info_once(
            "Reducing Torch threads from %d to 1 for serving. Set "
            "OMP_NUM_THREADS in the external environment to override.",
            torch.get_num_threads(),
        )
        torch.set_num_threads(1)



def set_random_seed(seed: int | None, data_parallel_index: int = 0) -> None:
    if seed is not None:
        if data_parallel_index:
            # DP engines share the seed but must not sample unseeded requests alike.
            ss = np.random.SeedSequence((seed, data_parallel_index))
            seed = int(ss.generate_state(1)[0])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        from nanovllm.platforms import current_platform

        current_platform.manual_seed_all(seed)