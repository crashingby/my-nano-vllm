import os
import contextlib
import multiprocessing
import ctypes
from .platform_utils import cuda_is_initialized
from nanovllm.platforms.interface import in_wsl
from nanovllm.logger import init_logger,set_vllm_process_name
import nanovllm.envs as envs

logger = init_logger(__name__)

def _maybe_force_spawn():
    """Check if we need to force the use of the `spawn` multiprocessing start
    method.
    """
    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn":
        return

    reasons = []
    
    if cuda_is_initialized():
        reasons.append("CUDA is initialized")


    if in_wsl():
        reasons.append("WSL is detected and NVML is not compatible with fork")

    if reasons:
        logger.warning(
            "We must use the `spawn` multiprocessing start method. "
            "Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. "
            "See https://docs.vllm.ai/en/latest/usage/"
            "troubleshooting.html#python-multiprocessing "
            "for more information. Reasons: %s",
            "; ".join(reasons),
        )
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

_OMP_PAUSE_HARD = 2  # omp_pause_resource_t, OpenMP 5.0
def _release_libgomp_threads() -> None:
    """Join libgomp's idle worker threads so a forked child can use OpenMP.

    libgomp has no fork handler: once this process has run a parallel region
    (e.g. any multi-threaded torch CPU op), a forked child deadlocks in its
    own first parallel region. Pausing the runtime (OpenMP 5.0) joins the
    workers; this process recreates them on its next parallel region.
    """
    try:
        paths = find_loaded_libraries("libgomp")
    except OSError:
        return
    for path in paths:
        with contextlib.suppress(OSError, AttributeError):
            ctypes.CDLL(path).omp_pause_resource_all(_OMP_PAUSE_HARD)

def get_mp_context():
    """Get a multiprocessing context with a particular method (spawn or fork).
    By default we follow the value of the VLLM_WORKER_MULTIPROC_METHOD to
    determine the multiprocessing method (default is fork). However, under
    certain conditions, we may enforce spawn and override the value of
    VLLM_WORKER_MULTIPROC_METHOD.
    """
    _maybe_force_spawn()
    # (ROCm): Sync GPU visibility env vars so spawned children inherit
    # consistent values. Must run after _maybe_force_spawn and regardless
    # of whether spawn was already set.
    # _sync_visible_devices_env_vars()
    mp_method = envs.VLLM_WORKER_MULTIPROC_METHOD
    if mp_method == "fork":
        _release_libgomp_threads()
    return multiprocessing.get_context(mp_method)


def set_process_title(
    name: str,
    suffix: str = "",
    prefix: str = envs.VLLM_PROCESS_NAME_PREFIX,
) -> None:
    """设置当前进程在 ps / htop 中显示的标题，与日志前缀分别配置。

    Args:
        name: 进程职责与并行身份，例如 Worker_TP0。
        suffix: 可选后缀，为空时不追加下划线。
        prefix: 进程标题前缀，默认读取 VLLM_PROCESS_NAME_PREFIX。

    Side Effects:
        使用 setproctitle 修改操作系统可见的进程标题；缺少该包时跳过。
        不修改 logger 的显示身份，日志前缀由 decorate_logs() 单独设置。
    """
    try:
        import setproctitle
    except ImportError:
        return

    if suffix:
        name = f"{name}_{suffix}"

    setproctitle.setproctitle(f"{prefix}::{name}")

def decorate_logs(
    process_name: str | None = None, *, skip_if_decorated: bool = False
) -> None:
    """设置当前进程的日志身份，formatter 在输出时读取它。

    Args:
        process_name: Worker_TP0 等显示名；省略时使用 multiprocessing 进程名。
        skip_if_decorated: 当前 PID 已设置身份时保留原名，fork 继承的父进程
            身份不阻止子进程设置自己的名字。

    Side Effects:
        更新本进程的日志显示状态，不重建已有 logger、handler 或记录工厂。
        使用项目 formatter 的日志后续会显示该名字和 PID。
    """
    if process_name is None:
        process_name = multiprocessing.current_process().name

    set_vllm_process_name(process_name, skip_if_set=skip_if_decorated)

def find_loaded_libraries(lib_name: str) -> list[str]:
    """
    According to https://man7.org/linux/man-pages/man5/proc_pid_maps.5.html,
    the file `/proc/self/maps` contains the memory maps of the process, which includes the
    shared libraries loaded by the process. We can use this file to find the path of the
    loaded library.
    """  # noqa
    # Match the mapped file's name, not the whole line: an unrelated library
    # whose name merely contains lib_name (e.g. TileLang's libcudart_stub.so
    # when looking for libcudart) or a directory component containing it must
    # not win. Legitimate filenames are {lib_name}.so[.*] or a name-mangled
    # {lib_name}-<hash>.so[.*], and /proc/self/maps is ordered by mapping
    # address rather than load order, so a substring hit is a
    # nondeterministic hijack.
    paths: dict[str, None] = {}
    with open("/proc/self/maps") as f:
        for line in f:
            start = line.find("/")
            if start == -1:
                continue
            path = line[start:].strip()
            filename = path.rsplit("/", maxsplit=1)[-1]
            if filename.startswith((f"{lib_name}.", f"{lib_name}-")):
                paths[path] = None
    return list(paths)
