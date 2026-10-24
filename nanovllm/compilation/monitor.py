"""维护进程内的 CUDA Graph 捕获许可，防止在线阶段意外重新捕获。"""
import contextlib
import time
from collections.abc import Generator

from nanovllm.config import  NanovllmConfig
from nanovllm.logger import init_logger

logger = init_logger(__name__)

# 为编译计时保留的进程级开始时刻，当前模块不主动启动或结束计时。
torch_compile_start_time: float = 0.0

cudagraph_capturing_enabled: bool = True  # 启动阶段允许捕获，预热后可由 Worker 关闭。

def validate_cudagraph_capturing_enabled() -> None:
    """检查当前是否允许启动 CUDA Graph 捕获。

    捕获入口在执行前调用。许可关闭时抛出 RuntimeError；本函数不开始捕获、
    不等待设备，也不根据当前流自行推断是否已经处于捕获中。
    """
    # 只检查显式许可，不进行 CUDA 状态查询或设备同步。
    global cudagraph_capturing_enabled
    if not cudagraph_capturing_enabled:
        raise RuntimeError(
            "CUDA graph capturing detected at an inappropriate "
            "time. This operation is currently disabled."
        )

def set_cudagraph_capturing_enabled(enabled: bool) -> None:
    """设置进程级 Graph 捕获许可，影响后续所有校验入口。

    Args:
        enabled: True 允许捕获，False 使后续校验抛出 RuntimeError。

    Side Effects:
        覆盖全局开关，不负责捕获已经开始时的取消或退出时恢复。
    """
    global cudagraph_capturing_enabled
    cudagraph_capturing_enabled = enabled
