# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""按 vLLM 的延迟读取方式提供当前已接入的环境变量。

调用方使用 ``import nanovllm.envs as envs``，再访问 envs.VLLM_HOST_IP 等属性。
属性查询经过模块级 __getattr__()，调用 environment_variables 中对应的读取函数，
而不是在导入时固化环境变量。当前接入网络初始化与共享内存消息队列所需配置，
保留上游名称便于对照源码，不引入环境变量缓存或未使用的部署配置。
"""

from __future__ import annotations
import functools
import os
import tempfile
from collections.abc import Callable
from typing import TYPE_CHECKING, Any,Literal

if TYPE_CHECKING:
    # 仅供编辑器和类型检查器识别；运行时不绑定这些属性，访问时才读取实际环境。
    VLLM_HOST_IP: str = ""
    VLLM_PORT: int | None = None
    VLLM_RINGBUFFER_WARNING_INTERVAL: int = 60
    VLLM_RPC_BASE_PATH: str = tempfile.gettempdir()
    VLLM_USE_SPINLOOP_EXT: bool = False,
    VLLM_MQ_MAX_CHUNK_BYTES_MB: int = 16,
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS: int = 300,
    VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS: int = 5,
    VLLM_WORKER_MULTIPROC_METHOD: Literal["fork", "spawn"] = "fork",



def env_with_choices(
    env_name: str,
    default: str | None,
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
) -> Callable[[], str | None]:
    """Create a lambda that validates environment variable against allowed choices.

    Args:
        env_name: Name of the environment variable
        default: Default value if not set (can be None)
        choices: List of valid string options or callable that returns list
        case_sensitive: Whether validation should be case sensitive

    Returns:
        Lambda function for environment_variables dict

    """

    def _get_validated_env() -> str | None:
        value = os.getenv(env_name)
        if value is None:
            return default

        # Resolve choices if it's a callable (for lazy loading)
        actual_choices = choices() if callable(choices) else choices

        if not case_sensitive:
            check_value = value.lower()
            check_choices = [choice.lower() for choice in actual_choices]
        else:
            check_value = value
            check_choices = actual_choices

        if check_value not in check_choices:
            raise ValueError(
                f"Invalid value '{value}' for {env_name}. "
                f"Valid options: {actual_choices}."
            )

        return value

    return _get_validated_env

def get_vllm_port() -> int | None:
    """读取通信起始端口；未设置 VLLM_PORT 时由网络工具向系统申请端口。

    Returns:
        port: 配置的整数端口，或表示未设置的 None。此处只解析配置，不绑定 socket。

    Raises:
        ValueError: 环境变量不能解析成整数。当前不保留上游 Kubernetes URI 诊断。
    """
    # 区分未设置与显式值，不在这里替调用方选取默认固定端口。
    if "VLLM_PORT" not in os.environ:
        return None

    # port 是原始环境变量字符串；整数转换失败时保留原异常链以便定位配置错误。
    port = os.getenv("VLLM_PORT", "0")
    try:
        return int(port)
    except ValueError as err:
        raise ValueError(f"VLLM_PORT {port!r} must be a valid integer") from err


# 每项保存读取函数；导入模块时不执行这些函数，也不探测网络或初始化 CUDA。
environment_variables: dict[str, Callable[[], Any]] = {
    # 非空时直接覆盖 get_ip() 的自动探测结果；单机可显式指定 127.0.0.1。
    "VLLM_HOST_IP": lambda: os.getenv("VLLM_HOST_IP", ""),
    # 非空配置作为空闲端口搜索起点；未设置时返回 None。
    "VLLM_PORT": get_vllm_port,
    "VLLM_RINGBUFFER_WARNING_INTERVAL": lambda: int(
        os.environ.get("VLLM_RINGBUFFER_WARNING_INTERVAL", "60")
    ),
    # ZMQ 的本地 IPC socket 使用该目录；沿用 vLLM 的系统临时目录默认值。
    "VLLM_RPC_BASE_PATH": lambda: os.getenv("VLLM_RPC_BASE_PATH", tempfile.gettempdir()),
    # 只在显式开启时导入并编译 CPU 等待扩展，普通单卡路径不增加构建步骤。
    "VLLM_USE_SPINLOOP_EXT": lambda: bool(int(os.getenv("VLLM_USE_SPINLOOP_EXT", "0"))),
    "VLLM_MQ_MAX_CHUNK_BYTES_MB": lambda: int(
        os.getenv("VLLM_MQ_MAX_CHUNK_BYTES_MB", "16")
    ),
    "VLLM_WORKER_MULTIPROC_METHOD": env_with_choices(
        "VLLM_WORKER_MULTIPROC_METHOD", "fork", ["spawn", "fork"]
    ),
    "VLLM_PROCESS_NAME_PREFIX": lambda: os.getenv("VLLM_PROCESS_NAME_PREFIX", "VLLM"),
    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": lambda: int(
        os.getenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "300")
    ),
    "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS": lambda: int(
        os.getenv("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", "5")
    ),
}


def __getattr__(name: str) -> Any:
    """在访问已登记的模块属性时读取环境变量，并返回解析后的值。

    Args:
        name: 访问的属性名，例如 VLLM_HOST_IP，由 Python 属性查找机制传入。

    Returns:
        env_value: 对应读取函数的结果；当前为字符串、整数、布尔值或 None。每次访问重新读取，
            不将结果写入模块属性，因此后续环境变量变化仍然可见。

    Raises:
        AttributeError: 属性没有登记；不将未知配置误当成空值。
        ValueError: 对应读取函数解析配置失败，例如 VLLM_PORT 不是整数。
    """
    # 只有登记的配置才使用环境变量读取协议，其余名称遵循普通模块的缺失属性语义。
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")



def _is_envs_cache_enabled() -> bool:
    """Checked if __getattr__ is wrapped with functools.cache."""
    global __getattr__
    return hasattr(__getattr__, "cache_clear")


def enable_envs_cache() -> None:
    """Enables caching of environment variables. This is useful for performance
    reasons, as it avoids the need to re-evaluate environment variables on
    every call.

    NOTE: Currently, it's invoked after service initialization to reduce
    runtime overhead. This also means that environment variables should NOT
    be updated after the service is initialized.
    """
    if _is_envs_cache_enabled():
        # Avoid wrapping functools.cache multiple times
        return
    # Tag __getattr__ with functools.cache
    global __getattr__
    __getattr__ = functools.cache(__getattr__)

    # Cache all environment variables
    for key in environment_variables:
        __getattr__(key)


def disable_envs_cache() -> None:
    """Resets the environment variables cache. It could be used to isolate environments
    between unit tests.
    """
    global __getattr__
    # If __getattr__ is wrapped by functions.cache, unwrap the caching layer.
    if _is_envs_cache_enabled():
        assert hasattr(__getattr__, "__wrapped__")
        __getattr__ = __getattr__.__wrapped__