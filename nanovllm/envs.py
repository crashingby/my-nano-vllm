# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""按 vLLM 的延迟读取方式提供当前已接入的环境变量。

调用方使用 ``import nanovllm.envs as envs``，再访问 envs.VLLM_HOST_IP 等属性。
属性查询经过模块级 __getattr__()，调用 environment_variables 中对应的读取函数，
而不是在导入时固化环境变量。当前只接入网络初始化需要的主机地址和端口配置，
保留上游名称便于对照源码，不引入环境变量缓存或未使用的部署配置。
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # 仅供编辑器和类型检查器识别；运行时不绑定这些属性，访问时才读取实际环境。
    VLLM_HOST_IP: str = ""
    VLLM_PORT: int | None = None


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
}


def __getattr__(name: str) -> Any:
    """在访问已登记的模块属性时读取环境变量，并返回解析后的值。

    Args:
        name: 访问的属性名，例如 VLLM_HOST_IP，由 Python 属性查找机制传入。

    Returns:
        env_value: 对应读取函数的结果；当前为字符串、整数或 None。每次访问重新读取，
            不将结果写入模块属性，因此后续环境变量变化仍然可见。

    Raises:
        AttributeError: 属性没有登记；不将未知配置误当成空值。
        ValueError: 对应读取函数解析配置失败，例如 VLLM_PORT 不是整数。
    """
    # 只有登记的配置才使用环境变量读取协议，其余名称遵循普通模块的缺失属性语义。
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
