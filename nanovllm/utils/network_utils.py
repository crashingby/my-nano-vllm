# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""提供 Worker 通信初始化使用的本机地址、空闲端口与 TCP URI 工具。

沿用 vLLM 的 get_ip()、get_open_port() 与地址格式化流程。环境变量统一从
nanovllm.envs 延迟读取；当前没有 DP 端口保留、ZMQ、远程部署或多端口批量分配。
导入模块不会创建 socket，网络操作只在调用地址探测或端口查询接口时发生。
"""

from __future__ import annotations

import ipaddress
import socket
import warnings

import nanovllm.envs as envs
from nanovllm.logger import init_logger

logger = init_logger(__name__)


def get_ip() -> str:
    """优先返回显式主机地址，否则依次通过 IPv4、IPv6 路由探测本机地址。

    Returns:
        VLLM_HOST_IP 的非空值，或系统为探测目标选择的本机源地址。两种探测
        均失败时沿用上游返回 0.0.0.0，并发出警告；单机可设置环境变量为
        127.0.0.1，使初始化明确使用回环地址。

    Side Effects:
        自动探测时短暂创建 UDP socket，通过 connect() 选择路由，不发送应用
        数据，也不要求目标服务器响应。socket 在每次尝试结束时关闭。
    """
    # host_ip 是调用时读取的显式覆盖值；非空时直接返回，不进行自动探测。
    host_ip = envs.VLLM_HOST_IP
    if host_ip:
        return host_ip

    # 先尝试 IPv4；UDP connect 让内核选择出口与源地址，不建立 TCP 连接。
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        # 当前网络可能没有 IPv4 出口，继续尝试上游的 IPv6 路径。
        pass

    # IPv6 探测只在 IPv4 失败后进行；同样不发送应用数据。
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_DGRAM) as s:
            s.connect(("2001:4860:4860::8888", 80))
            return s.getsockname()[0]
    except Exception:
        # 两个地址族都不可用时走统一告警分支，保留上游的兜底行为。
        pass

    warnings.warn(
        "Failed to get the IP address, using 0.0.0.0 by default. "
        "The value can be set by the environment variable VLLM_HOST_IP.",
        stacklevel=2,
    )
    return "0.0.0.0"


def is_valid_ipv6_address(address: str) -> bool:
    """判断 address 是否为合法 IPv6 字面量，供 TCP URI 选择方括号格式。"""
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def get_distributed_init_method(ip: str, port: int) -> str:
    """将主机地址和端口组成供 PyTorch process group 使用的 TCP 会合 URI。"""
    return get_tcp_uri(ip, port)


def get_tcp_uri(ip: str, port: int) -> str:
    """格式化 TCP URI；IPv6 地址用方括号与端口分隔，不绑定或保留该端口。

    例如 IPv4 返回 tcp://127.0.0.1:12345，IPv6 返回 tcp://[::1]:12345。
    """
    # IPv6 地址自身包含冒号，需要方括号明确主机部分的边界。
    if is_valid_ipv6_address(ip):
        return f"tcp://[{ip}]:{port}"
    else:
        return f"tcp://{ip}:{port}"


def get_open_port() -> int:
    """查询一个当前可绑定的通信端口，沿用 vLLM 的公开入口。

    当前没有 DP master，因此直接委托 _get_open_port()，省略上游 DP 预留
    端口范围的过滤。调用结束即关闭探测 socket，不持有端口租约；实际创建
    PyTorch store 前，该端口仍可能被其他进程占用。
    """
    return _get_open_port()


def _get_open_port(
    start_port: int | None = None,
    max_attempts: int | None = None,
) -> int:
    """从显式起点向上寻找端口，或通过 port=0 让操作系统分配空闲端口。

    Args:
        start_port: 优先使用的搜索起点；为 None 时读取 envs.VLLM_PORT。
            二者都为 None 时使用系统分配，不从某个固定端口开始扫描。
        max_attempts: 指定起点时的最大尝试次数；None 沿用上游的不限制策略。

    Returns:
        探测成功的整数端口。指定起点使用 IPv4；系统分配路径先尝试 IPv4，
        再尝试 IPv6，不要求调用方提供有效的外网地址。

    Raises:
        RuntimeError: 指定起点的尝试次数达到 max_attempts，仍未找到端口。
        OSError: 系统分配路径的 IPv4 和 IPv6 绑定均失败。
        ValueError: 环境变量 VLLM_PORT 无法解析成整数。

    Side Effects:
        临时创建并绑定 TCP socket，离开每个 with 时关闭；不初始化进程组。
    """
    # start_port 优先于环境配置；port 是当前候选值，在绑定失败时递增。
    start_port = start_port if start_port is not None else envs.VLLM_PORT
    port = start_port
    # 有搜索起点时依次尝试相邻端口，保留上游变量命名与更新顺序。
    if port is not None:
        attempts = 0  # 已结束的绑定尝试数，仅用于检查调用方提供的次数上限。
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", port))
                    return port
            except OSError:
                # 当前候选无法绑定时尝试下一个端口；日志记录旧值与新值。
                port += 1
                logger.info("Port %d is already in use, trying port %d", port - 1, port)
            attempts += 1
            # 达到调用方指定的上限后明确失败，不继续无界搜索。
            if max_attempts is not None and attempts >= max_attempts:
                raise RuntimeError(
                    f"Could not find open port after {max_attempts} "
                    f"attempts starting from port {start_port}"
                )

    # 未指定起点时让系统选择端口；必须读取实际绑定结果，而非返回传入的零。
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        # IPv4 不可用时沿用上游的 IPv6 回退；第二次失败由调用方处理。
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
