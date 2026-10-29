# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor Parallel 进程组的全局状态与初始化逻辑。

本模块遵循 vLLM ``distributed.parallel_state`` 的职责划分：

1. ``init_distributed_environment`` 初始化 PyTorch 默认 process group，并建立
   描述整个 world 的 ``_WORLD``；
2. ``initialize_model_parallel`` 从 world 中划分模型并行子组；
3. ``ensure_model_parallel_initialized`` 负责“首次创建或重复校验”的公共入口。
4. ``graph_capture`` 准备独立 CUDA stream，供模型预热和 Graph 捕获共享。
5. ``cleanup_dist_env_and_memory`` 在执行端关闭后销毁并行组，回收空闲设备缓存。

每个 GroupCoordinator 管理设备组和 Gloo CPU 协调组；多 rank TP 组另外安装
DeviceCommunicatorBase，通过 PyTorch/NCCL 执行设备集合通信。

当前 nano-vLLM 只支持 Tensor Parallelism，因此只维护 ``_WORLD`` 与 ``_TP``。
Pipeline / Data / Context / Expert Parallelism 的 group 并非在这里被简化成默认 group，
而是暂不实现；未来可沿用 vLLM 的全局状态与初始化顺序继续扩展。
"""

from __future__ import annotations

import contextlib
import gc
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing import shared_memory
from typing import Any
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from nanovllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from nanovllm.distributed.utils import (
    StatelessProcessGroup,
)
from nanovllm.logger import init_logger


logger = init_logger(__name__)


@dataclass
class GraphCaptureContext:
    """保存一次捕获环境使用的 CUDA stream，供各逻辑组共享。

    与上游一样只保存 stream，不持有 CUDA Graph，不保存 batch 描述符或图内存池。
    构造对象本身不会切换 stream，也不会开始记录 GPU 工作。

    Attributes:
        stream: 预热与实际捕获使用的 stream；调用方应在整个作用域内复用它。
    """

    stream: torch.cuda.Stream  # 保存原始 stream 引用，避免各组各自创建捕获 stream。


_group_name_counter: dict[str, int] = {}  # 同名逻辑组重建时递增，避免复用日志身份。


def _get_unique_name(name: str) -> str:
    """沿用 vLLM 的 name:index 格式生成当前进程内的组名。

    Returns:
        unique_name: 例如 ``tp:0``；各 rank 按相同顺序建组时名称一致。

    Side Effects:
        递增同名组计数；由初始化线程按照所有 rank 一致的顺序调用。
    """
    index = _group_name_counter.get(name, 0)
    _group_name_counter[name] = index + 1
    return f"{name}:{index}"


class GroupCoordinator:
    """管理逻辑组的成员身份、设备通信和 CPU 协调。

    沿用 vLLM 的双组设计：device_group 承载 tensor 通信，Gloo cpu_group
    承载对象广播和 barrier。CPU group 服务于 CUDA Worker 之间的协调，
    不表示支持 CPU 模型执行。WORLD 只创建进程组，TP 额外创建设备通信器。

    当前设备通信器使用 PyTorch/NCCL；未接入自定义通信算子、组内 MQ 广播、
    PP 点对点传输、EP dispatch/combine 或弹性进程组。

    Attributes:
        unique_name: 当前进程内唯一的逻辑组名，例如 ``tp:0``。
        rank: 默认 world 中的全局 rank。
        ranks: 本逻辑组的全局 rank 列表；src/dst 参数按其下标解释。
        world_size: 本逻辑组的 rank 数量。
        local_rank: 本机可见 CUDA device 编号，与组内 rank 是不同概念。
        device_index: 对齐 vLLM 的设备编号字段，当前等于 local_rank。
        device: 当前 Worker 的 CUDA 设备描述；构造描述本身不初始化 CUDA。
        rank_in_group: 当前进程在 ranks 中的位置。
        group_ranks: 同一种并行维度的全部分组，所有 rank 按此顺序创建子组。
        device_group: 使用指定设备 backend 的 PyTorch process group。
        cpu_group: 与 device_group 成员相同的 Gloo process group。
        use_device_communicator: 是否为多 rank 组创建设备通信器。
        device_communicator: 多 rank TP 的通信对象；单 rank 或禁用时为 None。
        torch_distributed_backend: device_group 使用的 backend 名称。
    """

    def __init__(
        self,
        group_ranks: list[list[int]],
        local_rank: int,
        torch_distributed_backend: str,
        use_device_communicator: bool,
        group_name: str | None = None,
    ) -> None:
        """按统一顺序创建双进程组，并为当前 rank 安装通信器。

        所有 world rank 都必须调用本构造函数，并按 group_ranks 中的顺序
        创建 device/Gloo 子组；不属于某一子组也不能跳过其 new_group()。

        Args:
            group_ranks: 当前并行维度的全部 rank 分组。
            local_rank: 当前进程绑定的本机可见 CUDA device 编号。
            torch_distributed_backend: 设备通信 backend，正常 CUDA 路径为 nccl。
            use_device_communicator: 多 rank 组是否额外创建设备通信执行对象。
            group_name: 逻辑组名称，None 时使用 anonymous。

        Side Effects:
            创建两套 PyTorch 子组并保存当前 rank 所属组；启用时创建通信器。
            不加载模型，不创建 Executor 的 RPC 队列，也不切换当前 CUDA device。

        Raises:
            RuntimeError: 默认进程组未初始化，或当前 rank 没有对应子组。
            AssertionError: local_rank 没有提供。
        """
        if not dist.is_initialized():
            raise RuntimeError("The default process group must be initialized first")
        assert local_rank >= 0, "local_rank must be provided when creating a group"
        self.unique_name = _get_unique_name(group_name or "anonymous")
        self.rank = dist.get_rank()  # global rank 不随当前 TP 子组变化。
        self.local_rank = local_rank
        self.device_index = local_rank  # 当前仅 CUDA，无需其他平台的设备映射。
        self.device = torch.device("cuda", self.device_index)
        self.group_ranks = group_ranks
        self.torch_distributed_backend = torch_distributed_backend

        self_device_group: ProcessGroup | None = None
        self_cpu_group: ProcessGroup | None = None
        # 各 rank 必须按相同顺序创建所有子组，再选中自己所属的那一对。
        for ranks in group_ranks:
            device_group = dist.new_group(ranks, backend=torch_distributed_backend)
            cpu_group = dist.new_group(ranks, backend="gloo")
            if self.rank in ranks:
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self_device_group = device_group
                self_cpu_group = cpu_group

        if self_device_group is None or self_cpu_group is None:
            raise RuntimeError(
                f"Global rank {self.rank} does not belong to any group: {group_ranks}"
            )
        self.device_group = self_device_group  # tensor 通信使用设备 backend。
        self.cpu_group = self_cpu_group  # 控制操作使用 Gloo，避免隐式 CUDA tensor。

        self.use_device_communicator = use_device_communicator
        self.device_communicator: DeviceCommunicatorBase | None = None
        # 单 rank 通信由公开接口直接返回；WORLD 不额外安装设备通信器。
        if use_device_communicator and self.world_size > 1:
            self.device_communicator = DeviceCommunicatorBase(
                cpu_group=self.cpu_group,
                device=self.device,
                device_group=self.device_group,
                unique_name=self.unique_name,
            )

    @property
    def first_rank(self) -> int:
        """返回组内首个进程的全局 rank，即 first_rank。"""
        return self.ranks[0]

    @property
    def last_rank(self) -> int:
        """返回组内最后一个进程的全局 rank，即 last_rank。"""
        return self.ranks[-1]

    @property
    def is_first_rank(self) -> bool:
        """返回当前进程是否为组内首个进程，即 is_first_rank。"""
        return self.rank == self.first_rank

    @property
    def is_last_rank(self) -> bool:
        """返回当前进程是否为组内最后一个进程，即 is_last_rank。"""
        return self.rank == self.last_rank

    @property
    def next_rank(self) -> int:
        """返回组内环形顺序的下一个全局 rank，即 next_rank。"""
        return self.ranks[(self.rank_in_group + 1) % self.world_size]

    @property
    def prev_rank(self) -> int:
        """返回组内环形顺序的上一个全局 rank，即 prev_rank。"""
        return self.ranks[(self.rank_in_group - 1) % self.world_size]

    @contextmanager
    def graph_capture(
        self,
        graph_capture_context: GraphCaptureContext | None = None,
    ):
        """建立通信与模型共享的 CUDA stream 环境，供外层预热或捕获。

        Args:
            graph_capture_context: 要复用的捕获上下文；None 时创建新 stream。

        Yields:
            graph_capture_context: 本作用域实际使用的上下文，保留传入对象身份。

        Side Effects:
            捕获 stream 等待进入前的当前 stream；退出时恢复原 stream。
            不启动 CUDA Graph 捕获，不调用 CPU barrier 或设备级 synchronize。
            当前通信器直接使用 PyTorch/NCCL，无自定义 all-reduce capture 上下文。
        """
        if graph_capture_context is None:
            stream = torch.cuda.Stream(device=self.device)
            graph_capture_context = GraphCaptureContext(stream)
        else:
            stream = graph_capture_context.stream

        curr_stream = torch.cuda.current_stream()
        if curr_stream != stream:
            stream.wait_stream(curr_stream)  # 只建立 GPU 依赖，不等待 CPU。
        with torch.cuda.stream(stream):
            yield graph_capture_context

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """在组内执行 SUM，返回归约后的 output。

        当前基础通信器原地更新 input_，返回值与输入共享存储。单 rank 直接返回
        input_；多 rank 必须以相同顺序参与，不能仅由结果回包 rank 调用。
        """
        if self.world_size == 1:
            return input_
        return self._all_reduce_out_place(input_)

    def _all_reduce_out_place(self, input_: torch.Tensor) -> torch.Tensor:
        """按 vLLM 的内部入口转发，返回通信器生成的 output。

        当前基类通信器采用原地归约；方法名对齐上游入口，不承诺独立存储。
        """
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        return self.device_communicator.all_reduce(input_)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """按组内 rank 顺序沿 dim 拼接输入，返回各 rank 都持有的 output。

        各 rank 的输入形状、dtype 必须一致；单 rank 返回 input_ 本身。
        多 rank 创建拼接结果，不修改输入，dim 支持负数。
        """
        if self.world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )
        return self._all_gather_out_place(input_, dim)

    def _all_gather_out_place(self, input_: torch.Tensor, dim: int) -> torch.Tensor:
        """转交设备通信器执行，返回沿 dim 拼接后的 output。"""
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        return self.device_communicator.all_gather(input_, dim)

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        """将各 rank 输入沿 dim 拼接，仅在目标进程返回 output。

        Args:
            input_: 本 rank 的 tensor，各 rank 必须使用相同形状和 dtype。
            dst: 组内目标 rank，通过 ranks[dst] 映射成全局 rank。
            dim: 拼接维度，支持负数。

        Returns:
            output: dst 上为完整 tensor，其他 rank 为 None；单 rank 返回 input_。

        Side Effects:
            多 rank 参与设备 gather，目标 rank 分配接收 tensor。
        """
        assert 0 <= dst < self.world_size, f"Invalid dst rank ({dst})"
        if self.world_size == 1:
            return input_
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        return self.device_communicator.gather(input_, dst, dim)

    def broadcast(self, input_: torch.Tensor, src: int = 0) -> torch.Tensor:
        """从组内 src 广播 tensor，原地更新并返回 input_。

        src 是组内编号；所有 rank 都提供形状和 dtype 一致的接收 tensor。
        tensor 通信直接使用 device_group，与上游相同，不要求设备通信器存在。
        """
        assert 0 <= src < self.world_size, f"Invalid src rank ({src})"
        if self.world_size == 1:
            return input_
        dist.broadcast(input_, src=self.ranks[src], group=self.device_group)
        return input_

    def broadcast_object(self, obj: Any | None = None, src: int = 0) -> Any:
        """通过 CPU group 广播 Python 对象，返回源 rank 提供的 obj。

        obj 只有在组内 src 上有意义；其他 rank 返回反序列化后的对象。
        此接口不使用 Executor 的 RPC MQ，所有组成员必须共同调用。
        """
        assert 0 <= src < self.world_size, f"Invalid src rank ({src})"
        if self.world_size == 1:
            return obj
        objects = [obj if self.rank_in_group == src else None]
        dist.broadcast_object_list(objects, src=self.ranks[src], group=self.cpu_group)
        return objects[0]

    def broadcast_object_list(
        self,
        obj_list: list[Any],
        src: int = 0,
        group: ProcessGroup | None = None,
    ) -> list[Any]:
        """广播并原地更新对象列表，返回 obj_list。

        src 是本逻辑组内编号；各 rank 提供相同长度的列表。默认沿用 vLLM
        的 device_group 路径；显式 group 必须具有相同的成员和 rank 顺序。
        """
        assert 0 <= src < self.world_size, f"Invalid src rank ({src})"
        if self.world_size == 1:
            return obj_list
        dist.broadcast_object_list(
            obj_list,
            src=self.ranks[src],
            group=self.device_group if group is None else group,
        )
        return obj_list

    def barrier(self) -> None:
        """通过 CPU group 等待全部组成员，避免 NCCL barrier 隐式创建 GPU tensor。"""
        dist.barrier(group=self.cpu_group)

    def cpu_all_reduce_(self, tensor: torch.Tensor) -> torch.Tensor:
        """通过 Gloo 原地 SUM 小型 CPU tensor，返回归约后的 tensor。

        此接口用于控制信息；不会把输入搬到 GPU，也不参与模型层设备计算。
        """
        dist.all_reduce(tensor, group=self.cpu_group)
        return tensor

    def destroy(self) -> None:
        """先释放通信器资源，再销毁设备组和 CPU 组。

        上层须先停止所有使用本组的执行。删除已销毁的属性，重复调用不会
        重复销毁进程组；默认 world process group 由外层单独销毁。
        """
        if self.device_communicator is not None:
            self.device_communicator.destroy()
            self.device_communicator = None
        if hasattr(self, "device_group"):
            dist.destroy_process_group(self.device_group)
            del self.device_group
        if hasattr(self, "cpu_group"):
            dist.destroy_process_group(self.cpu_group)
            del self.cpu_group


# ``_WORLD`` 描述默认全局进程组，``_TP`` 描述模型实际使用的 Tensor Parallel 子组。
_WORLD: GroupCoordinator | None = None
_TP: GroupCoordinator | None = None


def get_world_group() -> GroupCoordinator:
    """返回已初始化的全局 world group 描述。

    Returns:
        world_group: 封装当前默认 PyTorch process group 的 ``GroupCoordinator``。

    Raises:
        AssertionError: ``init_distributed_environment`` 尚未成功执行时抛出。
    """
    assert _WORLD is not None, "world group is not initialized"
    return _WORLD


def get_tp_group() -> GroupCoordinator:
    """返回已初始化的 Tensor Parallel group 描述。

    Returns:
        tp_group: 当前 rank 所属的 TP ``GroupCoordinator``。

    Raises:
        AssertionError: ``ensure_model_parallel_initialized`` 尚未成功执行时抛出。
    """
    assert _TP is not None, "tensor model parallel group is not initialized"
    return _TP


@contextmanager
def graph_capture(device: torch.device):
    """为指定 CUDA device 创建独立 stream，并进入 TP 组的捕获环境。

    对齐 vLLM 的模块级入口，当前仅保留已支持的 TP group，不进入 PP group。
    Runner 用该环境包围预热和捕获；真正记录图由内部 Graph Wrapper 完成。

    Args:
        device: 当前 Worker 的 CUDA device，例如 torch.device("cuda:0")。
            应已完成设备绑定和 TP group 初始化。

    Yields:
        context: 含指定 device 上新建 stream 的上下文；模型调用可使用其 stream 捕获图。

    Side Effects:
        创建 CUDA stream，通过组接口建立进入前的 stream 依赖并临时切换 stream。
        退出时恢复原 stream，不销毁通信组，不设置捕获许可，也不创建图对象。

    Raises:
        AssertionError: Tensor Parallel group 尚未初始化。
    """
    context = GraphCaptureContext(torch.cuda.Stream(device=device))  # stream 绑定到 Worker。
    with get_tp_group().graph_capture(context):  # 当前仅 TP，复用上下文而非再次创建 stream。
        yield context


def init_world_group(
    ranks: list[int],
    local_rank: int,
    backend: str,
) -> GroupCoordinator:
    """创建描述默认 world 的 ``GroupCoordinator``。

    Args:
        ranks: 默认 world 内全部 global rank，通常为 ``range(world_size)``。
        local_rank: 当前进程的本机 device 编号。
        backend: world 对应的通信 backend。

    Returns:
        world_group: 当前 rank 视角下的全局 group 描述对象。
    """
    return GroupCoordinator(
        group_ranks=[ranks],
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_device_communicator=False,
        group_name="world",
    )


def init_model_parallel_group(
    group_ranks: list[list[int]],
    local_rank: int,
    backend: str,
    group_name: str | None = None,
    use_device_communicator: bool = True,
) -> GroupCoordinator:
    """创建一种模型并行维度的全部子组。

    Args:
        group_ranks: 该并行维度的全部 rank 分组。
        local_rank: 当前进程的本机 device 编号。
        backend: 创建子组使用的通信 backend。
        group_name: 逻辑组名称，例如 ``"tp"``。
        use_device_communicator: 是否为多 rank 组创建设备通信器，默认启用。

    Returns:
        group: 当前 rank 所属的逻辑并行组描述。
    """
    return GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_device_communicator=use_device_communicator,
        group_name=group_name,
    )


def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    distributed_init_method: str = "env://",
    local_rank: int = -1,
    backend: str = "nccl",
    timeout: timedelta | None = None,
) -> None:
    """初始化默认 PyTorch process group，并建立 nano-vLLM 的 world group。

    此函数保留 vLLM 同名函数的职责和参数顺序。当前移除了与尚未实现 feature 相关的
    DP rank 重映射、Elastic EP、split group 与内部 DP world；默认 process group
    的 ``world_size`` 和 ``rank`` 因而直接由 Executor 或外部 launcher 传入。

    Args:
        world_size: 默认分布式 world 的进程数量。传入 ``-1`` 时由 PyTorch 在
            ``env://`` 初始化路径中读取 ``WORLD_SIZE``。
        rank: 当前进程在默认 world 中的全局 rank。传入 ``-1`` 时由 PyTorch 在
            ``env://`` 初始化路径中读取 ``RANK``。
        distributed_init_method: rendezvous 地址，例如 ``tcp://host:port`` 或
            ``env://``。
        local_rank: 当前进程绑定的本机 CUDA device 编号。未提供时，``env://``
            从 ``LOCAL_RANK`` 读取，其余情况退化为 global rank。
        backend: PyTorch distributed backend，GPU 路径通常为 ``"nccl"``。
        timeout: 可选的 process group 初始化超时。

    Raises:
        RuntimeError: backend 不可用且无法退化为 Gloo，或已有 world group 与当前
            默认 process group 不一致时抛出。
    """
    logger.debug(
        "world_size=%d rank=%d local_rank=%d distributed_init_method=%s "
        "backend=%s",
        world_size,
        rank,
        local_rank,
        distributed_init_method,
        backend,
    )

    if not dist.is_initialized():
        logger.info(
            "world_size=%d rank=%d local_rank=%d distributed_init_method=%s "
            "backend=%s",
            world_size,
            rank,
            local_rank,
            distributed_init_method,
            backend,
        )
        if not dist.is_backend_available(backend):
            logger.warning(
                "Distributed backend %s is not available; falling back to gloo.",
                backend,
            )
            if not dist.is_gloo_available():
                raise RuntimeError("Fallback Gloo backend is not available.")
            backend = "gloo"

        # 默认 group 是进程组建立的根；TP/PP 等逻辑子组随后从它派生。保留 ``-1``
        # 参数可让 PyTorch 在 ``env://`` 路径中读取 RANK/WORLD_SIZE，和 vLLM 一致。
        dist.init_process_group(
            backend=backend,
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank,
            timeout=timeout,
        )

    # PyTorch ProcessGroup 不保存 local rank，因此 nano-vLLM 像 vLLM 一样单独维护它。
    if local_rank == -1:
        if distributed_init_method == "env://":
            local_rank = int(os.environ["LOCAL_RANK"])
        else:
            local_rank = rank

    global _WORLD
    if _WORLD is None:
        ranks = list(range(dist.get_world_size()))
        _WORLD = init_world_group(ranks, local_rank, backend)
    else:
        if _WORLD.world_size != dist.get_world_size():
            raise RuntimeError(
                "world group already initialized with a different world size: "
                f"existing={_WORLD.world_size}, "
                f"current={dist.get_world_size()}."
            )
        if _WORLD.rank != dist.get_rank():
            raise RuntimeError(
                "world group already initialized for a different global rank: "
                f"existing={_WORLD.rank}, current={dist.get_rank()}."
            )


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    backend: str | None = None,
) -> None:
    """根据默认 world 划分 Tensor Parallel 子组。

    该函数是 vLLM ``initialize_model_parallel`` 的 TP 子集。它保留“从全局 world
    生成全部分组，再创建当前 rank 所属 group”的方式，而不是把默认 group 偷换成 TP
    group。这样在未来加入 PP/DP 后，TP group 的创建位置和 rank 布局无需重写。

    例如 world 为 ``[0, 1, 2, 3]``、TP size 为 ``2`` 时，将建立两个 TP group：
    ``[0, 1]`` 和 ``[2, 3]``。

    Args:
        tensor_model_parallel_size: 每个 TP group 内的 rank 数量。
        backend: 创建 TP group 的 backend。为 ``None`` 时复用默认 process group
            的 backend。

    Raises:
        AssertionError: 默认 world 未初始化，或 TP group 已被创建时抛出。
        ValueError: TP size 不能整除全局 world size 时抛出。
    """
    assert dist.is_initialized(), (
        "The default process group must be initialized before model parallel "
        "groups are created."
    )
    if tensor_model_parallel_size <= 0:
        raise ValueError(
            "tensor_model_parallel_size must be positive, got "
            f"{tensor_model_parallel_size}."
        )

    world_group = get_world_group()
    world_size = world_group.world_size
    # 每个 TP 组必须包含相同数量的 rank，不能留下无法成组的尾部进程。
    if world_size % tensor_model_parallel_size != 0:
        raise ValueError(
            "tensor_model_parallel_size must divide world size, got "
            f"tensor_model_parallel_size={tensor_model_parallel_size}, "
            f"world_size={world_size}."
        )
    # 未指定通信后端时沿用已初始化的默认进程组，保持所有 rank 的选择一致。
    if backend is None:
        backend = dist.get_backend()

    # vLLM 的 rank 布局让相邻 rank 落入同一个 TP group；PP/DP 加入后会在此基础上
    # 对完整 rank 张量做 reshape，而不是改变这里创建 group 的生命周期。
    all_ranks = list(range(world_size))
    group_ranks = [
        all_ranks[start_rank : start_rank + tensor_model_parallel_size]
        for start_rank in range(0, world_size, tensor_model_parallel_size)
    ]

    global _TP
    # 组对象只有一个所有者，重复初始化会丢失旧通信资源，必须先显式销毁。
    assert _TP is None, "tensor model parallel group is already initialized"
    # 每个进程按相同 group_ranks 顺序建组，再由组协调器选出当前 rank 所属组。
    _TP = init_model_parallel_group(
        group_ranks=group_ranks,
        local_rank=world_group.local_rank,
        backend=backend,
        group_name="tp",
    )


def model_parallel_is_initialized() -> bool:
    """检查 Tensor Parallel group 是否已初始化。

    vLLM 还要求 PP group 同时存在；nano-vLLM 当前尚未实现 PP，因此 TP group 存在
    即表示当前支持范围内的模型并行状态已就绪。

    Returns:
        is_initialized: 当前 TP group 已初始化时为 True；当前项目未实现 PP。
    """
    return _TP is not None


def get_tensor_model_parallel_world_size() -> int:
    """返回当前 rank 所属 Tensor Parallel group 的大小。

    Returns:
        tp_size: 当前 rank 所属 TP group 的 world_size。
    """
    return get_tp_group().world_size


def get_tensor_model_parallel_rank() -> int:
    """返回本进程在 TP 组中的编号，即 tp_rank。

    权重分片必须使用组内编号；默认 world 的 global rank 可能属于其他 TP 子组。
    """
    return get_tp_group().rank_in_group


def ensure_model_parallel_initialized(
    tensor_model_parallel_size: int,
    backend: str | None = None,
) -> None:
    """首次初始化或校验 Tensor Parallel group 与期望配置一致。

    这是 vLLM ``ensure_model_parallel_initialized`` 在仅保留 TP feature 后的版本。
    首次调用创建 TP group；后续调用不重复创建 NCCL communicator，而是验证既有 group
    的 size，防止不同配置意外复用同一进程内的全局分布式状态。

    Args:
        tensor_model_parallel_size: 期望的 TP group 大小。
        backend: 首次创建 TP group 时使用的 backend。为 ``None`` 时复用默认 group。

    Raises:
        AssertionError: 已初始化 TP group 的大小与调用方期望不一致时抛出。
    """
    if not model_parallel_is_initialized():
        initialize_model_parallel(tensor_model_parallel_size, backend)
        return

    actual_tp_world_size = get_tensor_model_parallel_world_size()
    assert actual_tp_world_size == tensor_model_parallel_size, (
        "tensor parallel group already initialized, but of unexpected size. "
        f"got: {actual_tp_world_size=} vs. "
        f"wanted: {tensor_model_parallel_size=}"
    )


def destroy_model_parallel() -> None:
    """销毁当前支持的 TP group，并恢复模型并行未初始化状态。

    Engine 关闭执行端后调用。沿用 vLLM 的全局状态清理方式，当前只存在 _TP；
    将其重置为 None 后，后续 Engine 可以重新创建模型并行组。
    """
    global _TP

    # 仅销毁已经创建的 TP 子组；默认 process group 由后续环境清理负责。
    if _TP is not None:
        _TP.destroy()
    _TP = None


def destroy_distributed_environment() -> None:
    """先销毁 WORLD 逻辑子组，再销毁 PyTorch 默认 process group。

    调用方应先销毁模型并行子组。WORLD 是单独创建的逻辑组，不能用一次默认组
    destroy 替代它的关闭；_WORLD 重置为 None 后，允许后续重新初始化环境。
    """
    global _WORLD

    # WORLD 与 TP 一样由 dist.new_group() 创建，持有独立的通信组资源。
    if _WORLD is not None:
        _WORLD.destroy()
    _WORLD = None
    # 重复关闭或从未初始化分布式环境时，不向 PyTorch 提交无效的默认组销毁。
    if dist.is_initialized():
        dist.destroy_process_group()


def cleanup_dist_env_and_memory() -> None:
    """在执行端关闭后，重置分布式环境并回收空闲 CUDA allocator 缓存。

    对齐 vLLM 同名接口的当前 CUDA / TP 子集。按 GC 解冻、模型并行组、WORLD 与
    默认组、GC 回收、设备缓存的顺序执行；没有 Ray、环境变量缓存或其他设备分支。

    Side Effects:
        清空进程级 _TP / _WORLD 状态，解除 Python GC 冻结，回收不可达对象并释放
        CUDA allocator 的空闲缓存。PyTorch 暴露 pinned-host 清理接口时也清理
        其空闲缓存；不会释放仍有引用的 tensor，也不初始化尚未使用的 CUDA。
    """
    logger.debug("[shutdown] Distributed: cleanup start")
    # 预热冻结的是进程级 Python GC 对象，先恢复扫描范围再回收不再被引用的资源。
    gc.unfreeze()
    # 先销毁依赖默认组的 TP 子组，再销毁 WORLD 和默认组，避免残留旧通信句柄。
    destroy_model_parallel()
    destroy_distributed_environment()
    # 先释放不可达对象对存储的引用，再归还 allocator 中已经空闲的显存缓存。
    gc.collect()
    # 当前仅支持 CUDA；empty_cache() 在 CUDA 尚未初始化时直接返回，不触发设备初始化。
    torch.cuda.empty_cache()
    # 项目支持的 PyTorch 版本不一定暴露该私有接口，存在时才回收空闲 pinned-host 缓存。
    host_empty_cache = getattr(torch._C, "_host_emptyCache", None)
    if host_empty_cache is not None:
        host_empty_cache()
    logger.debug("[shutdown] Distributed: cleanup complete")



def in_the_same_node_as(
    pg: ProcessGroup | StatelessProcessGroup, source_rank: int = 0
) -> list[bool]:
    """This is a collective operation that returns if each rank is in the same node
    as the source rank. It tests if processes are attached to the same
    memory system (shared access to shared memory).
    """
    if isinstance(pg, ProcessGroup):
        assert torch.distributed.get_backend(pg) != torch.distributed.Backend.NCCL, (
            "in_the_same_node_as should be tested with a non-NCCL group."
        )
        # local rank inside the group
        rank = torch.distributed.get_rank(group=pg)
        world_size = torch.distributed.get_world_size(group=pg)

        # global ranks of the processes in the group
        ranks = torch.distributed.get_process_group_ranks(pg)
    else:
        rank = pg.rank
        world_size = pg.world_size
        ranks = list(range(world_size))

    # local tensor in each process to store the result
    is_in_the_same_node = torch.tensor(
        [0] * world_size, dtype=torch.int32, device="cpu"
    )

    magic_message = b"magic_message"
    shm = None

    try:
        with contextlib.suppress(OSError):
            if rank == source_rank:
                # create a shared memory segment
                shm = shared_memory.SharedMemory(create=True, size=128)
                assert shm.buf is not None, "Buffer was not created"
                shm.buf[: len(magic_message)] = magic_message
                if isinstance(pg, ProcessGroup):
                    torch.distributed.broadcast_object_list(
                        [shm.name], src=ranks[source_rank], group=pg
                    )
                else:
                    pg.broadcast_obj(shm.name, src=source_rank)
                is_in_the_same_node[rank] = 1
            else:
                # try to open the shared memory segment
                if isinstance(pg, ProcessGroup):
                    recv = [None]
                    torch.distributed.broadcast_object_list(
                        recv, src=ranks[source_rank], group=pg
                    )
                    name = recv[0]
                else:
                    name = pg.broadcast_obj(None, src=source_rank)
                # fix to https://stackoverflow.com/q/62748654/9191338
                # Python incorrectly tracks shared memory even if it is not
                # created by the process. The following patch is a workaround.
                with patch(
                    "multiprocessing.resource_tracker.register",
                    lambda *args, **kwargs: None,
                ):
                    shm = shared_memory.SharedMemory(name=name)
                assert shm.buf is not None, "Buffer was not opened"
                if shm.buf[: len(magic_message)] == magic_message:
                    is_in_the_same_node[rank] = 1
    except Exception as e:
        logger.error("Error ignored in is_in_the_same_node: %s", e)
    finally:
        if shm:
            shm.close()

    if isinstance(pg, ProcessGroup):
        torch.distributed.barrier(group=pg)
    else:
        pg.barrier()

    # clean up the shared memory segment
    with contextlib.suppress(OSError):
        if rank == source_rank and shm:
            shm.unlink()

    if isinstance(pg, ProcessGroup):
        torch.distributed.all_reduce(is_in_the_same_node, group=pg)
        aggregated_data = is_in_the_same_node
    else:
        aggregated_data = torch.zeros_like(is_in_the_same_node)
        for i in range(world_size):
            rank_data = pg.broadcast_obj(is_in_the_same_node, src=i)
            aggregated_data += rank_data

    return [x == 1 for x in aggregated_data.tolist()]
