"""Tensor Parallel 进程组的全局状态与初始化逻辑。

本模块遵循 vLLM ``distributed.parallel_state`` 的职责划分：

1. ``init_distributed_environment`` 初始化 PyTorch 默认 process group，并建立
   描述整个 world 的 ``_WORLD``；
2. ``initialize_model_parallel`` 从 world 中划分模型并行子组；
3. ``ensure_model_parallel_initialized`` 负责“首次创建或重复校验”的公共入口。
4. ``graph_capture`` 准备独立 CUDA stream，供模型预热和 Graph 捕获共享。
5. ``cleanup_dist_env_and_memory`` 在执行端关闭后销毁并行组，回收空闲设备缓存。

当前 nano-vLLM 只支持 Tensor Parallelism，因此只维护 ``_WORLD`` 与 ``_TP``。
Pipeline / Data / Context / Expert Parallelism 的 group 并非在这里被简化成默认 group，
而是暂不实现；未来可沿用 vLLM 的全局状态与初始化顺序继续扩展。
"""

from __future__ import annotations

import gc
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

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


class GroupCoordinator:
    """封装一个逻辑并行组在当前 rank 上对应的 PyTorch process group。

    vLLM 的完整 ``GroupCoordinator`` 还管理 CPU group、device communicator、共享内存
    broadcaster 等优化组件。nano-vLLM 的当前层实现只需要知道 TP group 的成员关系和
    device-side process group，因此保留其核心状态，删除尚未使用的通信优化设施。

    Attributes:
        rank: 当前进程在默认 PyTorch world 中的全局 rank。
        local_rank: 当前进程在本机可见加速器中的编号，用于将来创建 device communicator。
        group_ranks: 全部同类逻辑组的全局 rank 列表。例如 8 卡 world、TP=2 时为
            ``[[0, 1], [2, 3], [4, 5], [6, 7]]``。
        ranks: 当前 rank 实际所属逻辑组的全局 rank 列表。
        world_size: 当前逻辑组内的 rank 数量。
        rank_in_group: 当前 rank 在 ``ranks`` 内的相对编号。
        device_group: 当前逻辑组对应的 PyTorch device-side process group。
        torch_distributed_backend: 创建 ``device_group`` 时使用的 backend 名称。
    """

    def __init__(
        self,
        group_ranks: list[list[int]],
        local_rank: int,
        torch_distributed_backend: str,
        group_name: str | None = None,
    ) -> None:
        """为当前 rank 创建并选出所属的逻辑进程组。

        所有 rank 必须以相同顺序调用 ``dist.new_group``。即使某个 rank 不属于某个
        group，它也仍要参与该 ``new_group`` 调用，否则多进程初始化会死锁；这正是
        ``group_ranks`` 保留“全部组”而不是只传当前组的原因。

        Args:
            group_ranks: 按创建顺序排列的全部逻辑组成员列表。
            local_rank: 当前进程的本机 device 编号。
            torch_distributed_backend: 创建子组使用的 backend，例如 ``"nccl"``。
            group_name: 仅用于异常信息的逻辑组名称。

        Raises:
            RuntimeError: 默认 process group 未初始化，或当前 rank 不属于任何给定
                逻辑组时抛出。
        """
        if not dist.is_initialized():
            raise RuntimeError(
                "The default process group must be initialized before "
                "creating a GroupCoordinator."
            )

        # rank 是默认 world 中的身份；每个逻辑 group 再提供 rank_in_group。
        self.rank = dist.get_rank()
        self.local_rank = local_rank
        self.group_ranks = group_ranks
        self.torch_distributed_backend = torch_distributed_backend

        current_device_group: ProcessGroup | None = None
        current_group_ranks: list[int] | None = None

        # 依序创建全部同类子组，保证每个 rank 的 collective 调用顺序完全一致。
        for ranks in group_ranks:
            device_group = dist.new_group(
                ranks=ranks,
                backend=torch_distributed_backend,
            )
            if self.rank in ranks:
                current_device_group = device_group
                current_group_ranks = ranks

        if current_device_group is None or current_group_ranks is None:
            raise RuntimeError(
                f"Global rank {self.rank} does not belong to any "
                f"{group_name or 'anonymous'} group: {group_ranks}."
            )

        # 这些字段与 vLLM GroupCoordinator 的核心含义一致，供模型层和未来通信封装读取。
        self.ranks = current_group_ranks
        self.world_size = len(current_group_ranks)
        self.rank_in_group = current_group_ranks.index(self.rank)
        self.device_group = current_device_group

    def destroy(self) -> None:
        """销毁当前逻辑组的 device process group，并解除本对象持有的引用。

        上层先停止依赖该组的模型执行，再按 TP、WORLD、默认 process group 的顺序
        关闭。当前没有独立 CPU group 或 device communicator，只销毁已有设备组。
        与 vLLM 一样删除 device_group 属性，重复调用时不会再次销毁同一组。

        Side Effects:
            调用 torch.distributed.destroy_process_group() 释放组内通信资源；
            组销毁失败时异常继续传播，属性仍保留以反映尚未完成的清理。
        """
        # 未创建或已经销毁的组无需清理；不把缺失属性误当成默认 process group。
        if hasattr(self, "device_group"):
            dist.destroy_process_group(self.device_group)
            del self.device_group

    @contextmanager
    def graph_capture(
        self,
        graph_capture_context: GraphCaptureContext | None = None,
    ):
        """为本逻辑组建立捕获 stream 环境，进入后将上下文交给调用方。

        沿用 vLLM 的创建/复用 stream、等待当前 stream、切换并 yield 的流程。
        当前没有 device_communicator、自定义 all-reduce 或 ROCm aiter，因而
        不进入上游 ca_comm.capture()/aiter.capture() 的通信专用上下文。

        Args:
            graph_capture_context: 要共享的已有上下文；None 时在当前 CUDA
                device 上创建新 stream。Runner 应优先使用模块级 graph_capture，
                由它为指定 device 创建上下文，再传给本接口。

        Yields:
            graph_capture_context: 本作用域实际使用的 GraphCaptureContext，保留传入对象身份。
                调用方可将其 stream 传给 torch.cuda.graph，发起真正的捕获。

        Side Effects:
            捕获 stream 等待进入前的当前 stream，保证其已有初始化工作先完成。
            作用域内切换当前 stream，正常或异常退出时恢复进入前的 stream。
            wait_stream 建立 GPU 依赖，不同步等待 CPU，也不调用设备级 synchronize。
            本接口不调用 cudaStreamBeginCapture，不修改 Graph 捕获许可。
            退出时不额外让原 stream 等待本 stream；跨 stream 消费结果的依赖
            由调用方维护。TP 通信的预热和多 rank 一致捕获顺序也由上层保证。
        """
        if graph_capture_context is None:
            stream = torch.cuda.Stream()  # 直接调用组接口时使用当前 device。
            graph_capture_context = GraphCaptureContext(stream)  # 新建可共享上下文。
        else:
            stream = graph_capture_context.stream  # 复用同一 stream，不创建第二套环境。

        # 与上游一致，在切换 stream 前保证已提交的初始化工作先于捕获环境的工作。
        curr_stream = torch.cuda.current_stream()
        if curr_stream != stream:
            stream.wait_stream(curr_stream)  # 同 stream 已有顺序，无需等待自己。

        # torch.cuda.stream 在 finally 中恢复原 stream，模型/捕获抛异常时同样生效。
        with torch.cuda.stream(stream):
            yield graph_capture_context  # 保持作用域开放，由调用方预热或真正捕获。


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
        group_name="world",
    )


def init_model_parallel_group(
    group_ranks: list[list[int]],
    local_rank: int,
    backend: str,
    group_name: str,
) -> GroupCoordinator:
    """创建一种模型并行维度的全部子组。

    Args:
        group_ranks: 该并行维度的全部 rank 分组。
        local_rank: 当前进程的本机 device 编号。
        backend: 创建子组使用的通信 backend。
        group_name: 逻辑组名称，例如 ``"tp"``。

    Returns:
        group: 当前 rank 所属的逻辑并行组描述。
    """
    return GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=local_rank,
        torch_distributed_backend=backend,
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
