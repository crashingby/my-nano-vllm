"""nano-vLLM 的并行执行配置。"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from nanovllm.config.utils import config

if TYPE_CHECKING:
    from nanovllm.executor import Executor
else:
    Executor = Any  # 为 Pydantic 提供运行时可解析的类型，避免加载执行器。

DistributedExecutorBackend = Literal["mp", "uni"]


@config(slots=True)
class ParallelConfig:
    """描述模型执行时采用的并行策略。

    当前支持单机 CUDA TP。未指定后端时，TP=1 选择 UniProcExecutor，
    TP>1 选择 MultiprocExecutor；TP=1 也可显式选择 mp，验证子进程执行路径。
    可见 GPU 数量由执行器启动时检查，配置构造不查询硬件。PP 只接受 1，
    尚未实现 DP 或 EP。

    Attributes:
        tensor_parallel_size: 一个模型副本使用的 TP rank 数，各 rank 共同执行所有层。
        pipeline_parallel_size: 当前必须为 1，不建立流水线并行组。
        distributed_executor_backend: uni、mp、自定义 Executor 类或其导入路径；
            None 在初始化时按 TP size 解析为 uni 或 mp。
        world_size: 当前模型并行 world 的 rank 总数。它由当前并行策略推导，不允许
            调用方单独指定；只支持 TP 时恒等于 ``tensor_parallel_size``。
        rank: 当前 Worker 在模型并行 world 中的全局 rank。调用方创建配置时默认是
            ``0``；WorkerBase 构造时会将其更新为 Executor 指定的 rank。
    """

    # 配置只描述逻辑分组，可见设备数量留到执行器创建资源前检查。
    tensor_parallel_size: int = Field(default=1, ge=1)
    pipeline_parallel_size: int = Field(default=1, ge=1)
    distributed_executor_backend: (
        str | DistributedExecutorBackend | type[Executor] | None
    ) = None

    is_moe_model: bool | None = None
    # 当前仅有 TP，因此模型并行 world size 由 TP size 唯一决定。
    world_size: int = field(init=False)
    # WorkerBase 在各自进程中回写实际 rank，父进程配置保持默认值。
    rank: int = 0
    enable_shm_tensor_arena: bool = False

    def __post_init__(self) -> None:
        """推导纯 TP world 与默认后端，拒绝不支持的并行组合。

        Side Effects:
            写入 world_size 和默认后端；不创建进程组，模型维度由 ModelConfig 检查。

        Raises:
            ValueError: TP size、PP size、uni/TP 组合或 rank 范围不合法。
        """
        if self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")
        # 当前执行链只实现 TP，不能接受尚未实现的 PP 配置。
        if self.pipeline_parallel_size != 1:
            raise ValueError("Pipeline parallelism is not supported")
        # 纯 TP 场景下，global model-parallel world 与 TP group 完全相同。
        self.world_size = self.tensor_parallel_size
        # 未指定后端时，单卡使用同进程 Worker，多卡使用多进程 Worker。
        if self.distributed_executor_backend is None:
            self.distributed_executor_backend = (
                "uni" if self.world_size == 1 else "mp"
            )
        # uni 只创建一个 Worker，不能让它等待不存在的其他 TP rank。
        if self.distributed_executor_backend == "uni" and self.world_size != 1:
            raise ValueError("UniProcExecutor requires tensor_parallel_size=1")
        # 配置构造先检查默认或显式 rank；实际 Worker 身份稍后由执行器确定。
        if not 0 <= self.rank < self.world_size:
            raise ValueError(
                "rank must be in [0, world_size), got "
                f"rank={self.rank}, world_size={self.world_size}"
            )
