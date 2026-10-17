"""nano-vLLM 的并行执行配置。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ParallelConfig:
    """描述模型执行时采用的并行策略。

    当前 nano-vLLM 仅规划并支持 Tensor Parallelism（TP）。保留独立配置对象，
    使 Executor、Worker 与模型层不再从总配置直接读取并行细节；未来增加 PP、DP
    或 EP 时，可在本模块扩展，而无需再次改变 ``NanovllmConfig`` 的顶层接口。

    Attributes:
        tensor_parallel_size: 一个模型副本使用的 TP rank 数。当前 Executor 只实现
            ``1``；大于 1 的值允许被配置对象表达，但会在选择 Executor 时明确拒绝，
            直到 MultiprocExecutor 实现完成。
        world_size: 当前模型并行 world 的 rank 总数。它由当前并行策略推导，不允许
            调用方单独指定；只支持 TP 时恒等于 ``tensor_parallel_size``。
        rank: 当前 Worker 在模型并行 world 中的全局 rank。Engine 构造配置时默认是
            ``0``；Worker 初始化时会将其更新为自身 rank。
    """

    # TP world size；每个 rank 持有模型权重和 KV cache 的一个分片。
    tensor_parallel_size: int = 1
    # 当前仅有 TP，因此模型并行 world size 由 TP size 唯一决定。
    world_size: int = field(init=False)
    # rank 是运行期身份；WorkerBase/ModelRunner 创建时会写入实际 rank。
    rank: int = 0

    def __post_init__(self) -> None:
        """校验不依赖具体 GPU 或模型结构的 TP 参数。

        Raises:
            ValueError: TP size 不是正整数时抛出。
        """
        if self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")
        # 纯 TP 场景下，global model-parallel world 与 TP group 完全相同。
        self.world_size = self.tensor_parallel_size
        if not 0 <= self.rank < self.world_size:
            raise ValueError(
                "rank must be in [0, world_size), got "
                f"rank={self.rank}, world_size={self.world_size}"
            )
