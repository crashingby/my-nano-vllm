"""nano-vLLM 的并行执行配置。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ParallelConfig:
    """描述模型执行时采用的并行策略。

    当前保留 Tensor Parallelism（TP）所需的 size、world 和 rank 描述；模型层和
    分布式工具可据此组织分片与分组。端到端 Engine 目前只选择 TP=1 的
    UniProcExecutor，配置对象可以表达更大的 TP size，但不代表执行器已支持多卡。
    当前没有 PP、DP 或 EP 配置。

    Attributes:
        tensor_parallel_size: 一个模型副本使用的 TP rank 数。当前 Executor 只实现
            ``1``；大于 1 的值允许被配置对象表达，但会在选择 Executor 时明确拒绝，
            不会仅因 ParallelConfig 构造成功就启动多个 Worker。
        world_size: 当前模型并行 world 的 rank 总数。它由当前并行策略推导，不允许
            调用方单独指定；只支持 TP 时恒等于 ``tensor_parallel_size``。
        rank: 当前 Worker 在模型并行 world 中的全局 rank。调用方创建配置时默认是
            ``0``；WorkerBase 构造时会将其更新为 Executor 指定的 rank。
    """

    # 描述一个模型副本的 TP rank 数，Executor.get_class() 另行检查当前执行能力。
    tensor_parallel_size: int = 1
    # 当前仅有 TP，因此模型并行 world size 由 TP size 唯一决定。
    world_size: int = field(init=False)
    # WorkerBase 构造时回写实际 rank；当前单卡执行器始终传入 0。
    rank: int = 0

    def __post_init__(self) -> None:
        """检查 TP size 为正，推导 world_size，再校验 rank 范围。

        Side Effects:
            写入 world_size；不创建 process group，也不检查模型 head 数的整除关系。

        Raises:
            ValueError: TP size 非正，或 rank 不在 [0, world_size) 内时抛出。
        """
        if self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")
        # 纯 TP 场景下，global model-parallel world 与 TP group 完全相同。
        self.world_size = self.tensor_parallel_size
        # 配置构造先检查默认或显式 rank；实际 Worker 身份稍后由执行器确定。
        if not 0 <= self.rank < self.world_size:
            raise ValueError(
                "rank must be in [0, world_size), got "
                f"rank={self.rank}, world_size={self.world_size}"
            )
