"""KV cache 的静态配置定义。"""

from dataclasses import dataclass, field
from typing import Literal
CacheDType = Literal[
    "auto",
    "float16",
    "bfloat16",
]

@dataclass(slots=True)
class CacheConfig:
    """保存 KV cache 的配置策略和少量初始化后的运行时统计镜像。

    Engine 根据策略和显存 profiling 创建 KVCacheConfig，后者是实际分配方案。
    Worker 正式初始化时将 block 数回填到 num_gpu_blocks；配置中的镜像字段
    不替代 KVCacheConfig 的 tensor 布局、group 信息或 Scheduler 的动态占用状态。

    Attributes:
        block_size: 一个物理 KV cache block 能容纳的 token 数。当前 Attention
            backend 要求它是 256 的整数倍。
        gpu_memory_utilization: 当前 Worker 最多用于模型权重、激活值和 KV cache
            的 GPU 总显存比例。Worker 的 profiling 会用此比例换算可用 KV 内存。
        prefix_caching_hash_algo: 构造完整 KV block 哈希时使用的算法名称。
        num_gpu_blocks_override: 可选的用户指定 block 数。非 ``None`` 时，KV cache
            规划器会用它替代 profiling 得到的 block 数，方便复现实验或调试显存
            问题；
        num_gpu_blocks: 正式 KV 配置的物理 block 总数镜像，包含保留空块。
            初始为 None，由 Worker.initialize_from_config() 回填。
        num_cpu_blocks: 上游 CPU cache 容量统计的预留字段；当前不支持 CPU KV
            offload / swap，保持 None。
        kv_cache_size_tokens: 上游 KV token 容量统计的预留字段；当前尚无回填路径。
        kv_cache_memory_bytes: 显式 KV 字节预算的预留配置，默认 None；当前仅用于
            Worker 显存诊断日志的分支判断，尚未接入 determine_available_memory()
            的显式预算分支，不能用它替代 gpu_memory_utilization 进行实际规划。
    """

    # 一个物理 page 覆盖的 token 数；当前原始 attention 实现要求 256 对齐。
    block_size: int = 256
    # 预留给当前 Worker 的 GPU 总显存比例。
    gpu_memory_utilization: float = 0.92
    # prefix cache 的 block 哈希算法；当前哈希工具仅实现 sha256。
    prefix_caching_hash_algo: Literal["sha256"] = "sha256"
    # 用户覆盖 profiling 结果的显式开关；None 表示始终相信 profiling 结果。
    num_gpu_blocks_override: int | None = None

    # 由正式 KV 初始化回填；init=False 表示不是用户输入的容量覆盖参数。
    num_gpu_blocks: int | None = field(default=None, init=False)
    """GPU 物理 block 总数镜像；真实分配方案仍由 KVCacheConfig 表达。"""
    # 当前没有 CPU KV cache 管理器，保留上游字段名但不产生实际 CPU 分配。
    num_cpu_blocks: int | None = field(default=None, init=False)
    """CPU KV block 数的预留统计字段，当前保持 None。"""

    # 预留 KV 初始化后的 token 容量统计，当前消费者应从正式 KV 配置推导容量。
    kv_cache_size_tokens: int | None = field(default=None, init=False)
    # 该配置尚未接通显式预算计算；设置它目前只会影响显存建议日志的输出分支。
    kv_cache_memory_bytes: int | None = None
    def __post_init__(self) -> None:
        """校验不会依赖模型结构或 GPU 状态的缓存参数。

        Raises:
            ValueError: block 大小、显存比例或显式 block 数不满足基本约束时抛出。
        """
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.block_size % 256 != 0:
            raise ValueError("block_size must be divisible by 256")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError(
                "gpu_memory_utilization must be in the interval (0, 1]"
            )
        if (
            self.num_gpu_blocks_override is not None
            and self.num_gpu_blocks_override <= 0
        ):
            raise ValueError("num_gpu_blocks_override must be positive")
