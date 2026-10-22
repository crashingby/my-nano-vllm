"""KV cache 的分配策略及正式初始化后的容量镜像。"""

from dataclasses import field
from typing import Literal

from nanovllm.config.utils import config
# 保留缓存精度的类型别名；当前未提供独立 dtype 字段，实际 KV 精度跟随模型。
CacheDType = Literal[
    "auto",
    "float16",
    "bfloat16",
]

@config(slots=True)
class CacheConfig:
    """保存 KV cache 的配置策略和少量初始化后的运行时统计镜像。

    Engine 根据本配置和 Worker 的显存 profiling 结果创建 KVCacheConfig，再将
    分配方案下发给执行端。Engine 规划完成和 Worker 接收正式方案时都会回写
    num_gpu_blocks；实际 tensor 布局和请求的动态占用仍由 KVCacheConfig、Runner
    与 Scheduler 管理。本类构造和校验均不查询 GPU 或分配 KV tensor。

    Attributes:
        block_size: 一个物理 KV cache block 能容纳的 token 数。当前 Attention
            backend 要求它是 256 的整数倍。
        gpu_memory_utilization: 当前实例的预算占设备总显存的比例，不是占启动时
            空闲显存的比例。Worker 从该预算扣除权重、峰值激活、非 PyTorch 分配
            和 Graph 估算占用，剩余部分才是候选 KV 字节预算。
        prefix_caching_hash_algo: 完整逻辑 block 的 prefix hash 算法，当前仅支持
            sha256；Engine 据此初始化哈希根并创建所有请求共享的 block hasher。
        num_gpu_blocks_override: 可选的用户指定 block 数。非 ``None`` 时，KV cache
            规划器用它替代根据显存预算计算的 block 数，仍会执行 profiling。
            该总数包含空块 0，正式规划还会检查能否容纳最大上下文和保留空块。
        num_gpu_blocks: 正式 KV 配置的物理 block 总数镜像，包含保留空块。
            初始为 None，由 LLMEngine._initialize_kv_caches() 和
            Worker.initialize_from_config() 回填；不能作为构造参数指定。
        num_cpu_blocks: 上游 CPU cache 容量统计的预留字段；当前不支持 CPU KV
            offload / swap，保持 None。
        kv_cache_size_tokens: KV token 容量统计的预留字段，当前没有消费者或回填
            路径。当前单 group 的请求可分配容量应由 (num_blocks - 1) * block_size
            推导，并继续受每请求 max_model_len 和实际占用约束。
        kv_cache_memory_bytes: 显式 KV 字节预算的预留配置，默认 None；当前仅用于
            Worker 显存诊断日志的分支判断，尚未接入 determine_available_memory()
            的显式预算分支，不能用它替代 gpu_memory_utilization 进行实际规划。
    """

    # 每个物理 page 容纳的 token 数；当前接入的上游 FA2 paged 接口要求 256 对齐。
    block_size: int = 256
    # 权重、激活、Graph 和 KV 共同使用这一总显存预算，不表示 KV 独占的比例。
    gpu_memory_utilization: float = 0.92
    # prefix cache 的 block 哈希算法；当前哈希工具仅实现 sha256。
    prefix_caching_hash_algo: Literal["sha256"] = "sha256"
    # 覆盖规划出的总 block 数；None 表示使用预算推导结果，包含保留空块 0。
    num_gpu_blocks_override: int | None = None

    # 由正式 KV 初始化回填；init=False 表示不是用户输入的容量覆盖参数。
    num_gpu_blocks: int | None = field(default=None, init=False)
    """GPU 物理 block 总数镜像；例如 129 表示空块 0 加 128 个可分配 blocks。"""
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
        # page 大小必须为正，随后按当前 FA2 paged kernel 的粒度约束检查。
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.block_size % 256 != 0:
            raise ValueError("block_size must be divisible by 256")
        # 这里校验预算比例的取值范围；实际空闲显存是否足够由 Worker 启动时检查。
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError(
                "gpu_memory_utilization must be in the interval (0, 1]"
            )
        # override 是可选的容量输入，真实分配能力仍由后续规划和设备分配检验。
        if (
            self.num_gpu_blocks_override is not None
            and self.num_gpu_blocks_override <= 0
        ):
            raise ValueError("num_gpu_blocks_override must be positive")
