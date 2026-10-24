from dataclasses import dataclass, field
import torch
import math
from nanovllm.utils.mem_utils import MemorySnapshot, format_gib
from nanovllm.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
)
from nanovllm.config import CacheConfig, NanovllmConfig
from nanovllm.model_executor.layers.attention import Attention
from nanovllm.attention.backend import AttentionBackend, AttentionMetadataBuilder, MultipleOf

@dataclass
class AttentionGroup:
    """共享一个后端与 KV 规格的 Attention 层集合。

    Attributes:
        backend: 当前组使用的 Attention 后端类。
        layer_names: 组内各层在静态 forward context 中的完整名称。
        kv_cache_spec: 组内层共享的 KV cache 规格。
        kv_cache_group_id: 所属逻辑 KV cache group 的编号。
        metadata_builders: 已创建的元数据 Builder；当前同步路径只有一个。
    """

    backend: type[AttentionBackend]
    layer_names: list[str]
    kv_cache_spec: KVCacheSpec
    kv_cache_group_id: int
    metadata_builders: list[AttentionMetadataBuilder] = field(default_factory=list)

    def create_metadata_builders(
        self,
        vllm_config: NanovllmConfig,
        device: torch.device,
        kernel_block_size: int | None = None,
        num_metadata_builders: int = 1,
    ) -> None:
        """在 kernel page 大小确定后创建组内共享的 Builder。

        Args:
            vllm_config: 当前 Worker 的完整配置。
            device: 当前 TP rank 的执行设备。
            kernel_block_size: 后端实际使用的 page token 数。
            num_metadata_builders: 独立 Builder 的数量；当前传 1。

        Side Effects:
            替换 metadata_builders，并将实际 kernel page 大小传给 Builder。
        """
        builder_spec = (
            self.kv_cache_spec.copy_with_new_block_size(kernel_block_size)
            if kernel_block_size is not None
            else self.kv_cache_spec
        )
        self.metadata_builders = [
            self.backend.get_builder_cls()(
                builder_spec,
                self.layer_names,
                vllm_config,
                device,
            )
            for _ in range(num_metadata_builders)
        ]

    def get_metadata_builder(self, ubatch_id: int = 0) -> AttentionMetadataBuilder:
        """返回已创建的 Builder；当前同步执行只使用下标 0。

        Args:
            ubatch_id: 要取得的 Builder 下标。

        Returns:
            metadata_builder: 与此 AttentionGroup 关联的元数据 Builder。
        """
        return self.metadata_builders[ubatch_id]


def select_common_block_size(
    kv_manager_block_size: int,
    backends: list[type[AttentionBackend]],
) -> int:
    """为一个逻辑 KV group 选择所有后端都支持的 kernel page 大小。

    Args:
        kv_manager_block_size: KVCacheManager 的逻辑 block token 数。
        backends: 此 KV group 中各 AttentionGroup 的后端类。

    Returns:
        kernel_block_size: 能整除逻辑 block 大小的 kernel page token 数。

    Raises:
        ValueError: 组内没有后端或不存在共同支持的 page 大小时抛出。
    """
    if not backends:
        raise ValueError("Attention KV cache group has no backend")

    def supports_all(block_size: int) -> bool:
        """检查候选 page 大小是否满足全部后端约束。

        Returns:
            is_supported: 候选 block_size 满足全部 Attention 后端的 page 约束时为 True。
        """
        for backend in backends:
            sizes = backend.get_supported_kernel_block_sizes()
            if not any(
                block_size == size
                if isinstance(size, int)
                else block_size % size.base == 0
                for size in sizes
            ):
                return False
        return True

    # 原 manager block 若可直接用于 kernel，就无需拆分 page。
    if supports_all(kv_manager_block_size):
        return kv_manager_block_size

    fixed_sizes = {
        size
        for backend in backends
        for size in backend.get_supported_kernel_block_sizes()
        if isinstance(size, int)
    }
    # 固定 page 需整除 manager block；选最大的共同支持值以减少 page 数。
    for block_size in sorted(fixed_sizes, reverse=True):
        if kv_manager_block_size % block_size == 0 and supports_all(block_size):
            return block_size
    raise ValueError(
        f"No common kernel block size for manager block size {kv_manager_block_size}"
    )


def prepare_kernel_block_sizes(
    kv_cache_config: KVCacheConfig,
    attn_groups: list[list[AttentionGroup]],
) -> list[int]:
    """按逻辑 KV group 顺序计算 Attention kernel page 大小。

    Args:
        kv_cache_config: Engine 生成的 KV cache 分配计划。
        attn_groups: 按 KV group 编号组织的 AttentionGroup 列表。

    Returns:
        kernel_block_sizes: 与 kv_cache_groups 一一对齐的 kernel page 大小列表。
    """
    kernel_block_sizes: list[int] = []
    for group_id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
        group_backends = [group.backend for group in attn_groups[group_id]]
        kernel_block_sizes.append(
            select_common_block_size(
                kv_cache_group.kv_cache_spec.block_size,
                group_backends,
            )
        )
    return kernel_block_sizes


def request_memory(
    init_snapshot:MemorySnapshot,
    cache_config:CacheConfig,
)->int:
    """计算当前 nano-vLLM 实例的显存总预算，并验证启动时空闲显存足够。

    ``gpu_memory_utilization`` 的语义是“当前 nano-vLLM 实例最多可使用设备总显存的
    比例”，因此预算基于 ``total_memory``，而不是当前 ``free_memory``。当前空闲
    显存不足以满足此预算时，后续即使不分配 KV cache，也无法可靠加载模型并完成
    profiling，应尽早报错。

    Args:
        init_snapshot: NCCL 初始化完成、模型权重加载前采集的设备基线快照。
        cache_config: 包含 ``gpu_memory_utilization`` 的静态缓存策略配置。

    Returns:
        requested_memory: 当前 Worker 可使用的 GPU 总显存预算，单位为字节。它包含模型权重、forward
            峰值激活、非 PyTorch 内存、CUDA Graph 内存和最终 KV cache，而不只是 KV
            cache 自身。

    Raises:
        ValueError: 启动时设备空闲显存低于用户请求的总预算时抛出。
    """
    # 向上取整，确保 0.92 之类的浮点比例不会因截断而少计算一个字节。
    requested_memory=math.ceil(
        init_snapshot.total_memory*cache_config.gpu_memory_utilization
    )
    if init_snapshot.free_memory<requested_memory:
        raise ValueError(
            f"Free memory on device {init_snapshot.device_} "
            f"({format_gib(init_snapshot.free_memory)}/"
            f"{format_gib(init_snapshot.total_memory)} GiB) on startup "
            f"is less than desired GPU memory utilization "
            f"({cache_config.gpu_memory_utilization}, "
            f"{format_gib(requested_memory)} GiB). Decrease GPU memory "
            "utilization or reduce GPU memory used by other processes."
        )
    return requested_memory

def bind_kv_cache(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, Attention],
    runner_kv_caches: list[torch.Tensor],
) -> None:
    """将按层组织的 combined KV cache 绑定给 Runner 与 Attention。

    Args:
        kv_caches: 完整层名到后端 KV cache view 的映射。当前布局是
            [num_kernel_blocks, K/V, kernel_block_size, kv_heads, head_size]。
        forward_context: 模型构造期注册的完整层名到 Attention module 映射。
        runner_kv_caches: Runner 持有的 cache view 列表，调用前必须为空。

    Side Effects:
        Runner 保留各层 view；对应 Attention module 的 kv_cache 指向同一 tensor。
    """
    assert not runner_kv_caches
    for layer_name, kv_cache in kv_caches.items():
        runner_kv_caches.append(kv_cache)
        attention_layer = forward_context[layer_name]
        # Attention.forward 通过该 combined view 调用后端，与 Runner 共享底层存储。
        attention_layer.kv_cache = kv_cache
