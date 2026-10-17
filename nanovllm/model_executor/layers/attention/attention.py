# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""沿用 vLLM 的 Attention 包装层流程，将缓存写入和计算委托给后端。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

from nanovllm.attention.backend import AttentionBackend
from nanovllm.attention.backends.flash_attn import FlashAttentionBackend
from nanovllm.config import get_current_config
from nanovllm.forward_context import get_forward_context
from nanovllm.kv_cache_interface import FullAttentionSpec
from nanovllm.model_executor.layers.attention_layer_base import AttentionLayerBase

if TYPE_CHECKING:
    from nanovllm.attention.backend import AttentionImpl, AttentionMetadata
    from nanovllm.config import NanovllmConfig
    from nanovllm.forward_context import ForwardContext
    from nanovllm.kv_cache_interface import KVCacheSpec


class Attention(nn.Module, AttentionLayerBase):
    """保存单层 decoder Attention 状态，并调度后端的缓存写入与计算。

    构造时将自身登记到 static_forward_context，Worker 随后规划并绑定
    combined KV cache。forward 不区分 prefill/decode，也不直接调用 kernel；
    后端根据 Runner 在 ForwardContext 中安装的 metadata 执行计算。

    当前保留 vLLM 的直接调用路径；未接入 torch.compile custom op、量化、
    sliding window、MLA 或跨层 KV cache 共享。

    Attributes:
        layer_name: 模型中的完整层名，用于关联 metadata、slot mapping 和缓存。
        attn_backend: 声明后端能力、实现类及缓存布局的后端类型。
        impl: 当前层独立持有的后端实现对象。
        dtype: 构造时读取的模型计算精度；ModelLoader 在构建期间设置默认 dtype。
        kv_cache_torch_dtype: 缓存存储精度；当前 auto 从 ModelConfig.dtype 取得。
        kv_cache: 当前层的 combined KV cache，由 Worker 在构造后绑定。
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        attn_backend: type[AttentionBackend] | None = None,
    ) -> None:
        """创建后端实现，并把本层注册到当前模型的静态上下文。

        Args:
            num_heads: 当前 TP rank 的 Query head 数。
            head_size: 每个 head 的维度，沿用 vLLM 命名。
            scale: Attention score 的缩放系数。
            num_kv_heads: 当前 TP rank 的 KV head 数，缺省时等于 num_heads。
            prefix: 本层完整路径，例如 model.layers.0.self_attn.attn。
            attn_backend: 显式指定后端；缺省使用当前支持的 FlashAttention。

        Raises:
            ValueError: 层名为空或与已登记的层重名。
            AssertionError: 未安装构建期配置，或 Query head 数不是 KV head 数的整数倍。

        Side Effects:
            将 prefix -> self 写入当前配置的 static_forward_context；
            此处仅创建空缓存占位 tensor，不分配实际 KV page pool。
        """
        super().__init__()
        # 模型构建期配置提供缓存精度和静态层注册表。
        vllm_config = get_current_config()
        
        # 当前 CacheConfig 未提供独立缓存精度选项，auto 跟随模型精度。
        self.kv_cache_dtype: str = "auto"
        self.kv_cache_torch_dtype: torch.dtype = vllm_config.model_config.dtype
        if num_kv_heads is None:
            num_kv_heads = num_heads  # 普通 MHA 的 Query 和 KV head 数相同。
        # 每个 KV head 必须对应整数个 Query heads，保持 GQA 分组成立。
        assert num_heads % num_kv_heads == 0, (
            f"num_heads ({num_heads}) is not divisible by "
            f"num_kv_heads ({num_kv_heads})"
        )
        self.num_heads: int = num_heads  # 本 rank 的 Query head 数。
        self.head_size: int = head_size  # Query/Key 的每 head 维度。
        self.num_kv_heads: int = num_kv_heads  # 本 rank 的 KV head 数。
        self.scale: float = scale  # 后端计算 Attention score 时使用。
        self.head_size_v: int = head_size  # 当前 Value 与 Query/Key 维度相同。
        self.layer_name: str = prefix  # 运行时按此名称查找当前层的数据。

        # ModelLoader 在模型构建期间将默认 dtype 设为模型权重和激活精度。
        # 这里沿用 vLLM 读取构造期计算精度，缓存存储精度由上方配置单独记录。
        dtype = torch.get_default_dtype()
        # 后端选择和实例创建沿用 vLLM 的 attn_backend / impl 分工；
        # 当前缺省固定使用 FlashAttention，尚未根据 dtype 自动筛选后端。
        self.attn_backend: type[AttentionBackend] = (
            FlashAttentionBackend if attn_backend is None else attn_backend
        )
        # 工厂基类返回类型较宽；此包装层使用普通 decoder AttentionImpl。
        impl_cls = cast(type["AttentionImpl[Any]"], self.attn_backend.get_impl_cls())
        self.impl: AttentionImpl[Any] = impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            kv_cache_dtype=self.kv_cache_dtype,
        )
        self.dtype = dtype  # 保存构造期计算精度；此赋值本身不转换输入 tensor。
        # 同名层会竞争缓存和 metadata，必须在修改共享注册表前拒绝。
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        # 与 vLLM 一致：Worker 的 bind_kv_cache 后续替换该占位 tensor。
        self.kv_cache: torch.Tensor = torch.tensor([])

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        """准备输出和 head 视图，按后端能力先更新缓存再计算 Attention。

        调用前 Runner 必须通过 set_forward_context 安装本轮按层组织的
        metadata 和 slot mapping。这里只保留 vLLM 的直接函数调用路径。

        Args:
            query: packed Query，形状为 [T, H * D] 或 [T, H, D]。
            key: packed Key，形状为 [T, Hkv * D] 或 [T, Hkv, D]。
            value: packed Value，与 Key 使用相同的 token 顺序和 head 数。
            output_shape: 预分配输出的形状，缺省为 [T, H * D]。

        Returns:
            后端写入的输出，形状为 [-1, output_shape[-1]]；默认是 [T, H * D]。
            例如 Query [6, 4, 8] 对应输出 [6, 32]。

        Side Effects:
            若后端要求独立缓存更新且本层存在 slot mapping，先调用后端
            do_kv_cache_update 修改 self.kv_cache，再由 impl.forward 写入输出。
        """
        # 输出跟随本轮 Query 的实际精度，不由构造期 self.dtype 或缓存精度强制指定。
        output_dtype = query.dtype
        if output_shape is None:
            num_tokens = query.shape[0]  # 二维和三维 Query 均以首维表示 token。
            output_shape = torch.Size((num_tokens, self.num_heads * self.head_size_v))
        # 包装层持有输出分配，后端只负责写入，统一各后端的输出接口。
        output = torch.empty(output_shape, dtype=output_dtype, device=query.device)
        hidden_size = output_shape[-1]  # 最后还原成模型使用的二维输出。
        query = query.view(-1, self.num_heads, self.head_size)
        output = output.view(-1, self.num_heads, self.head_size_v)
        if key is not None:
            key = key.view(-1, self.num_kv_heads, self.head_size)
        if value is not None:
            value = value.view(-1, self.num_kv_heads, self.head_size_v)

        # 保留 vLLM 的依赖参数命名；当前顺序由直接调用保证，未注册编译算子。
        kv_cache_dummy_dep: torch.Tensor | None = None
        if (
            not self.attn_backend.forward_includes_kv_cache_update
            and key is not None
            and value is not None
        ):
            kv_cache_dummy_dep = unified_kv_cache_update(key, value, self.layer_name)
        unified_attention_with_output(
            query,
            key,
            value,
            output,
            self.layer_name,
            kv_cache_dummy_dep=kv_cache_dummy_dep,
        )
        return output.view(-1, hidden_size)

    def extra_repr(self) -> str:
        """在 module 展示中给出本层 head 参数和实际后端实现名称。"""
        return (
            f"head_size={self.impl.head_size}, num_heads={self.impl.num_heads}, "
            f"num_kv_heads={self.impl.num_kv_heads}, scale={self.impl.scale}, "
            f"backend={self.impl.__class__.__name__}"
        )

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        """把权重加载后的处理委托给后端，不在包装层执行量化处理。

        Args:
            act_dtype: ModelLoader 提供的模型激活精度。

        Side Effects:
            后端可准备执行状态；具体修改由 impl 的实现决定。
        """
        self.impl.process_weights_after_loading(act_dtype)

    def get_attn_backend(self) -> type[AttentionBackend]:
        """返回后端类型，供 Runner 创建 AttentionGroup 和 metadata builder。"""
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: NanovllmConfig) -> KVCacheSpec:
        """返回本层普通 decoder Attention 的 KV cache 存储规格。

        Args:
            vllm_config: 当前 Worker 配置；每次读取最新 block_size，允许其在
                模型加载后、缓存规划前调整。

        Returns:
            使用本 rank KV head 数、head_size 和模型缓存精度的 FullAttentionSpec。
        """
        block_size = vllm_config.cache_config.block_size
        return FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            dtype=self.kv_cache_torch_dtype,
        )


def get_attention_context(
    layer_name: str,
) -> tuple[AttentionMetadata | None, Attention, torch.Tensor, torch.Tensor | None]:
    """从活动 ForwardContext 中提取指定层的运行时数据。

    Args:
        layer_name: 构造 Attention 时注册的完整层名。

    Returns:
        (本层 metadata, 本层实例, 本层 combined KV cache, 本层 slot mapping)。
        profiling 时 metadata 可以为 None；未安排缓存写入时 slot mapping 为 None。
        返回原始对象引用，不复制 tensor。

    Raises:
        AssertionError: 未安装上下文，或 slot_mapping 不是按层组织的字典。
        KeyError: 静态注册表或非空 metadata 字典中缺少指定层名。
    """
    forward_context: ForwardContext = get_forward_context()
    attn_metadata_raw = forward_context.attn_metadata
    if isinstance(attn_metadata_raw, dict):
        attn_metadata = attn_metadata_raw[layer_name]  # 按层查找本轮后端元数据。
    else:
        attn_metadata = attn_metadata_raw  # profiling 无真实 metadata 时保留 None。
    attn_layer = cast(Attention, forward_context.no_compile_layers[layer_name])
    kv_cache = attn_layer.kv_cache  # 使用 Worker 已绑定的层级缓存视图。
    slot_mapping = forward_context.slot_mapping
    # 当前不支持多 microbatch；缓存映射必须采用单次 forward 的按层字典。
    assert isinstance(slot_mapping, dict), (
        f"Expected slot_mapping to be a dict, got {type(slot_mapping)}."
    )
    layer_slot_mapping = slot_mapping.get(layer_name)
    return attn_metadata, attn_layer, kv_cache, layer_slot_mapping


def unified_kv_cache_update(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    """将本轮 Key/Value 的缓存写入委托给指定层的后端实现。

    Args:
        key: 已整理成 [T, Hkv, D] 的本轮 Key。
        value: 已整理成 [T, Hkv, D] 的本轮 Value。
        layer_name: 需要更新缓存的完整 Attention 层名。

    Returns:
        与缓存同设备、同精度的空 tensor，沿用 vLLM 的依赖参数接口。
        当前未注册 custom op，它本身不提供 torch.compile 的顺序保证。

    Side Effects:
        存在本层 slot mapping 时，由 impl.do_kv_cache_update 修改缓存。
        无映射时跳过更新；padding slot -1 的处理由后端负责。

    Raises:
        AssertionError: 要求写入缓存，但后端尚未实现 do_kv_cache_update。
    """
    _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)
    if layer_slot_mapping is not None:
        # 能力声明要求独立更新的后端必须提供实际写入接口。
        assert hasattr(attn_layer.impl, "do_kv_cache_update"), (
            f"{attn_layer.impl.__class__.__name__} does not support kv cache update"
        )
        attn_layer.impl.do_kv_cache_update(  # type: ignore[attr-defined]
            attn_layer, key, value, kv_cache, layer_slot_mapping
        )
    return torch.empty(0, device=kv_cache.device, dtype=kv_cache.dtype)


def unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    """按层名查找运行时数据，调用后端将 Attention 结果写入 output。

    Args:
        query: 本轮 Query，形状为 [T, H, D]。
        key: 本轮 Key，形状为 [T, Hkv, D]。
        value: 本轮 Value，形状为 [T, Hkv, D]。
        output: 包装层预分配的 [T, H, D] 输出视图。
        layer_name: 当前 Attention 的完整层名。
        kv_cache_dummy_dep: 独立缓存更新返回的空 tensor，保留 vLLM 接口命名。

    Side Effects:
        impl.forward 写入 output；对于内部包含缓存更新的后端，也可修改缓存。
        不以 impl.forward 的返回值替换包装层已经分配的输出。
    """
    del kv_cache_dummy_dep  # 仅保留依赖接口；直接调用路径已经完成缓存更新。
    attn_metadata, attn_layer, kv_cache, _ = get_attention_context(layer_name)
    attn_layer.impl.forward(
        attn_layer, query, key, value, kv_cache, attn_metadata, output=output
    )
