from nanovllm.config import (
    NanovllmConfig,
    get_layers_from_nanovllm_config,
)
from nanovllm.kv_cache_interface import KVCacheSpec
from nanovllm.model_executor.layers.attention import Attention


def get_kv_cache_spec(
    vllm_config: NanovllmConfig,
) -> dict[str, KVCacheSpec]:
    """收集当前 Worker 模型中所有需要独立 KV cache 的 Attention 规格。

    调用时机在模型构造完成之后。每个 Attention 已在构造期间写入
    ``vllm_config.compilation_config.static_forward_context``；本函数不遍历
    ``model.named_modules()``，而是直接读取该静态注册表。

    Args:
        vllm_config: 当前 Worker 的总配置及其静态 forward context。

    Returns:
        从 Attention 完整层名到该层 KVCacheSpec 的映射。对于 Qwen3，key 形如
        ``"model.layers.0.self_attn.attn"``。

    Notes:
        当前 MVP 的所有 Qwen3 Attention 都是 full attention，因此每个注册层都会
        返回一个 ``FullAttentionSpec``。未来若加入不需要 KV cache 的 Attention，
        可在这里跳过其 ``get_kv_cache_spec()`` 返回 ``None`` 的情形。
    """
    kv_cache_spec:dict[str,KVCacheSpec]={}
    attn_layers = get_layers_from_nanovllm_config(vllm_config, Attention)
    # TODO:以后好像是通过config直接获得层的名字
    for layer_name,module in attn_layers.items():
        # 我们现在暂时只支持Attention里的KV Cache
        if not isinstance(module,Attention):
            continue
        kv_cache_spec[layer_name]=module.get_kv_cache_spec(vllm_config)
    return kv_cache_spec
