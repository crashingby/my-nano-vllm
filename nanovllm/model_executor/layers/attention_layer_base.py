from abc import ABC, abstractmethod

from nanovllm.config import NanovllmConfig
from nanovllm.attention.backend import AttentionBackend, AttentionImpl
from nanovllm.kv_cache_interface import KVCacheSpec


class AttentionLayerBase(ABC):
    """为支持 V1 Engine 的 Attention 类层提供统一发现接口。

    Worker 通过本基类读取 Attention 后端与每层 KV cache 规格，不需要
    判断具体模型层类型。接口只描述层能力，不负责执行 Attention 计算。

    Attributes:
        impl: 当前层持有的后端实现；由具体层在需要时创建和填充。
    """

    impl: "AttentionImpl"

    @abstractmethod
    def get_attn_backend(self) -> type[AttentionBackend]:
        """返回本层使用的 Attention 后端类。

        Returns:
            attn_backend: 用于构建 Attention metadata 和执行后端的类。
        """
        pass

    @abstractmethod
    def get_kv_cache_spec(self, vllm_config: NanovllmConfig) -> KVCacheSpec | None:
        """返回本层所需的 KV cache 规格。

        Args:
            vllm_config: 提供 block size、模型精度等规划参数的总配置。

        Returns:
            kv_cache_spec: 本层的 KVCacheSpec；不需要 KV cache 的层返回 None。
        """
        pass