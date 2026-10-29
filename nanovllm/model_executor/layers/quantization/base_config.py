# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""沿用 vLLM 的权重方法与量化配置接口，供普通精度及后续量化实现共用。

这里只定义配置和执行协议，不解析具体 checkpoint 量化格式，也不注册算法。
当前加载器支持参数创建、checkpoint 装填、加载后处理和执行这条同步路径；
名称映射通过自动加载器统一应用；在线量化与预处理权重缓存尚未接入。
"""

from abc import ABC, abstractmethod
import re
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from nanovllm.model_executor.models.utils import WeightsMapper


class QuantizeMethodBase(ABC):
    """描述一层参数的存储和执行方式，普通精度方法也遵守此协议。

    层构造时调用 create_weights，Loader 装填后调用 process_weights_after_loading，
    推理时调用 apply。基类名称沿用 vLLM，继承它本身不会执行量化。

    Attributes:
        requires_device_loading: 后处理是否要求参数位于目标执行设备；Loader 据此
            决定是否进入临时设备搬运上下文。仅校验元信息的方法可设为 False。
    """

    requires_device_loading: bool = True

    @abstractmethod
    def create_weights(
        self,
        layer: nn.Module,
        *weight_args: Any,
        **extra_weight_attrs: Any,
    ) -> None:
        """为 layer 注册参数，具体形状及加载元信息由层传入。

        Side Effects:
            创建未初始化的参数；checkpoint 装填完成前不能用于推理。
        """
        raise NotImplementedError

    @abstractmethod
    def apply(self, layer: nn.Module, *args: Any, **kwargs: Any) -> torch.Tensor:
        """使用 layer 已加载的参数执行计算。

        Returns:
            output: 当前方法计算的本地输出，TP 通信由外层模型层负责。
        """
        raise NotImplementedError

    def embedding(self, layer: nn.Module, *args: Any, **kwargs: Any) -> torch.Tensor:
        """声明可选查表接口，支持 embedding 的方法需要自行实现。

        Returns:
            output: 子类按 token ID 查得的 embedding；基类调用抛出异常。
        """
        raise NotImplementedError

    def tie_weights(self, layer: nn.Module, embed_tokens: nn.Module) -> nn.Module:
        """让输出层共享输入 embedding 的同一个权重参数。

        Returns:
            layer: 完成参数共享的原输出层。

        Side Effects:
            替换 layer.weight；特殊打包布局的方法须重写此接口。
        """
        layer.weight = embed_tokens.weight
        return layer

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """权重装填后可原地重排或预处理参数；默认保留原布局。"""
        return

    def dequantize_weight(
        self,
        layer: nn.Module,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """声明可选反量化接口，基类调用明确报错。

        Returns:
            weight: 子类还原的浮点权重或权重二元组，基类不提供默认转换。
        """
        raise NotImplementedError(f"{type(self).__name__} does not support dequantization")


class QuantizationConfig(ABC):
    """描述某种量化方案的约束，并按层类型和 prefix 选择执行方法。

    NanovllmConfig.quant_config 保存具体配置实例，Qwen 将它传给各线性层。
    当前没有内置量化算法或按名称构造配置的注册表；None 选择普通精度方法。
    添加具体算法时，配置子类还需提供对应的参数存储、加载与执行实现。

    Attributes:
        packed_modules_mapping: 融合模块短名称到原始模块短名称列表的映射，
            例如 {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}。模型构造时更新，
            用于将针对原始投影的配置匹配到融合层，不负责实际权重切分。
    """

    # 自动加载器只忽略模型未创建的这些参数，存在的参数仍正常装填。
    _ignore_unexpected_suffixes = (
        ".q_scale",
        ".k_scale",
        ".v_scale",
        ".q_zero_point",
        ".k_zero_point",
        ".v_zero_point",
    )

    def __init__(self) -> None:
        """为当前配置创建独立的融合模块映射。"""
        self.packed_modules_mapping: dict[str, list[str]] = {}

    @abstractmethod
    def get_name(self) -> str:
        """返回量化方案标识。

        Returns:
            quantization_method: 具体配置实现使用的方案名称。
        """
        raise NotImplementedError

    @abstractmethod
    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        """声明方法支持的输入激活类型。

        Returns:
            act_dtypes: 具体算法能够处理的浮点输入类型列表。
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int:
        """返回最低 CUDA 计算能力，以主版本乘十加次版本表示。

        Returns:
            capability: 例如 80 表示 SM 8.0。
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_config_filenames() -> list[str]:
        """声明具体算法使用的配置文件名；当前 Loader 不自动发现这些文件。

        Returns:
            filenames: 可供未来配置解析入口查找的文件名列表。
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict[str, Any]) -> "QuantizationConfig":
        """由算法配置字典构造具体实例。

        Returns:
            quant_config: 已解析配置的具体 QuantizationConfig 子类实例。
        """
        raise NotImplementedError

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> str | None:
        """为未来的算法识别保留覆盖入口，默认不提出替代方案。

        Returns:
            quantization_method: 建议替换的方案名称；默认 None 表示不覆盖。
        """
        return None

    @staticmethod
    def get_from_keys(config: dict[str, Any], keys: list[str]) -> Any:
        """按优先级查找算法配置中可能使用的不同字段名。

        Returns:
            value: 第一个存在的字段值，保留显式的 None。

        Raises:
            ValueError: 所有候选字段均不存在。
        """
        for key in keys:
            if key in config:
                return config[key]
        raise ValueError(f"Cannot find any of {keys} in the model's quantization config")

    @staticmethod
    def get_from_keys_or(config: dict[str, Any], keys: list[str], default: Any) -> Any:
        """读取可选配置字段，不存在时使用默认值。

        Returns:
            value: 第一个存在的字段值，或 default。
        """
        try:
            return QuantizationConfig.get_from_keys(config, keys)
        except ValueError:
            return default

    @abstractmethod
    def get_quant_method(self, layer: nn.Module, prefix: str) -> QuantizeMethodBase | None:
        """根据层及完整路径选择方法；线性层要求返回一个有效方法。

        返回 UnquantizedLinearMethod 可让指定线性层保留普通精度；None 表示
        不支持该层，LinearBase 会明确报错，不会静默改变用户选择。

        Returns:
            quant_method: 对应层的执行方法；不适用时为 None。
        """
        raise NotImplementedError

    @staticmethod
    def get_cache_scale_mapper() -> "WeightsMapper":
        """提供 checkpoint KV scale 到 Attention 参数名称的上游映射规则。

        Returns:
            mapper: AutoWeightsLoader 自动合并的映射器，具体缓存量化与 scale
                参数仍由对应 Attention 实现提供，本接口本身不创建这些参数。
        """
        from nanovllm.model_executor.models.utils import WeightsMapper

        orig_to_new_regex = {
            re.compile(r"\.kv_scale$"): r".attn.k_scale",
            re.compile(r"\.self_attn\.[kv]_proj\.([kv])_scale$"): (
                r".self_attn.attn.\1_scale"
            ),
            re.compile(r"\.self_attn\.qk(?:qk)?v_proj\.([kv])_scale$"): (
                r".self_attn.attn.\1_scale"
            ),
            re.compile(r"\.mixer\.[kv]_proj\.([kv])_scale$"): r".mixer.attn.\1_scale",
            re.compile(r"\.self_attn\.q\.scale$"): r".self_attn.attn.q_scale",
            re.compile(r"\.self_attn\.([kv])_cache\.scale$"): (
                r".self_attn.attn.\1_scale"
            ),
            # 已含 .attn 的标准名称不再重复添加路径。
            re.compile(r"(?<!\.attn)\.([qkv])_scale$"): r".attn.\1_scale",
            re.compile(r"(?<!\.attn)\.([qkv])_zero_point$"): r".attn.\1_zero_point",
        }
        return WeightsMapper(orig_to_new_regex=orig_to_new_regex)

    @staticmethod
    def get_checkpoint_weight_mapper() -> "WeightsMapper":
        """丢弃上游已支持计算路径不使用的 activation-order 元数据。

        Returns:
            mapper: 将 .g_idx 后缀条目映射为 None 的过滤规则。
        """
        from nanovllm.model_executor.models.utils import WeightsMapper

        return WeightsMapper(orig_to_new_suffix={".g_idx": None})
