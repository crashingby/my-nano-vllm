"""早期加载入口的兼容包装，复用模型当前的映射与自动分派协议。"""

import os
from glob import glob

from torch import nn

from nanovllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    safetensors_weights_iterator,
)
from nanovllm.model_executor.models.utils import AutoWeightsLoader

__all__ = ["default_weight_loader", "load_model"]


def load_model(model: nn.Module, path: str) -> None:
    """读取本地 safetensors，交给模型加载入口或通用自动加载器。

    Args:
        model: 已构造参数的模型；有 load_weights 时由该接口解释名称与融合关系。
        path: 含一个或多个 safetensors 文件的本地目录，不下载远端权重。

    Side Effects:
        惰性读取 CPU tensor 并原地装填模型，不再单独解释 packed_modules_mapping，
        避免兼容入口继续依赖旧映射结构。此包装不执行后处理与缺失参数检查。
    """
    weights = safetensors_weights_iterator(
        glob(os.path.join(path, "*.safetensors")), use_tqdm_on_load=False,
    )
    model_load_weights = getattr(model, "load_weights", None)
    if callable(model_load_weights):
        model_load_weights(weights)
    else:
        loader = AutoWeightsLoader(model)
        loader.load_weights(weights, mapper=getattr(model, "hf_to_vllm_mapper", None))
