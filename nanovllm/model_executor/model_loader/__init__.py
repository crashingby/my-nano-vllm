"""模型加载器工厂；当前只提供本地 safetensors 加载。"""

from typing import Literal

from nanovllm.config.load import LoadConfig
from nanovllm.model_executor.model_loader.base_loader import BaseModelLoader
from nanovllm.model_executor.model_loader.default_loader import DefaultModelLoader

# auto/hf 保留入口名称，但当前不包含 PT 回退或远程下载。
LoadFormats = Literal["auto", "hf", "safetensors"]

_LOAD_FORMAT_TO_MODEL_LOADER: dict[str, type[BaseModelLoader]] = {
    "auto": DefaultModelLoader,
    "hf": DefaultModelLoader,
    "safetensors": DefaultModelLoader,
}


def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    """根据加载格式构造对应 Loader，不立即读取权重。

    Args:
        load_config: 描述加载格式、进度条与设备覆盖选项的配置。

    Returns:
        model_loader: 持有该配置引用的 Loader 实例。

    Raises:
        ValueError: 格式不在当前支持列表中。
    """
    # 与 vLLM 一样通过映射选择实现，后续添加格式不应修改 Runner。
    load_format = load_config.load_format
    if load_format not in _LOAD_FORMAT_TO_MODEL_LOADER:
        raise ValueError(f"Load format {load_format!r} is not supported")
    return _LOAD_FORMAT_TO_MODEL_LOADER[load_format](load_config)
