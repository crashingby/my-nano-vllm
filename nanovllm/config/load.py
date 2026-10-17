"""模型权重加载策略配置，不负责模型构造或 checkpoint 文件读取。"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

# 这些类型只供静态检查，运行时不导入 model_executor，避免加载器工厂循环依赖。
if TYPE_CHECKING:
    from nanovllm.model_executor.model_loader import LoadFormats
    from nanovllm.model_executor.model_loader.tensorizer import TensorizerConfig
else:
    LoadFormats = Any
    TensorizerConfig = Any


@dataclass
class LoadConfig:
    """描述权重加载方式，供后续模型加载器选择和执行加载流程。

    模型目录由 ModelConfig.model 指定，模型结构与计算 dtype 也不归本类管理。
    本类仅保存选项，不会下载模型、读取权重或改变当前 CUDA device。
    当前 DefaultModelLoader 仅支持本地 safetensors，不包含 PT 回退或远程下载。

    Attributes:
        load_format: 权重加载格式或加载器选择标识。当前 auto、hf、safetensors
            都选择本地 safetensors 路径；不同于完整 vLLM，auto/hf 不回退到 PT。
            没有 safetensors 文件时直接报错，也不会从 Hub 下载。
        use_tqdm_on_load: 是否在加载 checkpoint 时显示进度条，由 Loader 消费；
            不控制推理生成阶段的进度条。
        device: 可选的权重加载目标设备，例如 cuda:0。None 表示沿用
            NanovllmConfig.device_config.device；未指定设备编号时使用 Worker
            绑定的当前 CUDA 设备。它不是 TP rank 配置，也不是 safetensors 读取时的
            CPU 暂存位置。BaseModelLoader 在模型构造阶段应用该设备覆盖。
    """

    # 保留与 vLLM 一致的选择入口；接受哪些格式由实际加载器决定。
    load_format: str | LoadFormats = "auto"
    # 文件加载进度的显示策略，不参与模型结构或权重数值计算。
    use_tqdm_on_load: bool = True
    # 默认不覆盖 Worker 的设备选择，避免在配置中写死某个 TP rank 的 GPU。
    device: str | None = None
