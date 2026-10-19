"""nano-vLLM 的配置公共入口。

调用方从这里导入各子配置，组装 NanovllmConfig 后传给 LLM 或 LLMEngine。
子配置负责各自的参数解析，总配置负责跨配置约束和 Graph 尺寸初始化；Engine
直接持有传入对象，Executor、Worker 和 Runner 继续共享这一配置引用。

配置构造不会启动 Worker 或加载权重，但 ModelConfig 会读取本地 HF 结构配置。
EOS、实际 KV block 数和编译统计等字段，则在 Engine 启动过程中回写。
"""

from nanovllm.config.cache import CacheConfig
from nanovllm.config.compilation import CompilationConfig,CUDAGraphMode
from nanovllm.config.device import DeviceConfig
from nanovllm.config.kernel import KernelConfig
from nanovllm.config.load import LoadConfig
from nanovllm.config.model import ModelConfig
from nanovllm.config.parallel import ParallelConfig
from nanovllm.config.scheduler import SchedulerConfig
from nanovllm.config.vllm import (
    NanovllmConfig,
    get_current_config,
    get_current_config_or_none,
    get_layers_from_nanovllm_config,
    set_current_config,
)

__all__ = [
    "CacheConfig",
    "CompilationConfig",
    "CUDAGraphMode",
    "DeviceConfig",
    "KernelConfig",
    "LoadConfig",
    "ModelConfig",
    "NanovllmConfig",
    "ParallelConfig",
    "SchedulerConfig",
    "get_current_config",
    "get_current_config_or_none",
    "get_layers_from_nanovllm_config",
    "set_current_config",
]
