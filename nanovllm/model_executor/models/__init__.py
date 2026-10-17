"""模型注册表公共入口；导入本包不会立即导入 Qwen 模型实现。"""

from nanovllm.model_executor.models.registry import ModelRegistry

__all__ = ["ModelRegistry"]
