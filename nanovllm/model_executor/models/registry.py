# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""内置文本模型的最小延迟注册表，不包含插件或跨进程模型能力探测。"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from nanovllm.config import ModelConfig


@dataclass(frozen=True)
class _LazyRegisteredModel:
    """记录模型的导入位置，注册时不导入模型实现。

    Attributes:
        module_name: Python 模块路径，例如 nanovllm.model_executor.models.qwen3。
        class_name: 该模块导出的模型类名称。
    """

    module_name: str
    class_name: str

    def load_model_cls(self) -> type[nn.Module]:
        """导入模块并返回模型类，不构造实例。

        Returns:
            对应的 nn.Module 子类。

        Side Effects:
            首次调用会执行模型模块的导入代码；导入错误直接向上传播。
        """
        # mod 是实际 Python 模块；importlib 复用已导入模块，不重复构造模型。
        mod = importlib.import_module(self.module_name)
        return getattr(mod, self.class_name)


@dataclass
class _ModelRegistry:
    """按 HF 架构名称查找内置模型类。

    Attributes:
        models: 架构名称到延迟导入描述的映射；不保存模型实例。
    """

    models: dict[str, _LazyRegisteredModel] = field(default_factory=dict)

    def _try_load_model_cls(self, model_arch: str) -> type[nn.Module] | None:
        """尝试加载指定架构，不支持时返回 None。

        Args:
            model_arch: HF config.json 中的架构名称。

        Returns:
            注册的模型类；没有注册该名称时为 None。
        """
        if model_arch not in self.models:
            return None
        return self.models[model_arch].load_model_cls()

    def resolve_model_cls(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> tuple[type[nn.Module], str]:
        """按候选顺序返回第一个已注册的模型实现。

        Args:
            architectures: 单个架构名称或按优先级排列的候选名称列表。
            model_config: 保留 vLLM 调用协议；当前没有依赖配置的后端回退分支。

        Returns:
            模型类及命中的候选架构名称。

        Raises:
            ValueError: 没有候选架构，或所有候选都没有本地实现。
        """
        # 统一为列表，使字符串与列表输入遵循相同的选择顺序。
        if isinstance(architectures, str):
            architectures = [architectures]
        if not architectures:
            raise ValueError("No model architectures are specified")
        # arch 是候选名称；model_cls 是延迟导入得到的类，None 表示未注册。
        for arch in architectures:
            model_cls = self._try_load_model_cls(arch)
            if model_cls is not None:
                return model_cls, arch
        raise ValueError(
            f"Model architectures {architectures} are not supported. "
            f"Supported architectures: {list(self.models)}"
        )


# 与 vLLM 一样导出进程内注册表实例，目前只登记已有 Qwen3 实现的导入位置。
ModelRegistry = _ModelRegistry(
    {
        "Qwen3ForCausalLM": _LazyRegisteredModel(
            module_name="nanovllm.model_executor.models.qwen3",
            class_name="Qwen3ForCausalLM",
        ),
    }
)
