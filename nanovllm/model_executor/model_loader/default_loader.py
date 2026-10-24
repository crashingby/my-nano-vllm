# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""默认权重加载器：仅保留本地 safetensors 的同步加载路径。"""

from collections.abc import Generator, Iterable
import dataclasses
import glob
import os
import time
from typing import cast

import torch
from torch import nn
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from nanovllm.config import ModelConfig
from nanovllm.config.load import LoadConfig
from nanovllm.logger import init_logger
from nanovllm.model_executor.model_loader.base_loader import BaseModelLoader
from nanovllm.model_executor.model_loader.weight_utils import (
    filter_duplicate_safetensors_files,
    safetensors_weights_iterator,
)

logger = init_logger(__name__)


class DefaultModelLoader(BaseModelLoader):
    """按 vLLM 的 Source -> iterator -> model.load_weights 流程装填权重。

    本类只准备文件和检查结果，不实现模型参数映射或 TP 切片。
    当前没有远程下载、PT 回退、多线程/预取、量化、EP 权重过滤。

    Attributes:
        load_config: 基类保存的加载策略引用。
        counter_before_loading_weights: 本轮首次建立权重迭代器的 perf_counter 时间。
        counter_after_loading_weights: 模型消费完迭代器后的 perf_counter 时间。
            两者差值为权重加载耗时，不包含 BaseModelLoader 中的模型构造。
    """

    @dataclasses.dataclass
    class Source:
        """一个本地权重来源，保留 vLLM 与文件格式无关的源描述字段。

        Attributes:
            model_or_path: 本地 checkpoint 目录，不接受 Hub 模型 ID。
            subfolder: 该目录中的可选权重子目录。
            prefix: 产出权重名称前添加的前缀，不改变 tensor 内容。
            allow_patterns_overrides: 可选文件 glob 模式；只接受匹配到的 safetensors。
                与 vLLM 一样按模式顺序选择第一个非空集合，而不是合并所有模式。
        """

        model_or_path: str
        subfolder: str | None = None
        prefix: str = ""
        allow_patterns_overrides: list[str] | None = None

    counter_before_loading_weights: float = 0.0
    counter_after_loading_weights: float = 0.0

    def __init__(self, load_config: LoadConfig) -> None:
        """保存加载策略并限制为当前支持的格式。

        Args:
            load_config: auto/hf/safetensors 均选择本地 safetensors 路径。

        Raises:
            ValueError: 请求了 PT、量化等尚未实现的加载格式。
        """
        super().__init__(load_config)
        # 直接构造 Loader 也要满足格式约束，不能绕过工厂后静默使用错误格式。
        if load_config.load_format not in ("auto", "hf", "safetensors"):
            raise ValueError(f"Unsupported load format: {load_config.load_format}")

    def _prepare_weights(
        self,
        model_name_or_path: str,
        subfolder: str | None,
        allow_patterns_overrides: list[str] | None,
    ) -> tuple[str, list[str], bool]:
        """查找本地文件，并通过分片索引排除重复 checkpoint。

        Args:
            model_name_or_path: 本地模型目录。
            subfolder: 可选权重子目录。
            allow_patterns_overrides: 替代默认 *.safetensors 的匹配模式。

        Returns:
            (hf_folder, hf_weights_files, use_safetensors):
                hf_folder: 本地权重所在目录。
                hf_weights_files: 已筛选并排序的权重文件路径列表，尚未加载 tensor。
                use_safetensors: 文件采用 safetensors 时为 True，当前实现始终为 True。

        Raises:
            ValueError: 模型目录不存在，或传入远程模型 ID。
            RuntimeError: 没有找到有效的 safetensors 文件。
            FileNotFoundError: 本地索引引用了缺失的分片。
        """
        # 没有远程下载分支，路径不是目录时立即失败，不调用任何 Hub 接口。
        if not os.path.isdir(model_name_or_path):
            raise ValueError(
                f"Only local model directories are supported: {model_name_or_path}"
            )
        # 保留 vLLM 变量名：hf_folder 是本地 checkpoint 根目录，不表示远程仓库。
        hf_folder = model_name_or_path
        if subfolder is not None:
            hf_folder = os.path.join(hf_folder, subfolder)
        # 当前固定使用标准 safetensors 索引，不自动探测 Mistral 专有文件布局。
        index_file = SAFE_WEIGHTS_INDEX_NAME
        use_safetensors = True
        allow_patterns = ["*.safetensors"]
        if allow_patterns_overrides is not None:
            allow_patterns = allow_patterns_overrides

        # 按模式优先级选择文件；即使覆盖模式过宽，也不加载 .bin/.pt 或目录。
        hf_weights_files: list[str] = []
        for pattern in allow_patterns:
            hf_weights_files = [
                filename
                for filename in glob.glob(os.path.join(hf_folder, pattern))
                if filename.endswith(".safetensors") and os.path.isfile(filename)
            ]
            if hf_weights_files:
                break

        # 有索引时只保留其引用文件，防止单文件与分片版本同时被加载。
        hf_weights_files = filter_duplicate_safetensors_files(
            hf_weights_files, hf_folder, index_file
        )
        if not hf_weights_files:
            raise RuntimeError(
                f"Cannot find any safetensors model weights with {hf_folder!r}"
            )
        return hf_folder, hf_weights_files, use_safetensors

    def _get_weights_iterator(
        self, source: Source
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """为一个来源建立惰性权重迭代器，并为权重名称添加前缀。

        Args:
            source: 本地文件位置与名称前缀。

        Returns:
            weights_iterator: 依次产出 (name, tensor) 的生成器；name 带 source.prefix，
                tensor 是对应的 CPU 权重。

        Side Effects:
            扫描文件与索引，并在第一次调用时记录加载起点。真正读取 tensor
            发生在调用方消费生成器时，不一次性保留整个模型的权重字典。
        """
        # _prepare_weights 返回 (hf_folder, hf_weights_files, use_safetensors)，这里仅使用文件列表。
        _, hf_weights_files, _ = self._prepare_weights(
            source.model_or_path,
            source.subfolder,
            source.allow_patterns_overrides,
        )
        weights_iterator = safetensors_weights_iterator(
            hf_weights_files, self.load_config.use_tqdm_on_load
        )
        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        # name/tensor 是单个 checkpoint 条目；prefix 只改变名称，模型负责解释名称。
        return ((source.prefix + name, tensor) for name, tensor in weights_iterator)

    def get_all_weights(
        self, model_config: ModelConfig, model: nn.Module
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """按顺序产出主权重源以及模型声明的附加权重源。

        Args:
            model_config: 指定主模型的本地目录。
            model: 可通过 allow_patterns_overrides 和 secondary_weights 定制来源。

        Yields:
            (name, tensor):
                name: 带 Source.prefix 的 checkpoint 权重名称。
                tensor: 对应的 CPU 权重 tensor，当前不涉及下载或 EP 筛选。

        Side Effects:
            消费生成器时打开本地权重文件。
        """
        # 主来源使用模型目录，模型可限定自身所需的文件模式。
        primary_weights = DefaultModelLoader.Source(
            model_config.model,
            prefix="",
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
        )
        yield from self._get_weights_iterator(primary_weights)

        # 保留 vLLM 的附加来源协议；普通 Qwen 没有该成员，默认不追加任何来源。
        secondary_weights = cast(
            Iterable[DefaultModelLoader.Source],
            getattr(model, "secondary_weights", ()),
        )
        for source in secondary_weights:
            yield from self._get_weights_iterator(source)

    def download_model(self, model_config: ModelConfig) -> None:
        """保留基类预准备接口，但只检查本地权重，绝不下载。

        Args:
            model_config: 指定本地模型目录的配置。

        Returns:
            无返回值；路径或文件不合法时由 _prepare_weights 抛出异常。
        """
        self._prepare_weights(
            model_name_or_path=model_config.model,
            subfolder=None,
            allow_patterns_overrides=None,
        )

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """让模型消费权重迭代器，然后检查应加载参数是否有遗漏。

        Args:
            model: 必须提供 load_weights(iterable) 接口的模型。
            model_config: 指定本地 checkpoint 目录的模型配置。

        Side Effects:
            原地装填模型参数、更新时间统计并输出日志。失败不会回滚参数。

        Raises:
            ValueError: 模型返回的已加载参数名集合缺少必需参数。
        """
        # 同一 Loader 可以再次加载权重，每次重新计时而不是累计首次加载以来的时间。
        self.counter_before_loading_weights = 0.0
        self.counter_after_loading_weights = 0.0
        # loaded_weights 应是模型参数名称集合，而不是 checkpoint 的原始名称。
        loaded_weights = model.load_weights(self.get_all_weights(model_config, model))
        self.counter_after_loading_weights = time.perf_counter()
        logger.info_once(
            "Loading weights took %.2f seconds",
            self.counter_after_loading_weights - self.counter_before_loading_weights,
        )
        # 无量化路径无需额外追踪开关；沿用 vLLM 的 None 表示模型不报告加载集合。
        if loaded_weights is not None:
            self.track_weights_loading(model, loaded_weights)

    def track_weights_loading(
        self, model: nn.Module, loaded_weights: set[str] | None
    ) -> None:
        """检查模型报告的已加载参数名称是否覆盖所有参数。

        Args:
            model: 已装填权重的模型，共享的同一个 Parameter 按 named_parameters 去重。
            loaded_weights: 模型返回的参数名集合；为 None 时不执行集合检查。

        Raises:
            ValueError: 仍有参数没有被模型报告为已加载。

        Notes:
            这是参数名级别的覆盖检查，不检查数值，也不单独检查融合参数的每个分片。
        """
        # weights_to_load 是模型参数集合，不包括 buffer 或 checkpoint 独有的条目。
        weights_to_load = {name for name, _ in model.named_parameters()}
        if loaded_weights is not None:
            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                raise ValueError(
                    "Following weights were not initialized from "
                    f"checkpoint: {weights_not_loaded}"
                )
