# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""模型加载器基类：统一构造模型、加载权重与权重后处理的顺序。

本模块沿用 vLLM 的加载主流程，不包含 checkpoint 格式解析、模型架构选择、
在线量化或 CPU offloading。具体文件读取由子类负责，模型构造由独立工具负责。
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from nanovllm.config import ModelConfig, NanovllmConfig
from nanovllm.config.load import LoadConfig
from nanovllm.logger import init_logger

from nanovllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from nanovllm.utils.mem_utils import format_gib

from nanovllm.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)


class BaseModelLoader(ABC):
    """定义模型加载器的公共接口和与权重文件格式无关的加载流程。

    Runner 通过具体 Loader 调用 load_model()，取得已加载权重且处于 eval 模式的
    模型。子类实现 download_model() 和 load_weights()，不必重复模型构造流程。

    本类不初始化分布式通信、不分配请求的 KV cache，也不统计模型加载显存增量；
    这些职责分别属于 Worker、KV 初始化链条和 Runner 的加载阶段显存统计。

    Attributes:
        load_config: 创建此 Loader 时传入的加载配置引用，不复制配置。
            工厂与调用方应使用同一个 NanovllmConfig.load_config，避免加载器选择
            与实际执行时的设备、文件格式等选项不一致。
    """

    def __init__(self, load_config: LoadConfig) -> None:
        """保存加载器配置；此时不构造模型、不读取文件。

        Args:
            load_config: 控制权重加载格式、进度条以及可选目标设备的配置。
        """
        # 子类的文件读取与进度显示也使用这一配置，不另建独立副本。
        self.load_config: LoadConfig = load_config

    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None:
        """准备后续加载所需的 checkpoint 文件。

        保留 vLLM 的独立预下载接口。当前只支持本地模型，具体子类可以检查本地
        文件是否已准备好，而不访问网络；load_model() 不会自动调用本接口，
        子类的 load_weights() 仍需能够独立找到并读取所需文件。

        Args:
            model_config: 包含模型目录等信息的模型配置。

        Returns:
            无返回值；准备好的文件由后续加载流程读取。

        Side Effects:
            由子类决定。未来支持远程模型时可能下载文件并更新本地缓存；当前
            本地加载器不应仅因调用此接口就发起网络请求。
        """
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """将 checkpoint 权重原地写入已经构造好的模型。

        子类负责读取对应文件格式，并调用模型的权重加载接口完成参数名称映射、
        融合参数装填与 TP 分片。此方法只加载权重，不重新构造模型；
        独立调用它并不自动执行 load_model() 中的后处理或 eval()。

        Args:
            model: 参数存储已创建的模型，通常已经位于加载目标设备。
            model_config: 指定权重来源目录及模型结构的配置。

        Returns:
            无返回值；通过修改 model 的参数交付结果。

        Side Effects:
            读取 checkpoint 文件并覆盖模型参数，可能产生临时 CPU/GPU tensor。
            出错时异常向上传递，不保证已写入的参数恢复为加载前状态。
        """
        raise NotImplementedError

    def load_model(
        self,
        vllm_config: NanovllmConfig,
        model_config: ModelConfig,
        prefix: str = "",
    ) -> nn.Module:
        """构造模型、加载权重、执行后处理，并返回推理模式的模型。

        顺序与 vLLM 的 BaseModelLoader 保持一致。默认采用 DeviceConfig.device，
        显式 LoadConfig.device 可以覆盖加载目标。未指定 CUDA index 时使用
        Worker 已绑定的设备；参数精度统一由 ModelConfig.dtype 提供。
        TODO DeviceConfig: 基础配置已接入，跨平台探测后续扩展。

        Args:
            vllm_config: 当前 Worker 的总配置；沿用 vLLM 的参数命名以便对照源码。
                其中 load_config 应与创建本 Loader 时传入的配置一致。
            model_config: 本次加载的模型配置，提供结构、权重路径和计算 dtype。
            prefix: 模型在静态层注册表中的名称前缀，空字符串表示顶层模型。
                initialize_model() 负责将它与当前配置传给模型构造路径。

        Returns:
            model: 权重加载与后处理完成的 nn.Module，所有子模块均已切换到 eval 模式。
                eval() 不等同于 inference_mode()；执行推理时仍由调用方控制梯度模式。

        Side Effects:
            构造模型参数并分配设备内存；initialize_model() 应在配置上下文中
            完成 Attention 层注册。随后读取并写入权重，执行实现特有的后处理。
            临时 dtype 与设备上下文退出后恢复；失败不回滚已注册的层或已写参数。
        """
        # 执行设备与加载设备覆盖选项分开保存；无覆盖时沿用统一执行设备配置。
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = (
            device_config.device if load_config.device is None else load_config.device
        )
        # 将设备字符串转换为上下文对象，作用于模型构造中的默认 tensor 分配。
        target_device = torch.device(load_device)
        # 与当前 Runner 及 AttentionSpec 使用同一 dtype 来源，避免参数精度不一致。
        model_dtype = model_config.dtype

        # 覆盖构造与加载后处理阶段；工具必须在异常路径上也恢复原默认 dtype。
        with set_default_torch_dtype(model_dtype):
            # 仅模型构造采用目标设备上下文，文件迭代器仍可显式在 CPU 读取权重。
            with target_device:
                # 架构选择和静态层注册由 initialize_model 负责，基类不识别 Qwen。
                model = initialize_model(
                    vllm_config=vllm_config,
                    model_config=model_config,
                    prefix=prefix,
                )

            logger.debug("Loading weights on %s ...", load_device)
            # 子类原地填充已有参数；基类不重复实现 safetensors 或 TP 的加载细节。
            self.load_weights(model, model_config)

            # 保留 vLLM 的 CUDA 峰值调试日志；CPU 加载不查询 CUDA 分配器。
            if target_device.type == "cuda":
                # 单位为字节，是该设备最近一次重置以来的峰值，不是加载显存增量。
                peak_memory = torch.cuda.max_memory_allocated(target_device)
                logger.debug(
                    "Peak GPU memory after loading weights: %s GiB",
                    format_gib(peak_memory),
                )

            # 保留后处理阶段以衔接 AttentionImpl 等实现；不在这里加入在线量化分支。
            process_weights_after_loading(model, model_config, target_device)

        # 在所有权重操作成功后才交付模型，避免调用方收到未完成加载的对象。
        return model.eval()
