# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""模型选择、构造与权重后处理工具，统一权重方法和 Attention 的加载后处理。"""

from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
import inspect
from typing import Any
import warnings

import torch
from torch import nn

from nanovllm.config import ModelConfig, NanovllmConfig, set_current_config
from nanovllm.model_executor.layers.attention import Attention
from nanovllm.model_executor.layers.quantization import QuantizeMethodBase
from nanovllm.logger import init_logger

logger = init_logger(__name__)


def initialize_model(
    vllm_config: NanovllmConfig,
    *,
    prefix: str = "",
    model_class: type[nn.Module] | None = None,
    model_config: ModelConfig | None = None,
) -> nn.Module:
    """选择模型类并在当前配置上下文中构造模型，不读取 checkpoint。

    沿用 vLLM 的新旧构造接口兼容流程。新接口接收 vllm_config 和 prefix；
    当前 Qwen3 使用新接口；旧接口分支仅为接收 HF config 的模型保留。
    dtype 和默认分配设备由外层 BaseModelLoader 的上下文决定。

    Args:
        vllm_config: 当前 Worker 的总配置，也是 Attention 静态注册表的所有者。
        prefix: 转交给支持该参数的模型构造函数的层名前缀。
            旧模型若不接收 prefix，本工具不会替它改写内部层名。
        model_class: 显式模型类；为 None 时根据 HF architectures 查询注册表。
        model_config: 模型选择及旧接口使用的配置，默认取总配置中的 model_config。
            新式构造函数仍从 vllm_config 读取配置，调用方应保持两者一致。

    Returns:
        model: 已构造但尚未加载权重的 nn.Module；此处不调用 eval()。

    Side Effects:
        分配模型参数，模型中的 Attention 可向当前配置注册层名。正常或异常
        退出都会恢复外层当前配置；构造失败不会回滚已经写入的静态注册表。
    """
    # 显式模型配置允许调用方选择构造目标；普通路径使用总配置中的模型配置。
    if model_config is None:
        model_config = vllm_config.model_config
    if model_class is None:
        # 架构名用于选择实现，构造阶段这里只需要实际 Python 类。
        model_class, _ = get_model_architecture(model_config)

    # 检查构造参数以区分 vLLM 新式接口与仅接收 HF config 的旧式接口。
    signatures = inspect.signature(model_class.__init__)
    all_params = [param.name for param in signatures.parameters.values()]
    if "vllm_config" in all_params and "prefix" in all_params:
        # 本地配置上下文暂不支持 check_compile/prefix，prefix 仍显式传入模型。
        with set_current_config(vllm_config):
            model = model_class(vllm_config=vllm_config, prefix=prefix)
            return model

    # 保留 vLLM 的兼容提示，便于以后将模型构造统一到新式接口。
    msg = (
        "Model classes should accept vllm_config and prefix. "
        f"Using the legacy constructor for {model_class.__name__}."
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    logger.warning(
        "Trying to guess the arguments for old-style model class %s", model_class
    )
    # 只传构造函数明确声明且本项目已支持的配置，避免凭空加入新 feature 参数。
    kwargs: dict[str, Any] = {}
    if "prefix" in all_params:
        kwargs["prefix"] = prefix
    if "config" in all_params:
        kwargs["config"] = model_config.hf_config
    if "cache_config" in all_params:
        kwargs["cache_config"] = vllm_config.cache_config
    if "scheduler_config" in all_params:
        kwargs["scheduler_config"] = vllm_config.scheduler_config
    # 旧式模型同样必须在配置上下文中构造，才能注册底层 Attention。
    with set_current_config(vllm_config):
        model = model_class(**kwargs)
    return model


def process_weights_after_loading(
    model: nn.Module,
    model_config: ModelConfig,
    target_device: torch.device,
) -> None:
    """在权重装填后，先处理各层权重方法，再处理 Attention 钩子。

    顺序沿用 vLLM：每个 quant_method 可以在这里将 checkpoint 布局转换为执行
    布局。普通精度方法默认无需转换。后续独立遍历 Attention，确保相关投影
    权重已经处理完；当前没有在线量化计时、权重缓存或权重重载路径。

    Args:
        model: 权重已经装填的模型。
        model_config: 提供解析后的模型计算 dtype。
        target_device: 后处理应使用的设备；CPU 参数会在必要时临时搬到该设备。

    Returns:
        无返回值；后处理钩子原地修改模块。

    Side Effects:
        可能临时搬运参数，并调用层级后处理；钩子异常直接向上传播。
    """
    # 权重方法只处理本层参数；是否需要临时搬运设备由方法自行声明。
    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            loading_context = (
                device_loading_context(module, target_device)
                if quant_method.requires_device_loading
                else nullcontext()
            )
            with loading_context:
                quant_method.process_weights_after_loading(module)
            # 方法可能重建专用参数，新的 TP 默认值未必符合本层 disable_tp 或覆盖值。
            if hasattr(module, "update_param_tp_status"):
                module.update_param_tp_status()

    # Attention 最后处理，以便使用已经完成方法级转换的投影参数。
    for _, module in model.named_modules():
        if isinstance(module, Attention) and hasattr(
            module, "process_weights_after_loading"
        ):
            # 设备上下文只管理临时参数位置，不替层实现布局转换。
            with device_loading_context(module, target_device):
                module.process_weights_after_loading(model_config.dtype)


@contextmanager
def device_loading_context(
    module: nn.Module,
    target_device: torch.device,
) -> Generator[nn.Module, None, None]:
    """临时将 CPU 参数移到目标设备，处理结束后恢复原来的设备位置。

    保留 vLLM 的普通参数搬运逻辑，不包含 UVA、pinned-memory offloading。
    只处理 named_parameters() 返回的参数，不移动 buffer，也不会搬动已经在
    非 CPU 设备上的参数。调用方应保证这些参数已经处于正确的执行设备。

    Args:
        module: 需要临时执行权重后处理的模块。
        target_device: 后处理目标设备；为 CPU 时直接交付原模块。

    Yields:
        module: 同一个 module，而非深拷贝。

    Side Effects:
        修改参数的 data 引用。上下文主体正常结束或抛异常后，按原参数名称
        恢复设备；保留处理后的数值，新增加的参数不参与恢复。
    """
    if target_device.type == "cpu":
        # 与 vLLM 一致：CPU 目标无需执行临时搬运，也不把现有 GPU 参数搬回 CPU。
        yield module
        return

    # 键为参数全名，值为搬运前的设备；恢复按名称匹配，可处理同名参数被替换。
    original_device_states: dict[str, torch.device] = {}
    # name/p 分别是模块树内的参数路径和 Parameter；只暂存原先在 CPU 的参数。
    for name, p in module.named_parameters():
        if p.device.type == "cpu":
            original_device_states[name] = p.device
            p.data = p.data.to(target_device)

    try:
        yield module
    finally:
        # 后处理可能替换参数对象，重新遍历而不是复用进入时的对象引用。
        for name, p in module.named_parameters():
            if name in original_device_states:
                original_device = original_device_states[name]
                p.data = p.data.to(original_device)


# 仅缓存模型类与架构名，不缓存 nn.Module 实例、权重或 KV cache。
_MODEL_ARCH_BY_HASH: dict[int, tuple[type[nn.Module], str]] = {}


def _get_model_architecture(
    model_config: ModelConfig,
) -> tuple[type[nn.Module], str]:
    """通过模型注册表解析 HF 的候选架构。

    Args:
        model_config: 提供 hf_config.architectures 和 registry 的模型配置。

    Returns:
        (model_cls, arch):
            model_cls: 具体 nn.Module 子类。
            arch: 命中的 HF 架构名称，当前不执行后端回退或任务转换。

    Raises:
        ValueError: 候选架构为空或没有受支持的模型实现。
    """
    # architectures 是 config.json 的候选类名列表，不是 checkpoint 参数名称。
    architectures = getattr(model_config.hf_config, "architectures", None) or []
    # registry 负责延迟导入；Loader 不应在这里写死 Qwen 类。
    model_cls, arch = model_config.registry.resolve_model_cls(
        architectures, model_config=model_config
    )
    return model_cls, arch


def get_model_architecture(
    model_config: ModelConfig,
) -> tuple[type[nn.Module], str]:
    """取得模型类与架构名，并复用当前进程内的解析缓存。

    Args:
        model_config: 模型目录及 HF 架构配置。

    Returns:
        (model_cls, arch):
            model_cls: 具体 nn.Module 子类。
            arch: 命中的 HF 架构名称，当前不执行后端回退或任务转换。

    Side Effects:
        首次解析时可能导入模型模块，并写入进程内缓存；不会构造模型。
    """
    # 保留 vLLM 的 hash 缓存结构，仅删除当前不存在的任务、远程代码和实现选择字段。
    key = hash(
        (
            model_config.model,
            tuple(getattr(model_config.hf_config, "architectures", None) or []),
        )
    )
    if key in _MODEL_ARCH_BY_HASH:
        return _MODEL_ARCH_BY_HASH[key]

    # model_arch 是 (Python 模型类, 架构名称)，不包含任何请求级可变状态。
    model_arch = _get_model_architecture(model_config)
    _MODEL_ARCH_BY_HASH[key] = model_arch
    return model_arch


def get_model_cls(model_config: ModelConfig) -> type[nn.Module]:
    """返回注册表解析出的模型类，不创建模型实例。

    Args:
        model_config: 用于架构解析的模型配置。

    Returns:
        model_cls: 具体的 nn.Module 子类。
    """
    return get_model_architecture(model_config)[0]


def get_architecture_class_name(model_config: ModelConfig) -> str:
    """返回实际选中的 HF 架构名称。

    Args:
        model_config: 用于架构解析的模型配置。

    Returns:
        arch: 例如 Qwen3ForCausalLM，而非模型目录或模块导入路径。
    """
    return get_model_architecture(model_config)[1]


@dataclass
class ParamMapping:
    """维护融合模块与其组成模块的双向名称映射。

    沿用 vLLM 数据结构；它不是 checkpoint 文件加载器，也不执行 TP 切片。
    Qwen 的 packed_modules_mapping 使用相同结构，可直接传入；checkpoint 名称
    转换和 shard 标识由独立的 WeightsMapper 处理。

    Attributes:
        packed_mapping: 融合模块名到有序子模块名列表的映射，例如
            {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}。
        inverse_packed_mapping: 子模块名到 (融合模块名, 子模块序号) 的映射，
            例如 q_proj 对应 ("qkv_proj", 0)。构造时补充这些反向条目。
    """

    packed_mapping: dict[str, list[str]]
    # 每个实例独立拥有反向表，不能使用共享的字典默认值。
    inverse_packed_mapping: dict[str, tuple[str, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """从正向映射生成反向条目，跳过模块映射到自身的情况。

        Side Effects:
            写入 inverse_packed_mapping；输入的 packed_mapping 不会被修改。
        """
        # packed_name 是融合模块名，sub_params 的顺序决定各组成模块的序号。
        for packed_name, sub_params in self.packed_mapping.items():
            if len(sub_params) == 1 and sub_params[0] == packed_name:
                continue
            # index 是融合内的序号，不是 TP rank；param_name 是子模块名称。
            for index, param_name in enumerate(sub_params):
                self.inverse_packed_mapping[param_name] = (packed_name, index)

    def get_sub_modules(self, module_name: str) -> tuple[str, list[str]] | None:
        """按模块路径后缀查找融合模块及其组成名称。

        Args:
            module_name: 完整模块路径，例如 model.layers.0.self_attn.qkv_proj。

        Returns:
            (key, value):
                key: 命中的融合模块短名称。
                value: 该融合模块的子模块名称列表，保留原列表引用；整体未命中时返回 None。

            未命中任何融合名称时，整体返回 None。
        """
        # key/value 是融合短名称和组成列表，匹配方式沿用 vLLM 的 endswith。
        for key, value in self.packed_mapping.items():
            if module_name.endswith(key):
                return key, value
        return None
