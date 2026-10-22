"""本地文本生成模型的路径、HF 结构和执行精度配置。"""

from __future__ import annotations

import os
from dataclasses import field
from typing import TYPE_CHECKING

import torch
from pydantic import ConfigDict
from transformers import AutoConfig, PretrainedConfig

from nanovllm.config.utils import config

if TYPE_CHECKING:
    from nanovllm.config.parallel import ParallelConfig
    from nanovllm.model_executor.models.registry import _ModelRegistry


# torch.dtype 与 HF 配置是运行时对象，保留实例类型而不要求 Pydantic 展开其结构。
@config(config=ConfigDict(arbitrary_types_allowed=True), slots=True)
class ModelConfig:
    """描述一个已选择语言模型的结构和执行方式。

    当前模型注册表只接入本地 Hugging Face Qwen3 decoder-only 文本模型。
    构造时读取 hf_config、解析 dtype，并按位置编码长度收紧 max_model_len；
    不创建模型层、不加载权重、不初始化 CUDA。量化、LoRA、多模态、pooling
    和独立 tokenizer 策略尚未接入。

    Attributes:
        model: 本地 Hugging Face 模型目录。
        max_model_len: 单个请求允许的最大总 token 数，包含 prompt 与生成 token。
            构造后会受模型位置编码长度限制。
        enforce_eager: 是否关闭模型 forward 的 CUDA Graph 捕获与回放。
            NanovllmConfig 会据此将 Graph 模式解析为 NONE；局部层已有的
            @torch.compile、Triton JIT 以及启动时的内核和采样预热仍会执行。
        dtype: 参数与浮点计算精度；输入为 auto、half/float16、bfloat16、
            float/float32 或相应 torch.dtype，构造后统一为 torch.dtype。
            auto 保留声明的 FP16/BF16；FP32 或未声明时选择 FP16。当前暂不做
            vLLM 的平台能力自适应选择，也不改变 hf_config 中的原始精度。
        hf_config: 从 ``model`` 目录读取的 Hugging Face 配置。它是派生状态，不能
            由调用方传入，以免模型路径与结构配置不一致。
    """

    # 模型目录是构造模型与加载权重的唯一来源。
    model: str
    # 该上限同时约束 Scheduler 接纳请求和 Worker 侧固定容量输入 buffer。
    max_model_len: int = 4096
    # 禁用 CUDA Graph；不等价于关闭局部函数编译、Triton JIT 或内核预热。
    enforce_eager: bool = False
    # HF 只提供 auto 的候选精度，运行时消费者统一读取这个已解析的字段。
    dtype: str | torch.dtype = "auto"
    # Hugging Face 配置必须由 model 路径派生，禁止调用方注入另一个模型的配置。
    hf_config: PretrainedConfig = field(init=False)

    def __post_init__(self) -> None:
        """校验路径和长度，加载结构配置并解析运行时 dtype。

        Side Effects:
            从本地 ``model`` 目录读取 ``config.json`` 等 Hugging Face 配置文件，并
            写入 hf_config 并将 dtype 解析为 torch.dtype；不修改 HF 原始精度。
            不会创建 torch.nn.Module 或加载权重。

        Raises:
            ValueError: 模型目录不存在、max_model_len 非正或 dtype 不支持。
        """
        # nano-vLLM 当前不负责 Hub 下载，因此启动前必须确保模型已位于本地。
        if not os.path.isdir(self.model):
            raise ValueError(f"Model directory does not exist: {self.model}")
        # 上下文长度是所有 request 的硬上限，零或负数没有有效语义。
        if self.max_model_len <= 0:
            raise ValueError("max_model_len must be positive")

        # 这里只读取结构配置；权重仍由 GPUModelRunner.load_model() 加载。
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # 在配置边界统一精度，避免参数构造和 KV cache 字节计算使用不同来源。
        self.dtype = _get_and_verify_dtype(self.hf_config, self.dtype)
        # 不同 HF 架构未必暴露该字段。存在且为正整数时才收紧用户长度。
        model_max_model_len = getattr(
            self.hf_config,
            "max_position_embeddings",
            None,
        )
        if isinstance(model_max_model_len, int) and model_max_model_len > 0:
            self.max_model_len = min(self.max_model_len, model_max_model_len)

    @property
    def registry(self) -> _ModelRegistry:
        """返回模型架构注册表，不导入或构造具体模型。

        Returns:
            与 vLLM 一样返回进程内 ModelRegistry 实例；当前只登记 Qwen3。
        """
        # 延迟导入避免配置模块和模型选择工具之间形成初始化循环。
        from nanovllm.model_executor.models import ModelRegistry

        return ModelRegistry

    def get_vocab_size(self) -> int:
        """返回模型词表大小。

        Returns:
            Hugging Face 配置中声明的全局 token 词表大小。

        Raises:
            ValueError: 当前模型配置没有有效的正整数 ``vocab_size`` 时抛出。
        """
        # 词表并行层、InputBatch 和采样负载均使用同一个全局词表大小。
        vocab_size = getattr(self.hf_config, "vocab_size", None)
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            raise ValueError(
                "Model Hugging Face config must define a positive vocab_size"
            )
        return vocab_size

    def get_num_attention_heads(self, parallel_config: ParallelConfig) -> int:
        """返回当前 TP rank 的 Query head 数，供 Runner 建立执行布局。

        沿用 vLLM 的同名接口：模型总 Query head 数除以 TP size。当前只注册
        Qwen3，结构信息直接来自 hf_config；实际 Engine 目前只选择 TP=1 执行器。

        Args:
            parallel_config: 当前模型并行配置，提供 tensor_parallel_size。

        Returns:
            单个 rank 负责的 Query head 数，不是 KV head 数。
        """
        # 这里只读取结构并计算分片数量，执行器支持范围由 Executor.get_class() 检查。
        num_heads = self.hf_config.num_attention_heads
        return num_heads // parallel_config.tensor_parallel_size


# 与 vLLM 一致的常用别名；不包含量化存储 dtype。
_STR_DTYPE_TO_TORCH_DTYPE: dict[str, torch.dtype] = {
    "half": torch.float16,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float": torch.float32,
    "float32": torch.float32,
}


def _get_and_verify_dtype(
    config: PretrainedConfig,
    dtype: str | torch.dtype,
) -> torch.dtype:
    """解析普通文本模型的 FP16、BF16 或 FP32 运行时精度。

    简化自 vLLM 同名工具。HF 精度仅作为 auto 的输入，不回写 HF 配置。
    不探测设备能力；配置接受 FP32 不代表当前 FA 后端支持 FP32 执行。

    Args:
        config: 本地模型的 HF 结构配置。
        dtype: 用户指定的精度字符串或 torch.dtype。

    Returns:
        模型参数、Runner 和 auto KV cache 共同使用的 torch.dtype。

    Raises:
        ValueError: 精度名称未知，或解析结果不是当前支持的浮点类型。
    """
    if isinstance(dtype, str):
        # 规范化用户输入，兼容 BFLOAT16 等大写写法。
        dtype_name = dtype.lower()
        if dtype_name == "auto":
            # 新版 Transformers 使用 dtype；兼容旧版 torch_dtype 字段。
            config_dtype = getattr(config, "dtype", None)
            if config_dtype is None:
                config_dtype = getattr(config, "torch_dtype", None)
            if config_dtype is None:
                # 缺省按 FP32 模型处理，之后应用 MVP 的自动降精度策略。
                config_dtype = torch.float32
            if isinstance(config_dtype, str):
                # 某些配置保留字符串表示，仅在本地解析，不修改源对象。
                config_dtype = _STR_DTYPE_TO_TORCH_DTYPE.get(config_dtype.lower())
            # 当前固定策略：FP32 自动降为 FP16，FP16/BF16 保持原精度。
            torch_dtype = (
                torch.float16 if config_dtype == torch.float32 else config_dtype
            )
        else:
            # 显式 FP32 不降精度；显式选择优先于 checkpoint 中的声明。
            torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE.get(dtype_name)
    else:
        # torch.dtype 对象也经过下面的校验，避免整数参数或 float64 混入。
        torch_dtype = dtype

    if not isinstance(torch_dtype, torch.dtype) or torch_dtype not in (
        torch.float16, torch.bfloat16, torch.float32
    ):
        raise ValueError(
            f"Unsupported model dtype: {dtype!r}; "
            "expected auto, float16/half, bfloat16, float32/float "
            "or an equivalent torch.dtype."
        )
    return torch_dtype
