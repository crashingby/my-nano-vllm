"""Qwen3 普通 decoder 模型，按 vLLM 接口构造并兼容 Hugging Face 权重。

模型类负责组织 embedding、Attention、MLP、残差与输出投影接口。
Runner 提供 packed token 输入、positions 和运行时 Attention 上下文，
并在调用 compute_logits 前选择所需 hidden states。
"""

from collections.abc import Iterable

import torch
from torch import nn
from transformers import Qwen3Config

from nanovllm.config import NanovllmConfig
from nanovllm.distributed import get_tensor_model_parallel_world_size
from nanovllm.model_executor.layers.activation import SiluAndMul
from nanovllm.model_executor.layers.attention import Attention
from nanovllm.model_executor.layers.layernorm import RMSNorm
from nanovllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from nanovllm.model_executor.layers.logits_processor import LogitsProcessor
from nanovllm.model_executor.layers.quantization import QuantizationConfig
from nanovllm.model_executor.layers.rotary_embedding import get_rope
from nanovllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from nanovllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper


class Qwen3Attention(nn.Module):
    """组合 QKV 投影、逐 head Q/K Norm、RoPE、后端 Attention 与输出投影。"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        """构造 Qwen3 的一个 self-attention 子层。

        Args:
            hidden_size: 输入 hidden state 的完整维度。
            num_heads: 全部 TP rank 合计的 Query head 数。
            num_kv_heads: 全局 KV head 数；不少于 TP size 时均分，少于时按整数倍复制。
            max_position: RoPE 缓存覆盖的位置上限。
            head_dim: 每个 head 的维度；缺省从 hidden_size 和 Query head 数计算。
            rms_norm_eps: Q/K Norm 的数值稳定项。
            qkv_bias: 是否为 QKV 投影创建 bias，不影响 Q/K Norm 是否执行。
            rope_theta: 当前普通 RoPE 的频率基数。
            rope_scaling: 保留当前 RoPE 配置入口，仅读取其中的 rope_theta。
            quant_config: QKV 和输出线性投影的权重方法配置；不改变 Attention/KV 精度。
            prefix: 当前 ``Qwen3Attention`` 在模型中的完整路径，例如
                ``"model.layers.0.self_attn"``。它会继续派生出底层 Attention
                的 ``".attn"`` 路径。
        """
        super().__init__()
        self.hidden_size = hidden_size  # 输出投影最终恢复到此维度。
        # Attention 与 Linear 共用 TP size；默认 world 不参与模型分片计算。
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # KV heads 足够时均分，每个 rank 加载不同的完整 head。
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # TP 大于 KV head 数时，多个相邻 rank 加载同一个源 KV head。
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        # 空 prefix 仅用于单独构造该 module 的场景；正常模型构造时它是完整父路径。
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            prefix=f"{prefix}.attn",
        )
        # 与 vLLM Qwen3 一致，逐 head Q/K Norm 始终存在，与投影 bias 无关。
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """执行本轮 Query 的 Attention，返回 [num_tokens, hidden_size]。

        Q/K Norm 在 RoPE 前作用于每个 head 的最后一维；分页缓存写入和
        Attention 计算委托给 self.attn，模型不维护另一套缓存或批次 Context。

        Returns:
            output: Attention 投影后的 tensor，形状为 [num_tokens, hidden_size]。
        """
        qkv, _ = self.qkv_proj(hidden_states)  # 偏置已在投影内相加，无待处理 bias。
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # 当前 RoPE 接口接收三维 head 视图，保持本项目已支持的张量契约。
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)  # attn_output 为包装层返回的 [num_tokens, q_size]。
        output, _ = self.o_proj(attn_output)  # 无偏置层的 output_bias 为 None。
        return output


class Qwen3MLP(nn.Module):
    """用融合 gate/up 投影、SwiGLU 和 down 投影构成前馈子层。"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        """构造 SiLU MLP，将 quant_config 和派生 prefix 传给两个线性投影。"""
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj" if prefix else "gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj" if prefix else "down_proj",
        )
        assert hidden_act == "silu"
        self.silu_and_mul = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将 [num_tokens, hidden_size] 经 SwiGLU 映射回相同形状。

        Returns:
            x: 经过 gate/up 投影、SiLU 乘法及 down 投影的 MLP 输出，与输入形状相同。
        """
        gate_up, _ = self.gate_up_proj(x)  # gate/up 合并为一次本地投影，无偏置。
        x = self.silu_and_mul(gate_up)  # SiLU(gate) * up，使用构造时按 kernel_config 绑定的实现。
        x, _ = self.down_proj(x)  # 恢复 hidden_size，TP 时归约各 rank 的贡献。
        return x


class Qwen3DecoderLayer(nn.Module):
    """执行一个 decoder block，并把延迟相加的 residual 传给下一层。"""

    def __init__(
        self,
        config: Qwen3Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        """构造一个 Transformer decoder layer。

        Args:
            config: Hugging Face Qwen3 模型配置。
            quant_config: 传给 Attention/MLP 中线性层的权重方法配置。
            prefix: 当前层的完整路径，例如 ``"model.layers.0"``。
        """
        super().__init__()
        self.hidden_size = config.hidden_size  # 本层输入和输出的完整模型维度。
        # 该名称与 self.self_attn 在 nn.Module 树中的实际路径严格一致。
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        # Attention 与 MLP 都输出本层 hidden_size，残差相加延后合入下一次 RMSNorm。
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        # 前归一化供 Attention 使用，后归一化先合并 Attention 输出与残差再供 MLP 使用。
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回本层 MLP 输出与待相加 residual，复用 fused add + RMSNorm。

        首层没有 residual 时先保存原始输入；后续层在 input_layernorm 中
        合并上一层 MLP 输出和 residual，避免在模型中另行执行残差相加。

        Returns:
            (hidden_states, residual):
                hidden_states: 本层 MLP 输出，尚未与返回的 residual 相加。
                residual: 留给下一次 fused add + RMSNorm 合并的残差 tensor。
        """
        if residual is None:
            residual = hidden_states  # 保存 Attention 子层的残差输入。
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """组织完整 decoder 主体，forward 返回所有 packed token 的 hidden states。"""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            # 名称转换同时把投影标识附到 tensor.shard_id，供融合层加载器读取。
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        }
    )

    def __init__(
        self,
        *,
        vllm_config: NanovllmConfig,
        prefix: str = "",
    ) -> None:
        """构造 Qwen3 decoder 主体，并为每个 decoder layer 生成稳定路径。

        Args:
            vllm_config: 总配置，从 model_config.hf_config 取得 Qwen3 结构参数。
            prefix: 当前模型主体的路径。顶层 CausalLM 会传入 ``"model"``。
        """
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config  # 自动加载器可从模型主体查找映射钩子。
        # ModuleList 的索引也是 PyTorch module path 的一部分，所以直接用 layer_idx
        # 构造 ``model.layers.<idx>``，不能依赖运行时遍历顺序。
        layers_prefix = f"{prefix}.layers" if prefix else "layers"
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(
                    config,
                    quant_config=vllm_config.quant_config,
                    prefix=f"{layers_prefix}.{layer_idx}",
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """执行查表和全部 decoder 层，保留输入 token 的 packed 行顺序。

        Returns:
            hidden_states: 全部 decoder 层及最终 RMSNorm 的输出，保留 packed token 的行顺序。
        """
        hidden_states = self.embed_tokens(input_ids)
        residual = None  # 首层创建 residual，之后每层更新并向下传递。
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)  # 合并最后一层 MLP 的残差。
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """把 HF 投影名称转换为融合名称，再递归分派给各层的加载入口。

        Args:
            weights: 相对于模型主体的 checkpoint 流，例如 layers.0.self_attn.q_proj.weight。

        Returns:
            loaded_params: 相对于当前主体的已加载参数名集合，使用融合后的名称。

        Side Effects:
            迭代消费源权重，映射器附加 shard_id，各层加载器原地写入本地 TP 分片。
        """
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 CausalLM 顶层模型。"""
    hf_to_vllm_mapper = Qwen3Model.hf_to_vllm_mapper
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        *,
        vllm_config: NanovllmConfig,
        prefix: str = "",
    ) -> None:
        """通过 vLLM 式构造接口创建 decoder 主体与输出头。

        Args:
            vllm_config: Worker 的总配置；Loader 在构造期间安装同一配置上下文。
            prefix: 此模型在外层 module 中的完整路径，默认根模型使用空字符串。

        Side Effects:
            创建模型参数，各 Attention 层以派生出的完整名称注册到静态上下文。
            当前不支持 PP 或投机解码，模型始终持有全部 decoder 层。
            quant_config 只向 decoder 线性层传递，词表层仍使用普通精度。
        """
        super().__init__()
        config = vllm_config.model_config.hf_config  # HF 配置只负责具体模型结构。
        self.config = config
        self.vllm_config = vllm_config  # 与 Loader/Worker 共用总配置对象。
        self.quant_config = vllm_config.quant_config
        if self.quant_config is not None:
            # 与 vLLM 一致，融合关系和 HF 名称转换分开保存，量化配置直接复用前者。
            self.quant_config.packed_modules_mapping.update(self.packed_modules_mapping)
        model_prefix = f"{prefix}.model" if prefix else "model"
        self.model = Qwen3Model(vllm_config=vllm_config, prefix=model_prefix)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            # 共享同一个 Parameter，而不只是 data，参数遍历才能正确去重。
            self.lm_head.weight = self.model.embed_tokens.weight
        # 沿用 vLLM：LMHead 保存权重，由独立接口计算和汇集词表分数。
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """返回 decoder hidden states，词表投影由 compute_logits 单独执行。

        Returns:
            hidden_states: decoder 输出的 hidden states；词表投影由 compute_logits() 另行完成。
        """
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """计算已选定 hidden states 的词表分数，汇集方式由 LogitsProcessor 决定。

        token 行选择由 Runner 完成；此处不读取 prefill/decode Context。

        Returns:
            logits: 当前 CUDA 默认 all-gather，各 TP rank 都返回完整词表分数；
                使用 gather 分支时，仅输出 rank 返回 tensor，其余返回 None。
        """
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        """沿用 vLLM 顶层入口，由自动加载器递归分派 checkpoint 权重。

        Args:
            weights: DefaultModelLoader 提供的原始 checkpoint 名称与 CPU tensor。

        Returns:
            loaded_params: 实际装填的完整模型参数名集合，供 Loader 检查覆盖情况。

        Side Effects:
            原地修改参数。自动加载器处理共享 embedding 去重，模型主体负责
            HF 名称映射，QKV/MLP 层负责融合装填，参数加载器负责 TP 切片。
        """
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
