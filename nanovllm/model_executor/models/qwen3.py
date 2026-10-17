"""Qwen3 普通 decoder 模型，按 vLLM 接口构造并兼容 Hugging Face 权重。

模型类负责组织 embedding、Attention、MLP、残差与输出投影接口。
Runner 提供 packed token 输入、positions 和运行时 Attention 上下文，
并在调用 compute_logits 前选择所需 hidden states。
"""

from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch import nn
from transformers import Qwen3Config

from nanovllm.config import NanovllmConfig
from nanovllm.model_executor.layers.activation import SiluAndMul
from nanovllm.model_executor.layers.attention import Attention
from nanovllm.model_executor.layers.layernorm import RMSNorm
from nanovllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from nanovllm.model_executor.layers.logits_processor import LogitsProcessor
from nanovllm.model_executor.layers.rotary_embedding import get_rope
from nanovllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from nanovllm.model_executor.model_loader.weight_utils import default_weight_loader


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
        prefix: str = "",
    ) -> None:
        """构造 Qwen3 的一个 self-attention 子层。

        Args:
            hidden_size: 输入 hidden state 的完整维度。
            num_heads: 全部 TP rank 合计的 Query head 数。
            num_kv_heads: 全部 TP rank 合计的 KV head 数；当前须能被 TP size 整除。
            max_position: RoPE 缓存覆盖的位置上限。
            head_dim: 每个 head 的维度；缺省从 hidden_size 和 Query head 数计算。
            rms_norm_eps: Q/K Norm 的数值稳定项。
            qkv_bias: 是否为 QKV 投影创建 bias，不影响 Q/K Norm 是否执行。
            rope_theta: 当前普通 RoPE 的频率基数。
            rope_scaling: 保留当前 RoPE 配置入口，仅读取其中的 rope_theta。
            prefix: 当前 ``Qwen3Attention`` 在模型中的完整路径，例如
                ``"model.layers.0.self_attn"``。它会继续派生出底层 Attention
                的 ``".attn"`` 路径。
        """
        super().__init__()
        self.hidden_size = hidden_size  # 输出投影最终恢复到此维度。
        # 当前进程组就是 TP 组；Linear 使用相同的 rank/world size 切分权重。
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
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
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
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
        attention_prefix = f"{prefix}.attn" if prefix else "attn"
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            prefix=attention_prefix,
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
        """
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # 当前 RoPE 接口接收三维 head 视图，保持本项目已支持的张量契约。
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)  # 包装层已返回 [num_tokens, q_size]。
        output = self.o_proj(attn_output)  # 当前 Linear 直接返回 tensor，无独立 bias 返回值。
        return output


class Qwen3MLP(nn.Module):
    """用融合 gate/up 投影、SwiGLU 和 down 投影构成前馈子层。"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        """构造当前仅支持 SiLU 激活的 Qwen3 MLP。"""
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将 [num_tokens, hidden_size] 经 SwiGLU 映射回相同形状。"""
        gate_up = self.gate_up_proj(x)  # gate/up 合并为一次本地投影。
        x = self.act_fn(gate_up)  # SiLU(gate) * up，继续使用现有编译算子。
        x = self.down_proj(x)  # 恢复 hidden_size，TP 时归约各 rank 的贡献。
        return x


class Qwen3DecoderLayer(nn.Module):
    """执行一个 decoder block，并把延迟相加的 residual 传给下一层。"""

    def __init__(
        self,
        config: Qwen3Config,
        prefix: str = "",
    ) -> None:
        """构造一个 Transformer decoder layer。

        Args:
            config: Hugging Face Qwen3 模型配置。
            prefix: 当前层的完整路径，例如 ``"model.layers.0"``。
        """
        super().__init__()
        self.hidden_size = config.hidden_size  # 本层输入和输出的完整模型维度。
        # 该名称与 self.self_attn 在 nn.Module 树中的实际路径严格一致。
        self_attention_prefix = (
            f"{prefix}.self_attn" if prefix else "self_attn"
        )
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
            prefix=self_attention_prefix,
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
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
        # ModuleList 的索引也是 PyTorch module path 的一部分，所以直接用 layer_idx
        # 构造 ``model.layers.<idx>``，不能依赖运行时遍历顺序。
        layers_prefix = f"{prefix}.layers" if prefix else "layers"
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(
                    config,
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
        """执行查表和全部 decoder 层，保留输入 token 的 packed 行顺序。"""
        hidden_states = self.embed_tokens(input_ids)
        residual = None  # 首层创建 residual，之后每层更新并向下传递。
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)  # 合并最后一层 MLP 的残差。
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 CausalLM 顶层模型。"""
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
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
            当前不支持 PP、量化或投机解码，模型始终持有全部 decoder 层。
        """
        super().__init__()
        config = vllm_config.model_config.hf_config  # HF 配置只负责具体模型结构。
        self.config = config
        self.vllm_config = vllm_config  # 与 Loader/Worker 共用总配置对象。
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
        """返回 decoder hidden states，词表投影由 compute_logits 单独执行。"""
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """计算已选定 hidden states 的词表分数，TP 时仅 rank 0 返回结果。

        token 行选择由 Runner 完成；此处不读取 prefill/decode Context。
        """
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        """按 Qwen 的融合参数规则消费 checkpoint 权重迭代器。

        保留 vLLM Qwen 普通权重装填的循环结构，复用本项目参数上的 weight_loader
        完成 QKV/MLP 融合和 TP 切片。这里不扫描目录，不调用文件下载接口。

        Args:
            weights: DefaultModelLoader 提供的原始 checkpoint 名称与 CPU tensor。

        Returns:
            实际装填的模型参数名集合，供 Loader 检查参数级覆盖。

        Side Effects:
            原地修改参数。共享 embedding 时跳过 lm_head 的重复条目；
            对应 Parameter 已通过 model.embed_tokens.weight 装填。
        """
        # 保留重复名称便于按 checkpoint 定位；最终完整性检查按共享 Parameter 去重。
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        # name 是 checkpoint 原始名称，loaded_weight 是该条目尚未做 TP 切片的 tensor。
        for name, loaded_weight in weights:
            # RoPE 频率由实现计算，不是需要从 checkpoint 装填的训练参数。
            if "rotary_emb.inv_freq" in name:
                continue
            if self.config.tie_word_embeddings and name.startswith("lm_head."):
                continue
            # 本项目映射方向是原始模块 -> (融合模块, 分片标识)，保留已有模型定义。
            for weight_name, (param_name, shard_id) in self.packed_modules_mapping.items():
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    # 参数加载器知道当前 TP rank，并将 q/k/v 或 gate/up 放到对应区域。
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # 与 vLLM 普通 Qwen 路径一致，跳过模型不持有的额外 checkpoint 条目。
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            # 记录融合后的目标名称，不把 q_proj 等原始名称交给模型参数覆盖检查。
            loaded_params.add(name)
        return loaded_params
