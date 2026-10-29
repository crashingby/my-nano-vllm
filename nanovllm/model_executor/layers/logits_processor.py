"""使用 LMHead 权重计算词表分数，保留 vLLM 的非量化 logits 计算流程。"""

import torch
import torch.nn.functional as F
from torch import nn

from nanovllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from nanovllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_gather,
)
from nanovllm.platforms import current_platform

class LogitsProcessor(nn.Module):
    """执行词表投影与 TP logits 汇集，不选择 token 行或执行随机采样。

    与 sample/logits_processor 中的采样策略接口不同，本类处理模型输出投影。
    Runner 已选择好每请求最后一个 Query 的 hidden state；这里逐行计算 logits，
    无需读取 ForwardContext 或原 nano-vLLM 的 Context。
    """

    def __init__(self, vocab_size: int) -> None:
        """保存完整词表大小，当前没有词表扩展、logits 缩放或 soft cap。

        Args:
            vocab_size: 模型原始词表大小，与 embedding/LMHead 的行数一致。
        """
        super().__init__()
        self.vocab_size = vocab_size  # 模型声明的完整输出词表大小。
        self.org_vocab_size = vocab_size  # 无 LoRA 词表扩展，原始词表与完整词表相同。
        self.use_all_gather = current_platform.use_all_gather()
    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """用输出权重计算传入 hidden states 的完整词表分数。

        Args:
            lm_head: 持有 [vocab_size / tp_size, hidden_size] 权重的模型层。
            hidden_states: [num_rows, hidden_size]，行选择已由调用方完成。
            embedding_bias: 上游保留的本 rank 词表 bias 参数，Qwen3 传 None。

        Returns:
            logits: 当前 CUDA 默认 all-gather，各 TP rank 返回完整的
                [num_rows, vocab_size]；gather 分支只在 TP rank 0 返回 tensor。
                保留全部输入行，不区分 prefill/decode。

        Side Effects:
            TP 模式汇集词表分片；不修改 hidden states 或权重。
        """
        return self._get_logits(hidden_states, lm_head, embedding_bias)

    def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor | None:
        """按平台选择汇集方式，返回完整 logits；gather 的非目标 rank 返回 None。"""
        if self.use_all_gather:
            # 当前 CUDA 默认走此分支，所有 TP rank 都得到完整 logits 并执行采样。
            logits = tensor_model_parallel_all_gather(logits)
        else:
            # 保留基础 gather 接口；启用此分支时须另外处理其他 rank 的采样状态。
            logits = tensor_model_parallel_gather(logits)
        return logits

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """先计算本地词表分数，再汇集并返回原始词表范围。

        Returns:
            logits: 去除词表 padding 后的 logits；非输出 rank 为 None。
        """
        # 当前方法分派仅接入 decoder 线性层；LMHead 仍保存普通浮点权重，
        # 在这里直接调用 F.linear，词表量化后续单独接入。
        logits = F.linear(hidden_states, lm_head.weight, bias=embedding_bias)
        logits = self._gather_logits(logits)
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]  # 保留上游原始词表范围约定。
        return logits
