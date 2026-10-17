"""使用 LMHead 权重计算词表分数，保留 vLLM 的非量化 logits 计算流程。"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from nanovllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)


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
            TP rank 0 返回 [num_rows, vocab_size]；其他 rank 返回 None。
            单卡直接返回完整 logits。保留全部输入行，不区分 prefill/decode。

        Side Effects:
            TP 模式调用 gather 汇集词表分片；不修改 hidden states 或权重。
        """
        return self._get_logits(hidden_states, lm_head, embedding_bias)

    def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor | None:
        """按 TP rank 顺序汇集词表分片，只有 rank 0 持有完整 logits。"""
        # 当前默认进程组就是 TP 组，沿用项目现有的原生 CUDA/NCCL 通信路径。
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return logits
        tp_rank = dist.get_rank()
        # gather 只有目标 rank 需要接收缓冲区，其他 rank 提供本地 logits。
        all_logits = (
            [torch.empty_like(logits) for _ in range(tp_size)]
            if tp_rank == 0
            else None
        )
        dist.gather(logits, gather_list=all_logits, dst=0)
        # 沿词表维度拼接；输入行顺序在各 rank 上保持一致。
        return torch.cat(all_logits, dim=-1) if tp_rank == 0 else None

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """先计算本地词表分数，再汇集并返回原始词表范围。"""
        # 上游通过 lm_head.quant_method.apply 执行；当前未引入量化方法框架，
        # 直接使用对应非量化实现的 F.linear，继续由 PyTorch 执行矩阵乘。
        logits = F.linear(hidden_states, lm_head.weight, bias=embedding_bias)
        logits = self._gather_logits(logits)
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]  # 保留上游原始词表范围约定。
        return logits
