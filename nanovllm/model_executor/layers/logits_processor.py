# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""词表投影、TP logits 汇集与词表并行候选选择。

沿用 vLLM 的接口和参数方法分派。当前使用 nn.Module，不接入 PluggableLayer、
独立输出头精度或 FlashInfer top-k；普通引擎仍由 Sampler 对完整 logits 随机采样。
"""

import torch
from torch import nn

from nanovllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_gather,
)
from nanovllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from nanovllm.platforms import current_platform


def _topk(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (values, ids)，沿用上游 torch.topk 回退实现，暂不接入 FlashInfer。"""
    return torch.topk(scores, k, dim=-1)


class LogitsProcessor(nn.Module):
    """组织输出头投影与汇集，支持直接查询词表并行 argmax/top-k。

    Runner 在调用前选择 hidden state 行。forward 返回词表分数，不选择 token；
    get_top_tokens/get_top_k_tokens 则在本地先选择候选，再跨 rank 比较。
    后两者是独立接口，不代替需要完整词表概率的温度随机采样。
    """

    def __init__(
        self,
        vocab_size: int,
        org_vocab_size: int | None = None,
        scale: float = 1.0,
        logits_as_input: bool = False,
        soft_cap: float | None = None,
    ) -> None:
        """保存词表范围及输出变换，汇集策略沿用平台设置。

        Args:
            vocab_size: 模型声明的词表大小。
            org_vocab_size: 返回分数的原始词表范围，None 时等于 vocab_size。
            scale: soft cap 后乘到 logits 上的系数。
            logits_as_input: 输入已经是 logits，跳过投影、汇集与 padding 裁剪。
            soft_cap: 可选正数 c，执行 c * tanh(logits / c)。
        """
        super().__init__()
        self.scale = scale
        self.vocab_size = vocab_size
        self.logits_as_input = logits_as_input
        self.org_vocab_size = org_vocab_size or vocab_size
        self.logits_as_input = logits_as_input
        self.soft_cap = soft_cap
        self.use_all_gather = current_platform.use_all_gather()

    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
        skip_gather: bool = False,
    ) -> torch.Tensor | None:
        """计算并变换 logits，或直接变换传入的分数。

        Args:
            lm_head: 提供 quant_method 和本地词表参数的输出头。
            hidden_states: [..., hidden_size]；logits_as_input=True 时为已有分数。
            embedding_bias: 本 rank 的词表偏置，默认不加。
            skip_gather: 返回本地分数，保留本地 padding 和词表分片。

        Returns:
            logits: 默认是 [..., org_vocab_size]；skip_gather 时为本地宽度；
                gather 模式的非目标 rank 返回 None。

        Side Effects:
            可执行 TP 通信；直接传入 logits 时，scale 可能原地修改输入。
        """
        if self.logits_as_input:
            logits = hidden_states
        else:
            logits = self._get_logits(hidden_states, lm_head, embedding_bias, skip_gather)
        if logits is not None:
            if self.soft_cap is not None:
                logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
            if self.scale != 1.0:
                logits *= self.scale
        return logits

    def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor | None:
        """返回汇集后的 logits；gather 模式只有 TP rank 0 返回 tensor。"""
        if self.use_all_gather:
            logits = tensor_model_parallel_all_gather(logits)
        else:
            logits = tensor_model_parallel_gather(logits)
        return logits

    def _apply_head(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """通过输出头参数方法投影，返回包含本地 padding 的 logits。"""
        return lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
        skip_gather: bool = False,
    ) -> torch.Tensor | None:
        """先执行本地投影，按需汇集并裁去全局词表 padding。

        Returns:
            logits: 本地或完整词表分数，gather 的非目标 rank 为 None。
        """
        logits = self._apply_head(lm_head, hidden_states, embedding_bias)
        if skip_gather:
            return logits  # 本地索引仍以当前 shard 为原点，不能按全局大小裁剪。
        if lm_head.tp_size > 1:
            logits = self._gather_logits(logits)
        if logits is not None:
            logits = logits[..., :self.org_vocab_size]
        return logits

    def get_top_tokens(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """各 rank 先选本地 argmax，再只汇集候选分数和全局 ID。

        Returns:
            top_tokens: [num_rows] 的 int64 全局 token ID，输入为二维 hidden states。
                所有 TP rank 都得到结果；相同分数优先选择较小的全局 ID。

        Side Effects:
            多 rank 执行候选 all-gather；不会汇集完整词表 logits。
        """
        if self.scale <= 0.0 and self.scale != 1.0:
            raise ValueError(
                "The local argmax reduction optimization is not supported for "
                "non-positive logit scaling factors."
            )
        tp_size = lm_head.tp_size

        logits = self._apply_head(lm_head, hidden_states, embedding_bias)
        if self.soft_cap is not None:
            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
        if self.scale != 1.0:
            logits = logits * self.scale
        # 与上游一致，先屏蔽当前 shard 尾部的词表 padding。
        num_pad = lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")

        local_max_vals, local_max_indices = logits.max(dim=-1)

        # 将本地候选行号转换为原始词表中的全局 token ID。
        vocab_start = lm_head.shard_indices.org_vocab_start_index
        global_indices = local_max_indices + vocab_start
        if tp_size == 1:
            return global_indices

        # FP32 能精确表示当前 Qwen 词表 ID；避免 BF16 直接保存 ID 时舍入。
        local_pair = torch.stack(
            [local_max_vals.float(), global_indices.float()], dim=-1,
        )
        # [num_rows, 2] 一次汇集为 [num_rows, 2 * tp_size]。
        gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
        gathered = gathered.view(hidden_states.shape[0], tp_size, 2)
        # 第 0 项是候选分数，第 1 项是 ID；比较分数后选出对应全局 ID。
        max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
        top_tokens = gathered[:, :, 1].gather(dim=-1, index=max_rank_idx)
        return top_tokens.squeeze(-1).to(torch.int64)

    def get_top_k_tokens(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        k: int,
        embedding_bias: torch.Tensor | None = None,
        *,
        return_log_probs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """从各 rank 的本地 top-k 中选择全局 top-k。

        Args:
            lm_head: 持有分片输出权重和原始词表区间的层。
            hidden_states: [..., hidden_size] 的投影输入。
            k: 候选数，须为正整数，且不超过本地分配的词表行数和原始词表大小。
            embedding_bias: 可选本地词表偏置。
            return_log_probs: 返回全词表归一化的 log-prob，而非候选内归一化。

        Returns:
            (ids, values): 均为 [..., k]；ids 是 int64 全局 ID，values 为 FP32
                logits 或 log-prob。并列分数的 ID 顺序沿用 torch.topk。

        Side Effects:
            多 rank 汇集候选；请求 log-prob 时额外汇集每个 rank 的 log-partition。

        Note:
            沿用本地上游流程：普通候选先选择，再转 FP32、scale、soft cap；
            log-prob 分支在选择前对完整本地 logits 做 soft cap、scale。
        """
        if self.scale <= 0.0 and self.scale != 1.0:
            raise ValueError(
                "The local top-k reduction optimization is not supported for "
                "non-positive logit scaling factors."
            )

        logits = self._apply_head(lm_head, hidden_states, embedding_bias)
        if return_log_probs:
            if self.soft_cap is not None:
                logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
            if self.scale != 1.0:
                logits = logits * self.scale

        # 屏蔽 shard 尾部 padding，使补齐行不参与候选选择和全局归一化。
        num_pad = lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")

        if return_log_probs:
            log_partition = torch.logsumexp(logits.float(), dim=-1, keepdim=True)
            if lm_head.tp_size > 1:
                log_partition = torch.logsumexp(
                    tensor_model_parallel_all_gather(log_partition, dim=-1),
                    dim=-1,
                    keepdim=True,
                )

        values, ids = _topk(logits, k)
        # 本地行号加 shard 起点，转换成全局 token ID。
        ids = ids.to(torch.int64) + lm_head.shard_indices.org_vocab_start_index

        if lm_head.tp_size > 1:
            values = tensor_model_parallel_all_gather(values, dim=-1)
            ids = tensor_model_parallel_all_gather(ids, dim=-1)
            values, selected = _topk(values, k)
            ids = ids.gather(-1, selected)

        values = values.float()
        if return_log_probs:
            return ids, values - log_partition
        if self.scale != 1.0:
            values = values * self.scale
        if self.soft_cap is not None:
            values = torch.tanh(values / self.soft_cap) * self.soft_cap
        return ids, values

    def extra_repr(self) -> str:
        """返回 s，展示词表大小与 logits 变换设置。"""
        return (
            f"vocab_size={self.vocab_size}, org_vocab_size={self.org_vocab_size}, "
            f"scale={self.scale}, logits_as_input={self.logits_as_input}"
        )
