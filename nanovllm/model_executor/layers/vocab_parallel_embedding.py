"""按词表维度分片的 embedding 与 LMHead 权重，沿用 vLLM 的职责划分。

VocabParallelEmbedding 执行 token 查表，ParallelLMHead 保存输出投影权重。
词表分数由独立的 LogitsProcessor 计算；两者均不读取批次 Context。
当前保留原生 PyTorch 运算，不包含量化、LoRA 或词表 padding。
"""

import torch
import torch.nn.functional as F
from torch import nn

from nanovllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from nanovllm.model_executor.utils import set_weight_attrs


class VocabParallelEmbedding(nn.Module):
    """保存本 rank 的词表分片，并汇总各 rank 的 embedding 结果。

    构造前须初始化 TP 组；词表分片使用组内 rank，不依赖默认 world 布局。
    不对词表补齐，因此 num_embeddings 必须能被 TP size 整除。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ) -> None:
        """创建本 rank 的 embedding 权重，登记 checkpoint 加载接口。

        Args:
            num_embeddings: 完整词表大小。
            embedding_dim: token embedding 和模型 hidden state 的维度。

        Side Effects:
            分配 [num_embeddings / tp_size, embedding_dim] 的未初始化参数。
            实际精度和设备沿用模型加载期间的默认 dtype 与设备上下文。
        """
        super().__init__()
        self.tp_rank = get_tensor_model_parallel_rank()  # 组内编号决定本地词表区间。
        self.tp_size = get_tensor_model_parallel_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings  # checkpoint 中未分片的词表行数。
        self.embedding_dim = embedding_dim  # 每行 embedding 的元素数。
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        # token ID 属于全局词表；查表前转换为本 rank 的局部行号。
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )
        # 模型权重加载器通过 Parameter 上的接口完成 TP 切片。
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        """从完整 checkpoint 权重中取本 rank 的词表行，原地装填参数。

        Args:
            param: 本 rank 的分片参数。
            loaded_weight: [num_embeddings, embedding_dim] 的完整词表权重。

        Side Effects:
            原地更新 param，不改变参数对象身份，保留 embedding/LMHead 权重共享。
        """
        param_data = param.data
        assert loaded_weight.shape[0] == self.num_embeddings
        shard_size = param_data.size(0)  # 当前 rank 需要装填的词表行数。
        start_idx = self.tp_rank * shard_size  # checkpoint 中分片的首行。
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """将全局 token ID 映射为 embedding，TP 时汇总各 rank 的贡献。

        Args:
            input_: token ID tensor，普通 packed 输入为 [num_tokens]。

        Returns:
            output_parallel: input_.shape + [embedding_dim] 的 embedding。每个 TP rank 都获得
                相同的完整结果，例如输入 [4] 返回 [4, embedding_dim]。

        Side Effects:
            TP size > 1 时执行 all_reduce；不修改输入 ID 或模型权重。
        """
        if self.tp_size > 1:
            # 非本 rank 的 token 临时查局部第 0 行，随后将其贡献清零。
            input_mask = (input_ < self.vocab_start_idx) | (
                input_ >= self.vocab_end_idx
            )
            masked_input = (input_ - self.vocab_start_idx).masked_fill(input_mask, 0)
        else:
            masked_input = input_  # 单卡直接使用全局词表行号。
        # Runner 的 token ID 可以为 int32，沿用 vLLM 转为 long 后查表。
        output_parallel = F.embedding(masked_input.long(), self.weight)
        if self.tp_size > 1:
            output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
            # 每个 token 只有所属 TP rank 提供非零贡献，组内求和恢复完整 embedding。
            output_parallel = tensor_model_parallel_all_reduce(output_parallel)
        return output_parallel


class ParallelLMHead(VocabParallelEmbedding):
    """保存词表投影权重，复用 embedding 的 TP 分片和权重加载方式。

    沿用 vLLM：本类不执行 hidden states 到 logits 的计算。模型应将此实例
    与已经选择好的 hidden states 一起传给 LogitsProcessor。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ) -> None:
        """创建输出词表权重，当前只支持不带 bias 的 Qwen3 路径。

        Args:
            num_embeddings: 完整输出词表大小。
            embedding_dim: 模型 hidden state 维度。
            bias: 当前必须为 False，未接入带 bias 的模型输出头。
        """
        assert not bias
        super().__init__(num_embeddings, embedding_dim)
        self.register_parameter("bias", None)  # 与上游无 bias 的 LMHead 状态一致。

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """拒绝直接调用；词表分数计算统一通过 LogitsProcessor 执行。"""
        del input_
        raise RuntimeError("LMHead's weights should be used in the sampler.")
