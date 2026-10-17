"""vLLM V1 Sampler 的 temperature-only 裁剪实现。"""

import torch
from torch import nn

from nanovllm.outputs import SamplerOutput
from nanovllm.sample.metadata import SamplingMetadata
from nanovllm.sample.ops.topk_topp_sampler import compiled_random_sample


class Sampler(nn.Module):
    """将每个请求的一行词表 logits 转为一个设备端候选 token。

    沿用 vLLM 的 ``forward -> apply_logits_processors -> sample`` 流程。
    Sampler 只认识 logits 行和采样参数，不判断请求是否完成 prefill；无效
    chunk 的候选由 GPUModelRunner 的 bookkeeping 丢弃。
    """

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        """处理词表分数并返回 ``[num_reqs, 1]`` 的 int32 候选 token。

        Args:
            logits: ``[num_reqs, vocab_size]`` 的模型输出，行序与 metadata 一致。
            sampling_metadata: 当前 batch 的 temperature 和已有 processor 容器。

        Returns:
            保留在 logits 所在设备上的 SamplerOutput，不执行 CPU 同步。

        Side Effects:
            输入原本为 FP32 时，temperature 和 processor 可原地修改 logits；
            FP16/BF16 输入会先转换为独立的 FP32 tensor。采样消耗默认 RNG 状态。
        """
        logits = logits.to(torch.float32)  # 对齐 vLLM，采样计算统一使用 FP32。
        logits = self.apply_logits_processors(logits, sampling_metadata)
        sampled = self.sample(logits, sampling_metadata)  # 每行选择一个词表下标。
        sampled = sampled.to(torch.int32)  # 缩小返回 tensor，与 vLLM 输出类型一致。
        return SamplerOutput(sampled_token_ids=sampled.unsqueeze(-1))

    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        """在 temperature 缩放前运行会影响 argmax 的已有处理器。

        当前默认容器为空；保留调用边界，与 InputBatch 的 update_state 配套。
        处理器可以原地修改输入，也可以返回替代 tensor。
        """
        for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
            logits = processor.apply(logits)
        return logits

    @staticmethod
    def apply_temperature(logits: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
        """按请求原地执行 ``logits[i] /= temp[i]``，返回同一个 tensor。

        当前只支持正 temperature，省去 vLLM 对混合 greedy 请求的零温度替换。
        小于 1 会使分布更集中，大于 1 会使分布更平缓。
        """
        return logits.div_(temp.unsqueeze(dim=1))

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        """缩放 temperature 后随机采样，返回 ``[num_reqs]`` token IDs。

        当前无 logprobs，因此省去 vLLM 返回元组中的 processed_logprobs。
        本方法会原地缩放 logits，并消耗默认设备 RNG 状态。
        """
        logits = self.apply_temperature(logits, sampling_metadata.temperature)
        # 对齐 vLLM：不影响 argmax 的处理器放在 temperature 缩放之后。
        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)
        return compiled_random_sample(logits)
