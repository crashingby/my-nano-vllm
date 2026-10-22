from __future__ import annotations

from dataclasses import dataclass, field
from nanovllm.compilation.cuda_graph import CUDAGraphStat
from abc import ABC, abstractmethod
import torch


@dataclass
class SamplerOutput:
    """Sampler 返回的设备端候选结果。

    ``sampled_token_ids`` 是 ``[num_reqs, 1]`` 的 int32 tensor，行顺序与
    InputBatch 相同。未完成 prefill 的请求此时仍有候选 token，Runner 随后
    会清除对应行；当前没有投机解码 padding 或 logprobs 字段。
    """

    sampled_token_ids: torch.Tensor  # 保留在采样设备上，不在 Sampler 内转为 list。


@dataclass
class ModelRunnerOutput:
    """一次 ModelRunner 执行产生的采样结果。

    当前 MVP 不支持投机解码，因此每个可采样请求在一次 step 中最多产生一个
    token。所有列表按 InputBatch 的请求槽位顺序对齐；未完成 chunked prefill
    的请求仍占一行，但其结果为 ``[]``。

    Attributes:
        req_ids: 当前 batch 的请求 ID 快照。
        req_id_to_index: 请求 ID 到输出行下标的快照。
        sampled_token_ids: 每行是 ``[token_id]`` 或 ``[]``，不包含 prompt。
    """
    # [num_reqs]
    req_ids: list[str]
    # req_id -> index
    req_id_to_index: dict[str, int]
    # 这里不考虑投机解码，但是元素还是list[int]
    sampled_token_ids: list[list[int]] = field(default_factory=list)

    cudagraph_stats: CUDAGraphStat | None = None

# ModelRunnerOutput wrapper for async scheduling.
class AsyncModelRunnerOutput(ABC):
    @abstractmethod
    def get_output(self) -> ModelRunnerOutput:
        """Get the ModelRunnerOutput for this async output.

        This is a blocking call that waits until the results are ready, which
        might involve copying device tensors to the host.
        This method should only be called once per AsyncModelRunnerOutput.
        """
        pass

EMPTY_MODEL_RUNNER_OUTPUT = ModelRunnerOutput(req_ids=[], req_id_to_index={})
