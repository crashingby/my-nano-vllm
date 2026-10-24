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
        sampled_token_ids: 回收后每行是 ``[token_id]`` 或 ``[]``，不包含 prompt。
            异步包装器回收前整个列表可以为空，不能提前当作已完成结果读取。
        cudagraph_stats: 可选 Graph 容量统计，缺省不采集。
    """
    # 按本批输出行排列的请求 ID，不随 InputBatch 后续重排而改变。
    req_ids: list[str]
    # Scheduler 通过请求 ID 找到本批输出行，不能使用当前 Runner 的槽位。
    req_id_to_index: dict[str, int]
    # 行列表保留无效 prefill 的空输出；异步回收前允许整个列表尚未填入。
    sampled_token_ids: list[list[int]] = field(default_factory=list)

    cudagraph_stats: CUDAGraphStat | None = None

class AsyncModelRunnerOutput(ABC):
    """持有尚未回收的执行结果，向 Executor 提供一次性阻塞回收接口。"""

    @abstractmethod
    def get_output(self) -> ModelRunnerOutput:
        """等待结果就绪，并返回可在 CPU 上消费的 ModelRunnerOutput。

        Returns:
            model_runner_output: 可在 CPU 消费的 ModelRunnerOutput；调用前可能需要等待设备到主机的拷贝。

        Side Effects:
            可能等待设备到 CPU 的拷贝并释放临时设备引用。每个包装器只应调用
            一次；需要重复 result() 时由 Executor 的 Future 缓存回收结果。
        """
        pass

EMPTY_MODEL_RUNNER_OUTPUT = ModelRunnerOutput(req_ids=[], req_id_to_index={})
