"""定义与持久 InputBatch 槽位对齐的采样元数据。"""

from dataclasses import dataclass, field

import torch

from nanovllm.sample.logits_processor import LogitsProcessors


@dataclass
class SamplingMetadata:
    """保存一次请求级采样所需的参数。

    参考 vLLM V1 的同名结构，当前只保留 temperature 和已有 processor 容器。
    不包含 greedy、top-k/top-p、独立随机种子、惩罚项或 logprobs 配置。

    Attributes:
        temperature: 与 logits 行顺序一致的设备 tensor，形状为 ``[num_reqs]``。
            它引用 InputBatch 的活跃 buffer 前缀；当前每个值必须为正数。
        logitsprocs: 已有的 logits processor 生命周期容器；默认没有具体处理器。
    """

    temperature: torch.Tensor  # 每个请求分别缩放自己的词表分数。
    logitsprocs: LogitsProcessors = field(default_factory=LogitsProcessors)
