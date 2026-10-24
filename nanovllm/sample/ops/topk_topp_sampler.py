# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""保留 vLLM 同名模块中的编译随机采样路径；当前不实现 top-k/top-p。"""

import torch


@torch.compile(dynamic=True)
def compiled_random_sample(logits: torch.Tensor) -> torch.Tensor:
    """从每行 logits 的 softmax 分布中随机选择一个 token。

    直接参考 vLLM 的同名函数。令每个词表位置的独立噪声为 ``q ~ Exp(1)``，
    则 ``argmax(probs / q)`` 服从该行的分类分布；无需调用 multinomial。
    使用 PyTorch 编译路径，由编译器生成算子，不另写 Triton kernel。

    Args:
        logits: 已完成 temperature 缩放的 ``[num_reqs, vocab_size]`` 分数。

    Returns:
        sampled_token_ids: ``[num_reqs]`` 的 int64 token IDs；输入 logits 不会被修改。

    Side Effects:
        消耗所在设备的默认随机数生成器状态；不支持每请求独立 seed。
    """
    probs = logits.softmax(dim=-1, dtype=torch.float32)  # 用 FP32 计算分类概率。
    q = torch.empty_like(probs)  # 每个请求的每个词表位置都有独立噪声。
    q.exponential_()  # 默认 rate=1，对应 vLLM 的指数噪声采样。
    return probs.div(q).argmax(dim=-1).view(-1)
