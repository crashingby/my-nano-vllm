# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""模型层使用的 TP 集合通信入口，由 GroupCoordinator 选择实际通信器。"""

import torch

from .parallel_state import get_tp_group


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """在 TP 组内 SUM，返回各 rank 都持有的 output。

    当前基础通信器原地更新输入；调用方应始终使用返回值，以兼容后续通信后端。
    TP=1 直接返回 input_，不会发起通信。
    """
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """按 TP rank 顺序沿 dim 拼接输入，返回所有 rank 持有的 output。"""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> torch.Tensor | None:
    """沿 dim 汇集输入，仅组内 dst 返回 output，其他 rank 返回 None。"""
    return get_tp_group().gather(input_, dst, dim)
