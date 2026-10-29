# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""模型参数的公共工具，不依赖具体模型层或权重加载器。"""

from typing import Any

import torch


def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: dict[str, Any] | None,
) -> None:
    """为权重或偏置添加加载元信息，沿用 vLLM 的禁止覆盖约定。

    Args:
        weight: 接收属性的 Tensor 或 Parameter，不改变其内容及存储。
        weight_attrs: 待添加的名称和值；None 表示没有额外属性。

    Raises:
        AssertionError: 待设置属性已经存在，包括参数类定义的属性。

    Side Effects:
        逐项在 weight 上新增属性；如需修改已有属性，调用方应显式赋值。
        当前 CUDA 路径无需其他平台的同步加载回调包装。
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        # 属性冲突通常意味着重复初始化，不能静默替换加载器或分片元信息。
        assert not hasattr(weight, key), f"Overwriting existing tensor attribute: {key}"
        setattr(weight, key, value)
