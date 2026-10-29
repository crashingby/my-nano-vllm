# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""普通线性权重的参数类型，封装 TP 分片选择和 checkpoint 原地装填。

层负责确定全局尺寸、融合投影的本地偏移及 KV 复制数，参数负责按自身的
输入/输出轴执行切分、检查形状和复制。当前只接入普通权重相关类型，
不包含量化打包参数、scale 参数或其他设备的同步加载逻辑。
"""

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from nanovllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

__all__ = ["BasevLLMParameter", "RowvLLMParameter", "ModelWeightParameter"]


class BasevLLMParameter(nn.Parameter):
    """携带加载入口和 TP 状态的推理参数，默认只执行完整复制。

    weight_loader 仍由模型的 load_weights 调用，具体层的 weight_loader_v2 再
    调用本参数的 load_* 方法。这样不同存储布局可扩展自己的装填行为。

    Attributes:
        tp_rank: 本参数加载时使用的 TP 组内 rank，须与所属层保持一致。
        tp_size: 本参数的分片数；所属层禁用 TP 时同步为 1。
        weight_loader: 读取 checkpoint 时调用的层级加载器，可替换或删除。

    Note:
        与 vLLM 一样从当前 TP 组读取缺省状态。额外允许显式 tp_rank/tp_size，
        供 disable_tp 或独立构造层时使用；所属层仍会在创建及后处理后同步状态。
        is_sharded_weight=True 表示源权重已按 rank 切分，各 load_* 不再重复切分。
    """

    def __new__(
        cls,
        data: torch.Tensor | None,
        **kwargs: Any,
    ) -> "BasevLLMParameter":
        """创建 Parameter 对象并关闭梯度，其余元信息由 __init__ 保存。

        Returns:
            param: 与 data 共享底层存储的推理参数。
        """
        return super().__new__(cls, data=data, requires_grad=False)

    def __init__(
        self,
        data: torch.Tensor,
        weight_loader: Callable[..., None],
        *,
        tp_rank: int | None = None,
        tp_size: int | None = None,
    ) -> None:
        """保存加载回调与 TP 状态，不读取 checkpoint、不创建通信组。

        Args:
            data: 已分配的本地参数存储，具体数值由加载器后续填充。
            weight_loader: 模型加载权重时调用的回调，第一个参数是目标参数自身。
            tp_rank: 显式加载 rank，None 时查询当前 TP 组。
            tp_size: 显式分片数，None 时查询当前 TP 组。
        """
        self._weight_loader: Callable[..., None] | None = weight_loader
        self.tp_rank = (
            tp_rank if tp_rank is not None else get_tensor_model_parallel_rank()
        )
        self.tp_size = (
            tp_size if tp_size is not None else get_tensor_model_parallel_world_size()
        )

    @property
    def weight_loader(self) -> Callable[..., None]:
        """取得加载回调；删除后按普通属性约定抛出 AttributeError。"""
        if self._weight_loader is None:
            raise AttributeError(f"{type(self).__name__}.weight_loader has been deleted")
        return self._weight_loader

    @weight_loader.setter
    def weight_loader(self, value: Callable[..., None]) -> None:
        """替换加载回调，供模型或权重方法定制加载行为。"""
        self._weight_loader = value

    @weight_loader.deleter
    def weight_loader(self) -> None:
        """删除回调，使外部 getattr 的默认加载器回退仍然有效。"""
        self._weight_loader = None

    def _is_1d_and_scalar(self, loaded_weight: torch.Tensor) -> bool:
        """判断是否为单元素向量接收 checkpoint 标量的兼容情况。

        Returns:
            is_scalar: 目标形状为 [1] 且源形状为 [] 时为 True。
        """
        return (
            self.data.ndim == 1 and self.data.numel() == 1 and loaded_weight.ndim == 0
        )

    def _assert_and_load(self, loaded_weight: torch.Tensor) -> None:
        """校验形状后原地复制，禁止广播掩盖参数布局错误。"""
        assert (
            self.data.shape == loaded_weight.shape or self._is_1d_and_scalar(loaded_weight)
        ), (
            f"Cannot load shape {tuple(loaded_weight.shape)} into {tuple(self.shape)}"
        )
        self.data.copy_(loaded_weight)

    def load_column_parallel_weight(self, loaded_weight: torch.Tensor) -> None:
        """完整复制源权重；具有输出切分轴的子类负责重写分片逻辑。"""
        self._assert_and_load(loaded_weight)

    def load_row_parallel_weight(self, loaded_weight: torch.Tensor) -> None:
        """完整复制源权重；具有输入切分轴的子类负责重写分片逻辑。"""
        self._assert_and_load(loaded_weight)

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs: Any) -> None:
        """完整复制无需分段的参数，输出轴参数子类使用融合投影元信息。"""
        self._assert_and_load(loaded_weight)

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs: Any) -> None:
        """完整复制无需分段的参数，输出轴参数子类使用 Q/K/V 元信息。"""
        self._assert_and_load(loaded_weight)


class _ColumnvLLMParameter(BasevLLMParameter):
    """沿 output_dim 装填列并行、融合 MLP 和 QKV 权重的公共实现。"""

    def __init__(self, output_dim: int, **kwargs: Any) -> None:
        """记录输出轴，并沿多继承链继续初始化输入轴与公共参数状态。"""
        self._output_dim = output_dim  # 通常为 0，即 [out, in] 布局的输出轴。
        super().__init__(**kwargs)

    @property
    def output_dim(self) -> int:
        """返回 checkpoint 和本地参数的输出特征轴。"""
        return self._output_dim

    def load_column_parallel_weight(self, loaded_weight: torch.Tensor) -> None:
        """选择本参数 tp_rank 对应的输出分片，原地写入完整本地权重。"""
        if not getattr(self, "is_sharded_weight", False):
            shard_size = self.shape[self.output_dim]
            loaded_weight = loaded_weight.narrow(
                self.output_dim, self.tp_rank * shard_size, shard_size,
            )
        self._assert_and_load(loaded_weight)

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs: Any) -> None:
        """把一个原始投影的本 rank 分片写入本地融合权重的对应区间。

        Args:
            loaded_weight: 指定投影的完整 checkpoint，或已切好的本地分片。
            kwargs: shard_offset/shard_size 为层计算的本地输出偏移与长度，
                shard_id 为逻辑投影编号，普通权重无需额外使用该编号。
        """
        shard_offset: int = kwargs["shard_offset"]
        shard_size: int = kwargs["shard_size"]
        param_data = self.data.narrow(self.output_dim, shard_offset, shard_size)
        if not getattr(self, "is_sharded_weight", False):
            loaded_weight = loaded_weight.narrow(
                self.output_dim, self.tp_rank * shard_size, shard_size,
            )
        assert param_data.shape == loaded_weight.shape, (
            f"Merged weight shape mismatch: {tuple(param_data.shape)} "
            f"vs {tuple(loaded_weight.shape)}"
        )
        param_data.copy_(loaded_weight)  # 只覆盖一个投影，其他区间保留原值。

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs: Any) -> None:
        """按 Q/K/V 和 KV 副本组选择源分片，原地装填对应本地区间。

        Args:
            loaded_weight: 单个 Q、K 或 V 投影的 checkpoint tensor。
            kwargs: shard_offset/shard_size 为本地偏移与长度，shard_id 为 q/k/v。
                num_heads 沿用 vLLM 的参数名，实际表示 num_kv_head_replicas，
                即共享同一个 KV head 的相邻 rank 数，不是本地 attention head 数。
        """
        shard_offset: int = kwargs["shard_offset"]
        shard_size: int = kwargs["shard_size"]
        shard_id: str = kwargs["shard_id"]
        num_heads: int = kwargs["num_heads"]
        assert shard_id in ("q", "k", "v") and num_heads > 0
        param_data = self.data.narrow(self.output_dim, shard_offset, shard_size)
        if not getattr(self, "is_sharded_weight", False):
            # Q 按 rank 独立切分，K/V 先把共享同一源 head 的 rank 映射为一组。
            shard_rank = self.tp_rank if shard_id == "q" else self.tp_rank // num_heads
            loaded_weight = loaded_weight.narrow(
                self.output_dim, shard_rank * shard_size, shard_size,
            )
        assert param_data.shape == loaded_weight.shape, (
            f"QKV weight shape mismatch: {tuple(param_data.shape)} vs {tuple(loaded_weight.shape)}"
        )
        param_data.copy_(loaded_weight)


class RowvLLMParameter(BasevLLMParameter):
    """记录输入特征轴，负责按 TP rank 加载行并行权重。"""

    def __init__(self, input_dim: int, **kwargs: Any) -> None:
        """记录输入轴，再沿多继承链初始化公共加载状态。"""
        self._input_dim = input_dim  # 通常为 1，即 [out, in] 布局的输入轴。
        super().__init__(**kwargs)

    @property
    def input_dim(self) -> int:
        """返回 checkpoint 和本地参数的输入特征轴。"""
        return self._input_dim

    def load_row_parallel_weight(self, loaded_weight: torch.Tensor) -> None:
        """选择本参数 tp_rank 对应的输入分片，原地写入本地权重。"""
        if not getattr(self, "is_sharded_weight", False):
            shard_size = self.shape[self.input_dim]
            loaded_weight = loaded_weight.narrow(
                self.input_dim, self.tp_rank * shard_size, shard_size,
            )
        self._assert_and_load(loaded_weight)


class ModelWeightParameter(_ColumnvLLMParameter, RowvLLMParameter):
    """普通线性权重，同时具备输入轴和输出轴的 TP 加载能力。

    两个父类通过 super() 协作初始化，构造时传入 input_dim、output_dim、data
    和 weight_loader。具体层选择调用哪一种 load_*，不沿两个轴同时切分。
    仍是 nn.Parameter，forward 可直接交给 PyTorch F.linear 使用。
    """
