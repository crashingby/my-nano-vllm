# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""使用 PyTorch process group 执行 TP 所需的基础设备集合通信。"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


class DeviceCommunicatorBase:
    """保存通信组身份，提供 PyTorch/NCCL 的基础实现。

    GroupCoordinator 拥有 CPU/device process group，本类只借用它们。
    保留 vLLM 的基础通信约定；当前不创建独立 NCCL communicator，
    也不管理自定义 all-reduce、EP all-to-all 或弹性进程组。

    Attributes:
        device: 当前 Worker 的设备描述。
        cpu_group: 用于查询组身份的 Gloo group，与 device_group 成员一致。
        device_group: 真正执行 tensor 通信的 process group。
        unique_name: 所属 GroupCoordinator 的唯一名称。
        rank: 当前进程在本通信组内的 rank。
        rank_in_group: 与 rank 相同；区别于 GroupCoordinator.rank 的全局语义。
        world_size: 本通信组的成员数量。
        ranks: 按组内 rank 排列的全局 rank 列表。
        global_rank: 当前进程在默认 world 中的 rank。
        global_world_size: 默认 world 的成员数量。
    """

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
    ) -> None:
        """借用已创建的进程组并保存 rank 元数据，不额外创建通信资源。

        Args:
            cpu_group: 与设备组成员相同的 Gloo process group。
            device: Worker 的 CUDA 设备描述；未提供时沿用基类的 CPU 描述默认值。
            device_group: tensor 通信组；None 沿用 PyTorch 的默认 process group。
            unique_name: 逻辑组的唯一名称，例如 tp:0。
        """
        self.device = device or torch.device("cpu")
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.unique_name = unique_name
        self.rank = dist.get_rank(cpu_group)  # 此处是组内编号，不是全局 rank。
        self.world_size = dist.get_world_size(cpu_group)
        self.ranks = dist.get_process_group_ranks(cpu_group)
        self.global_rank = dist.get_rank()
        self.global_world_size = dist.get_world_size()
        self.rank_in_group = dist.get_group_rank(cpu_group, self.global_rank)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """在设备组内原地 SUM 并返回 input_，所有 rank 获得归约结果。"""
        dist.all_reduce(input_, group=self.device_group)
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """按组内 rank 顺序汇集等形输入，返回沿 dim 拼接的 output_tensor。

        通信先按第 0 维接收各 rank 的输入，再把 rank 轴移至拼接位置。
        例如 [B, V/TP] 的 logits 在 dim=-1 时变成 [B, V]。
        分配新输出，不修改 input_；维度校验由 GroupCoordinator 完成。
        """
        if dim < 0:
            dim += input_.dim()  # 后续 shape 切片使用非负维度。
        input_size = input_.size()
        # 沿用 vLLM 的 concat 接收布局，避免 stack 布局与编译路径的兼容问题。
        output_size = (input_size[0] * self.world_size,) + input_size[1:]
        output_tensor = torch.empty(
            output_size, dtype=input_.dtype, device=input_.device
        )
        dist.all_gather_into_tensor(output_tensor, input_, group=self.device_group)
        # 显式拆出 rank 轴，再把该轴与目标维度合并，保持各 rank 的拼接顺序。
        output_tensor = output_tensor.reshape((self.world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim]
            + (self.world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )
        return output_tensor

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        """仅在组内 dst 接收各 rank 输入，返回拼接后的 output_tensor。

        input_ 在各 rank 上必须等形同 dtype；dst 是组内编号，发送时映射成
        ranks[dst]。目标 rank 分配接收缓冲区，其他 rank 返回 None。
        """
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )
        if dim < 0:
            dim += input_.dim()
        gather_list = (
            [torch.empty_like(input_) for _ in range(self.world_size)]
            if self.rank_in_group == dst
            else None
        )
        dist.gather(
            input_, gather_list, dst=self.ranks[dst], group=self.device_group
        )
        if self.rank_in_group == dst:
            output_tensor = torch.cat(gather_list, dim=dim)
        else:
            output_tensor = None
        return output_tensor

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """从组内 src 广播并原地更新 tensor，返回同一个 tensor。"""
        if self.world_size == 1:
            return tensor
        dist.broadcast(tensor, src=self.ranks[src], group=self.device_group)
        return tensor

    def destroy(self) -> None:
        """结束通信器生命周期；基础实现没有独立资源，不销毁借用的进程组。"""
        pass
