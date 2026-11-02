# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""按词表行分片的输入 embedding、输出投影参数及其布局与加载协议。

权重采用 [词表行数, 隐藏宽度] 布局，每个 TP rank 保存部分词表行的完整向量。
输入查表将非本地位置清零后求和；输出投影产生本地词表分数，由 LogitsProcessor
组织汇集。原始词表与追加词表分别补齐、分别分片，再接成一份本地参数。
当前保留追加区域协议，尚未接入适配器加载、具体量化算法或融合查表扩展。
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from nanovllm.distributed import (
    GroupCoordinator,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from nanovllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
    method_has_implemented_embedding,
)
from nanovllm.model_executor.layers.utils import dispatch_unquantized_gemm
from nanovllm.model_executor.parameter import BasevLLMParameter
from nanovllm.model_executor.utils import set_weight_attrs
from nanovllm.utils.math_utils import divide

DEFAULT_VOCAB_PADDING_SIZE = 64  # 全局词表区域的补齐倍数，不是额外增加的行数。


class UnquantizedEmbeddingMethod(QuantizeMethodBase):
    """为词表层提供普通浮点参数的创建、查表、投影与共享操作。

    方法对象不保存独立权重或 TP 状态，实际参数注册在传入的 layer 上。
    embedding 接收本地下标，apply 产生本地分数；ID 映射、无效位置清零及
    跨 rank 通信由外层负责。该方法同时服务于输入词表层与输出 LMHead。
    """

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        """按本地词表尺寸创建参数，并把层提供的加载回调绑定到参数上。

        Args:
            layer: 接收 weight 的词表层。
            input_size_per_partition: 本地每行的隐藏宽度，与全局隐藏宽度相同。
            output_partition_sizes: 本地输出分区的行数列表，和为原始与追加区域
                的本地分配总行数，包含两种 padding。
            input_size: 全局隐藏宽度；供参数工厂协议使用，本方法按本地尺寸分配。
            output_size: 原始与追加区域补齐后的全局总行数，包含所有 rank。
            params_dtype: 参数每个元素的浮点类型。
            extra_weight_attrs: 附加在 Parameter 上的加载元信息，其中 weight_loader
                是层提供的绑定回调，负责从 checkpoint 选择并装填本地词表行。

        Side Effects:
            在 layer 上注册未初始化数值、关闭梯度的 weight；附加轴编号和加载回调。
        """
        # 只分配本地词表行，隐藏轴保持完整；具体值由之后的 checkpoint 加载装填。
        weight = nn.Parameter(
            torch.empty(
                sum(output_partition_sizes), input_size_per_partition, dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})  # 轴编号，不是轴长度。
        layer.register_parameter("weight", weight)  # 使参数参与模块遍历和模型加载。
        set_weight_attrs(weight, extra_weight_attrs)  # 让加载器从参数上找到分片装填回调。

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """使用本地词表权重执行输出投影；TP 汇集由 LogitsProcessor 负责。

        Args:
            layer: 持有本地 [R, H] 权重的输出头，R 包含本地 padding。
            x: 形状 [..., H] 的隐藏向量，最后一维与完整隐藏宽度一致。
            bias: 可选的 [R] 本地偏置，沿输入的前导维度广播。

        Returns:
            logits: 形状 [..., R]，等于 x @ layer.weight.T + bias；列索引仍是
                本地物理行号，未转换为全局 token 顺序，也未移除 padding。
        """
        return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        """按本地整数行号读取权重向量，不处理 token 归属或跨 rank 合并。

        Args:
            layer: 持有本地词表权重的层。
            input_: 已转换为合法本地下标的整数 tensor；非本地 token 必须由
                调用方先映射到安全行号，并在查表之后清零对应输出位置。

        Returns:
            output: input_.shape + [embedding_dim] 的本地查表结果。
        """
        return F.embedding(input_, layer.weight)

    def tie_weights(self, layer: nn.Module, embed_tokens: nn.Module) -> nn.Module:
        """让输出头共享输入 embedding 的同一个参数对象。

        Args:
            layer: 接收共享权重的输出头，其词表分片布局须与 embed_tokens 相容。
            embed_tokens: 提供本地 weight 的输入词表层。

        Returns:
            layer: 完成参数共享的输出头；不复制底层存储。

        Side Effects:
            替换 layer.weight 的参数引用，偏置和其他状态不参与共享。
        """
        layer.weight = embed_tokens.weight
        return layer


def pad_vocab_size(vocab_size: int, pad_to: int = DEFAULT_VOCAB_PADDING_SIZE) -> int:
    """按指定倍数向上补齐词表区域的行数。

    Args:
        vocab_size: 当前区域需要容纳的非负行数。
        pad_to: 正整数补齐倍数；这里只处理对齐，TP 整除约束另行检查。

    Returns:
        vocab_size_padded: 不小于 vocab_size 的最小 pad_to 倍数。
    """
    return ((vocab_size + pad_to - 1) // pad_to) * pad_to


def vocab_range_from_per_partition_vocab_size(
    per_partition_vocab_size: int, rank: int, offset: int = 0
) -> Sequence[int]:
    """从均等分片的本地容量计算当前 rank 在指定坐标系中的分配区间。

    Args:
        per_partition_vocab_size: 每个 rank 为该区域分配的行数，包含 padding。
        rank: 当前通信组内从零开始的 rank。
        offset: 整个区域的坐标原点；原始区域从零计数，追加区域从原始有效
            词表大小计数，因此返回值不一定是本地参数中的物理行号。

    Returns:
        (vocab_start_index, vocab_end_index): 左闭右开的分配区间，等于
            (index_f + offset, index_l + offset)，长度为本地分配行数。
    """
    index_f = rank * per_partition_vocab_size  # 在当前区域内部按 rank 排列分片。
    index_l = index_f + per_partition_vocab_size  # 终点不包含在本分片内。
    return index_f + offset, index_l + offset


def vocab_range_from_global_vocab_size(
    global_vocab_size: int, rank: int, world_size: int, offset: int = 0
) -> Sequence[int]:
    """将一个区域的全局分配容量均分，返回当前 rank 的分配边界。

    Args:
        global_vocab_size: 该区域补齐后的全局行数，须能整除 world_size。
        rank: 当前通信组内的分片编号。
        world_size: 共同分担该区域的 rank 数量，须为正整数。
        offset: 该区域起点在返回区间所用坐标系中的偏移。

    Returns:
        (vocab_start_index, vocab_end_index): 当前 rank 的左闭右开分配区间，
            区间长度为 global_vocab_size / world_size，尚未裁去 padding。
    """
    # divide 同时检查整除，避免各 rank 分配容量不等或遗漏尾部行。
    per_partition_vocab_size = divide(global_vocab_size, world_size)
    return vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size, rank, offset=offset,
    )


@dataclass
class VocabParallelEmbeddingShardIndices:
    """描述一个 TP rank 在原始、追加词表中的有效 token 范围和分配容量。

    四组区间均左闭右开：不带 padded 的端点表示有效全局 token ID，带 padded
    的端点表示各区域均分后的概念区间，允许终点超出有效 token 范围。
    原始区域从零计数，追加区域以原始有效词表大小为偏移；两种 padded 区间
    不能解释成同一个物理数组的坐标，必须分别求长度后安排本地存储。

    本地参数先存原始区域，再存追加区域。设 q_o、q_a 是两段的分配长度，
    n_o、n_a 是两段的有效长度，则原始有效行在 [0, n_o)，追加有效行在
    [q_o, q_o + n_a)，余下位置是 padding。num_* 属性从端点派生这些长度。
    该对象只保存布局元数据，不持有权重，也不执行查表或通信。
    """

    padded_org_vocab_start_index: int  # 原始区域分配起点：rank * 原始区域本地容量。
    padded_org_vocab_end_index: int  # 原始分配终点，允许越过原始有效词表末尾。
    padded_added_vocab_start_index: int  # 追加区域分配起点，含原始有效词表大小偏移。
    padded_added_vocab_end_index: int  # 追加分配终点，允许越过全部有效 token 的末尾。

    org_vocab_start_index: int  # 本 rank 拥有的原始 token 起点；纯 padding 时截到末尾。
    org_vocab_end_index: int  # 原始有效 token 终点，不包含此 ID。
    added_vocab_start_index: int  # 本 rank 拥有的追加 token 起点，使用全局 token ID。
    added_vocab_end_index: int  # 追加有效 token 终点，等于起点时本 rank 无追加 token。

    @property
    def num_org_elements(self) -> int:
        """返回 num_org_elements：本 rank 的原始有效行数，不含原始 padding。"""
        return self.org_vocab_end_index - self.org_vocab_start_index

    @property
    def num_added_elements(self) -> int:
        """返回 num_added_elements：本 rank 的追加有效行数，不含追加 padding。"""
        return self.added_vocab_end_index - self.added_vocab_start_index

    @property
    def num_org_elements_padded(self) -> int:
        """返回 num_org_elements_padded：本地原始区域总行数，也是追加区域的本地起点。"""
        return self.padded_org_vocab_end_index - self.padded_org_vocab_start_index

    @property
    def num_added_elements_padded(self) -> int:
        """返回 num_added_elements_padded：本地追加区域的容量，包含有效行与 padding。"""
        return self.padded_added_vocab_end_index - self.padded_added_vocab_start_index

    @property
    def num_org_vocab_padding(self) -> int:
        """返回 num_org_vocab_padding：本地原始有效行之后、追加区域之前的空行数。"""
        return self.num_org_elements_padded - self.num_org_elements

    @property
    def num_added_vocab_padding(self) -> int:
        """返回 num_added_vocab_padding：本地追加有效行之后的尾部空行数。"""
        return self.num_added_elements_padded - self.num_added_elements

    @property
    def num_elements_padded(self) -> int:
        """返回 num_elements_padded：本地参数总行数，包含两段有效行和两种 padding。"""
        return self.num_org_elements_padded + self.num_added_elements_padded

    def __post_init__(self) -> None:
        """检查区间端点有序、有效范围经过上界裁剪且有效长度不超过分配容量。

        纯 padding 分片的有效起止点都截到词表末尾，可能小于其分配起点，
        因此不能要求有效空区间的起点仍落在原分配区间内部。

        Raises:
            AssertionError: 端点顺序或有效长度与分配容量不一致。
        """
        # 左闭右开区间允许为空，但起点不能大于终点。
        assert self.padded_org_vocab_start_index <= self.padded_org_vocab_end_index
        assert self.padded_added_vocab_start_index <= self.padded_added_vocab_end_index
        assert self.org_vocab_start_index <= self.org_vocab_end_index
        assert self.added_vocab_start_index <= self.added_vocab_end_index
        # 有效端点通过 min(分配端点, 有效词表末尾) 得到，不会大于分配端点。
        assert self.org_vocab_start_index <= self.padded_org_vocab_start_index
        assert self.added_vocab_start_index <= self.padded_added_vocab_start_index
        assert self.org_vocab_end_index <= self.padded_org_vocab_end_index
        assert self.added_vocab_end_index <= self.padded_added_vocab_end_index
        # 即使端点因纯 padding 被整体截断，有效行数也不能超过该段本地容量。
        assert self.num_org_elements <= self.num_org_elements_padded
        assert self.num_added_elements <= self.num_added_elements_padded


@torch.compile(dynamic=True)
def get_masked_input_and_mask(
    input_: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把全局 token ID 转成本地参数行号，并找出不属于当前 rank 的位置。

    原始 token 减去原始有效起点；追加 token 减去追加有效起点，再加上本地
    原始区域的分配长度。非本地 token 临时映射到本地第零行以保证查表安全，
    调用方必须使用返回的 mask 清零这些位置，才能正确合并各 rank 的贡献。

    Args:
        input_: 全局 token ID 的整数 tensor，各 TP rank 接收相同的输入位置。
        org_vocab_start_index: 本 rank 原始有效 token 区间的包含起点。
        org_vocab_end_index: 本 rank 原始有效 token 区间的不包含终点。
        num_org_vocab_padding: 本地原始区域末尾的 padding 行数，用于计算
            追加区域在本地参数中的真实起始行号。
        added_vocab_start_index: 本 rank 追加有效 token 区间的包含起点。
        added_vocab_end_index: 本 rank 追加有效 token 区间的不包含终点。

    Returns:
        (masked_input, input_mask): 两个与输入形状相同的 tensor。masked_input
            是可用于本地查表的行号；input_mask 是布尔标记，True 表示不属于
            当前 rank，查表后必须把该输入位置的完整向量清零。
    """
    # 归属判断只使用有效 token 区间；补齐位置不应被当成真实 token 查询。
    org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ < org_vocab_end_index)
    added_vocab_mask = (input_ >= added_vocab_start_index) & (
        input_ < added_vocab_end_index
    )
    # 追加 token 的本地行号 = 全局 ID - 追加起点 + 原始区域本地容量。
    # 将后两项合并成减数，便于与原始分支共用后面的逐元素索引表达式。
    added_offset = (
        added_vocab_start_index
        - (org_vocab_end_index - org_vocab_start_index)
        - num_org_vocab_padding
    )
    # 两个有效区间互斥，每个输入位置只选择所属区域的偏移；非本地偏移为零。
    valid_offset = (org_vocab_start_index * org_vocab_mask) + (
        added_offset * added_vocab_mask
    )
    vocab_mask = org_vocab_mask | added_vocab_mask  # 任一区域命中即为本 rank 的有效贡献。
    input_ = vocab_mask * (input_ - valid_offset)  # 非本地位置统一改为安全行号零。
    return input_, ~vocab_mask


class VocabParallelEmbedding(nn.Module):
    """按词表行分配权重，组织本地参数加载、全局 ID 查表与跨 rank 归约。

    权重布局为 [词表行数, embedding_dim]。TP 只切行，每个 token 的完整
    隐藏向量保存在一个 rank；forward 让非本地输入位置贡献零向量，再用
    all-reduce 求和，使各 rank 都能取得相同的完整 embedding。

    原始词表与追加词表分别补齐、分别按 TP 均分，随后接成本地参数。
    追加权重独立占据原始补齐区域之后的空间，便于与基础 checkpoint 分开装填。
    定义 V_o = org_vocab_size、V_a = num_added_embeddings、H = embedding_dim，
    P = tp_size。令 pad(n) 表示向上补到 padding_size 的倍数，则：

        V = V_o + V_a                  = num_embeddings
        V_op = pad(V_o)                = org_vocab_size_padded
        V_p = pad(V_op + V_a)          = num_embeddings_padded
        q_o = V_op / P                 = 本地原始区域容量
        q_a = (V_p - V_op) / P         = 本地追加区域容量
        R = q_o + q_a                  = num_embeddings_per_partition

    不分片时的物理行布局（“-”表示无有效 token；每行宽度均为 H）：

        region   | original | org padding | added              | added padding |
        token ID | [0, V_o) |      -      | [V_o, V)           |       -       |
        row      | [0, V_o) | [V_o, V_op) | [V_op, V_op + V_a) | [V_op+V_a, V_p) |

    对任意 rank，令 [s_o, e_o)、[s_a, e_a) 分别为其有效原始、追加 token
    区间，n_o = e_o - s_o，n_a = e_a - s_a。本地 weight.shape = [R, H]：

        region   | original   | org padding | added              | added padding |
        token ID | [s_o, e_o)  |      -      | [s_a, e_a)         |       -       |
        local row| [0, n_o)    | [n_o, q_o)  | [q_o, q_o + n_a)   | [q_o+n_a, R)  |

    num_org_embeddings_per_partition 和 num_added_embeddings_per_partition
    分别记录 n_o、n_a；num_embeddings_per_partition 记录包含 padding 的 R。
    input_size 与 input_size_per_partition 都是 H，output_size 是 V_p，
    output_size_per_partition 是 R：这些名称沿用输出投影的线性参数协议。

    构造阶段只创建空参数并绑定加载回调；weight_loader 装填原始有效行，
    将剩余行清零。当前保留追加区域的布局和索引协议，但尚未接入完整适配器
    加载与输出重排流程；单卡直接查表分支也未处理原始 padding 引起的追加 ID 间隙。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        disable_tp: bool = False,
        reduce_results: bool = True,
        quant_method: QuantizeMethodBase | None = None,
        parallel_group: GroupCoordinator | None = None,
    ) -> None:
        """建立本层的词表分片布局，并创建带有加载回调的本地参数。

        Args:
            num_embeddings: 原始与追加 token 的总数，不含 padding。
            embedding_dim: 完整 hidden size，词表 TP 不切输入轴。
            params_dtype: 参数精度，None 使用默认浮点精度。
            org_num_embeddings: 原始有效 token 数；缺省时全部 token 都视为原始
                词表。传入时须不大于 num_embeddings，差值作为追加有效 token 数。
            padding_size: 正整数补齐倍数；原始区域与追加区域的补齐容量
                还须分别能被 TP 大小整除。
            quant_config: 未显式传 quant_method 时，用于按层选择参数方法；
                未配置或选择结果为 None 时回退到普通浮点方法。
            prefix: 模型中的完整参数层路径。
            disable_tp: 是否让当前层按 rank=0、size=1 构造，不改变其他层的 TP。
            reduce_results: 是否对各 rank 的查表贡献求和；关闭时返回仅保留
                本地 token 贡献、其余位置为零的 tensor，由调用方自行合并。
            quant_method: 显式指定的方法，优先于配置分派。
            parallel_group: 可选的查表通信组，缺省使用 TP 组。

        Side Effects:
            查询组信息，保存布局和方法状态，注册参数及加载回调；具体参数值由
            后续 checkpoint 加载装填。构造过程不发起查表结果的 collective。
        """
        super().__init__()

        # 分片所属组同时决定 token 归属和结果归约范围，两者必须使用相同组。
        self.disable_tp = disable_tp  # 只控制本层是否保留完整词表。
        self.parallel_group = parallel_group  # None 表示使用默认 TP 组。
        self.reduce_results = reduce_results  # 控制 forward 返回完整结果还是局部贡献。
        if disable_tp:
            # 显式关闭本层分片时，无需查询任何通信组的 rank/size。
            tp_rank, self.tp_size = 0, 1
        elif parallel_group is not None:
            # 使用组内编号，而不是全局进程 rank，避免跨组偏移污染词表切片。
            tp_rank = parallel_group.rank_in_group
            self.tp_size = parallel_group.world_size
        else:
            tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = tp_rank  # 本层在所选组内的分片编号。
        self.num_embeddings = num_embeddings  # 全局有效 token 总数，不含任何 padding。
        self.padding_size = padding_size  # 原始容量和最终总容量的补齐倍数。
        self.org_vocab_size = org_num_embeddings or num_embeddings  # 基础 checkpoint 的词表行数。
        self.num_added_embeddings = num_embeddings - self.org_vocab_size  # 全局追加有效行数。
        # 原始区域先独立补齐，追加行不能复用这部分尾部 padding。
        self.org_vocab_size_padded = pad_vocab_size(
            self.org_vocab_size, self.padding_size,
        )
        # 在已补齐的原始区域之后安排追加有效行，再补齐整体容量。
        self.num_embeddings_padded = pad_vocab_size(
            self.org_vocab_size_padded + self.num_added_embeddings, self.padding_size,
        )
        assert self.org_vocab_size_padded <= self.num_embeddings_padded  # 追加容量不能为负。

        # 区间用于判断全局 token 归属；两段分配长度用于计算本地物理偏移。
        self.shard_indices = self._get_indices(
            self.num_embeddings_padded,
            self.org_vocab_size_padded,
            self.num_embeddings,
            self.org_vocab_size,
            tp_rank,
            self.tp_size,
        )
        self.embedding_dim = embedding_dim  # 每个 token 的完整向量宽度，TP 不切此轴。
        # 所有 rank 的本地参数形状相同，但各自的有效行数可以不同。
        self.num_embeddings_per_partition = divide(
            self.num_embeddings_padded, self.tp_size,
        )

        # 参数方法共用线性层工厂协议，这组元信息必须在方法选择前准备好。
        self.input_size = self.input_size_per_partition = embedding_dim  # 全局/本地隐藏宽度。
        self.output_size = self.num_embeddings_padded  # 全局词表输出容量，包含补齐行。
        self.output_size_per_partition = self.num_embeddings_per_partition  # 本地输出容量。
        self.output_partition_sizes = [self.output_size_per_partition]  # 整张本地表作为一个输出分区。
        self.prefix = prefix  # 参数方法选择时使用的完整层路径。

        # 显式方法优先；否则询问配置，未选中任何方法时使用普通浮点实现。
        if quant_method is None and quant_config is not None:
            quant_method = quant_config.get_quant_method(self, prefix)
        if quant_method is None:
            quant_method = UnquantizedEmbeddingMethod()

        # 输入 embedding 必须能查表；LMHead 只需能执行输出投影。
        is_embedding_layer = not isinstance(self, ParallelLMHead)
        quant_method_implements_embedding = method_has_implemented_embedding(
            type(quant_method),
        )
        if is_embedding_layer and not quant_method_implements_embedding:
            raise NotImplementedError(
                f"The class {type(quant_method).__name__} must implement "
                "the 'embedding' method, see UnquantizedEmbeddingMethod."
            )
        self.quant_method: QuantizeMethodBase = quant_method  # 负责参数创建和本地运算。

        if params_dtype is None:
            # 模型加载器可通过默认 dtype 上下文统一控制未显式指定的参数精度。
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype  # 后续输出头偏置也使用相同参数类型。
        # 两段各自分片后的容量之和，必须与总容量均分得到的本地行数一致。
        assert (
            self.shard_indices.num_elements_padded == self.num_embeddings_per_partition
        )
        # 下列两个计数只数有效行；它们的和可能小于包含 padding 的本地容量。
        self.num_org_embeddings_per_partition = (
            self.shard_indices.org_vocab_end_index
            - self.shard_indices.org_vocab_start_index
        )
        self.num_added_embeddings_per_partition = (
            self.shard_indices.added_vocab_end_index
            - self.shard_indices.added_vocab_start_index
        )

        # 尚未接入融合查表扩展，当前通过张量索引、查表和清零分别完成这些步骤。
        self.use_fused_embedding = False

        # 工厂注册空 weight 并附加绑定加载回调，不在此处读取 checkpoint。
        self.quant_method.create_weights(
            self,
            self.input_size_per_partition,
            self.output_partition_sizes,
            self.input_size,
            self.output_size,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader,
        )
        self.update_param_tp_status()  # 专用参数需要与本层使用相同的组内 rank/size。

    def update_param_tp_status(self) -> None:
        """将本层的组内 rank/size 写入具有 TP 加载语义的专用参数。

        在参数工厂完成创建后调用，避免参数自身记录的默认 TP 状态与本层的
        disable_tp 或自定义 parallel_group 不一致。普通 nn.Parameter 不处理。

        Side Effects:
            原地更新所有 BasevLLMParameter 的 tp_rank 和 tp_size，不重分配存储。
        """
        # 仅专用参数在自身加载方法里消费 TP 状态，普通参数由层回调负责分片。
        for param in self.parameters():
            if isinstance(param, BasevLLMParameter):
                param.tp_rank = self.tp_rank
                param.tp_size = self.tp_size

    @classmethod
    def _get_indices(
        cls,
        vocab_size_padded: int,
        org_vocab_size_padded: int,
        vocab_size: int,
        org_vocab_size: int,
        tp_rank: int,
        tp_size: int,
    ) -> VocabParallelEmbeddingShardIndices:
        """根据全局词表容量和 rank，计算两段分配范围及有效 token 范围。

        原始区域从全局 ID 零开始均分；追加区域从 org_vocab_size 开始均分。
        padded 端点保留分配容量，有效端点分别截到原始或全部 token 的末尾，
        使纯 padding 分片的有效区间为空。该方法只计算布局，不切分实际 tensor。

        Args:
            vocab_size_padded: 原始与追加区域补齐后的全局分配总行数。
            org_vocab_size_padded: 原始区域的全局分配行数，包含原始 padding。
            vocab_size: 原始和追加有效 token 总数，也是有效全局 ID 的排除上界。
            org_vocab_size: 原始有效 token 数，也是追加 token 全局 ID 的起点。
            tp_rank: 当前层所属通信组内的分片编号。
            tp_size: 均分两个区域的 rank 数；两段分配容量均须能被它整除。

        Returns:
            shard_indices: 当前 rank 的原始/追加有效全局 token 区间及对应的
                补齐分配区间。有效区间用于查表归属判断和 checkpoint 切片；
                补齐区间的长度用于确定本地两段容量及追加区域的物理起点。
        """
        # 差值包含追加有效行和追加 padding，原始 padding 仍留在原始区域内。
        num_added_embeddings_padded = vocab_size_padded - org_vocab_size_padded
        # 原始分配坐标从零起算，终点可能落在有效原始词表之外。
        padded_org_vocab_start_index, padded_org_vocab_end_index = (
            vocab_range_from_global_vocab_size(org_vocab_size_padded, tp_rank, tp_size)
        )
        # 追加分配坐标使用全局 token ID 的原点，不使用原始补齐后的物理偏移。
        padded_added_vocab_start_index, padded_added_vocab_end_index = (
            vocab_range_from_global_vocab_size(
                num_added_embeddings_padded, tp_rank, tp_size, offset=org_vocab_size,
            )
        )
        # 有效区间始终落在实际 token ID 范围内，纯 padding shard 的长度为零。
        org_vocab_start_index = min(padded_org_vocab_start_index, org_vocab_size)
        org_vocab_end_index = min(padded_org_vocab_end_index, org_vocab_size)
        added_vocab_start_index = min(padded_added_vocab_start_index, vocab_size)
        added_vocab_end_index = min(padded_added_vocab_end_index, vocab_size)
        return VocabParallelEmbeddingShardIndices(
            padded_org_vocab_start_index,
            padded_org_vocab_end_index,
            padded_added_vocab_start_index,
            padded_added_vocab_end_index,
            org_vocab_start_index,
            org_vocab_end_index,
            added_vocab_start_index,
            added_vocab_end_index,
        )

    def get_sharded_to_full_mapping(self) -> list[int] | None:
        """为按 rank 顺序拼接的词表行或分数列生成全局 token 顺序的读取索引。

        本地参数均按原始、追加两段排列，直接汇集会使某个 rank 的追加行
        位于后续 rank 的原始行之前。此方法将有效原始位置、有效追加位置和
        两种 padding 分别收集，保证重排后全部有效 token 按全局 ID 连续排列。
        本方法不发起通信、不改动参数，调用方须自行应用返回的索引。

        Returns:
            ret: 长度为 num_embeddings_padded 的物理位置排列，ret[j] 是新位置 j
                应从原汇集 tensor 读取的位置。用于权重时沿行索引，用于 logits
                时沿最后一维索引；padding 被集中放在有效 token 之后。
                tp_size < 2 时返回 None，表示本接口不提供单卡重排；不能据此
                推断含追加区域的单卡物理布局与 token ID 一定连续对应。
        """
        if self.tp_size < 2:
            return None

        base_embeddings: list[int] = []  # 汇集 tensor 中有效原始行的物理位置。
        added_embeddings: list[int] = []  # 汇集 tensor 中有效追加行的物理位置。
        padding: list[int] = []  # 两类 padding 的物理位置，最终统一移到尾部。
        # 所有 rank 的布局均可由相同全局尺寸重建，无需读取远端权重或发起通信。
        for tp_rank in range(self.tp_size):
            shard_indices = self._get_indices(
                self.num_embeddings_padded,
                self.org_vocab_size_padded,
                self.num_embeddings,
                self.org_vocab_size,
                tp_rank,
                self.tp_size,
            )
            range_start = self.num_embeddings_per_partition * tp_rank  # 本 rank 在汇集数组中的起点。
            range_end = self.num_embeddings_per_partition * (tp_rank + 1)  # 该物理分片的排除终点。
            # 原始有效行位于本地参数前部，其后到原始分配终点之间都是 padding。
            base_embeddings.extend(
                range(range_start, range_start + shard_indices.num_org_elements)
            )
            padding.extend(
                range(
                    range_start + shard_indices.num_org_elements,
                    range_start + shard_indices.num_org_elements_padded,
                )
            )
            # 追加有效行从原始区域的分配终点开始，必须跨过原始 padding。
            added_embeddings.extend(
                range(
                    range_start + shard_indices.num_org_elements_padded,
                    range_start
                    + shard_indices.num_org_elements_padded
                    + shard_indices.num_added_elements,
                )
            )
            # 追加有效终点到本 rank 分配终点之间的剩余空间是尾部 padding。
            padding.extend(
                range(
                    range_start
                    + shard_indices.num_org_elements_padded
                    + shard_indices.num_added_elements,
                    range_start
                    + shard_indices.num_org_elements_padded
                    + shard_indices.num_added_elements_padded,
                )
            )
            # 四段连续区域必须恰好覆盖该 rank 的物理分片，既不重叠也不遗漏。
            assert (
                range_start
                + shard_indices.num_org_elements_padded
                + shard_indices.num_added_elements_padded
                == range_end
            )
        ret = base_embeddings + added_embeddings + padding  # 有效 token 先按原始、追加顺序排好。
        assert len(ret) == self.num_embeddings_padded  # 重排包含全部物理位置，不在此裁剪 padding。
        return ret

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        """将原始 checkpoint 的词表参数装填到当前 rank，并清零未装填区域。

        该回调在参数创建时附加到 Parameter，由模型加载器读取源 tensor 后调用。
        普通权重按 output_dim 选取原始有效 token 的源行，写入本地参数前部；
        原始 padding、整个追加区域均清零。追加有效权重需要后续独立装填，
        当前没有对应的适配器加载流程。

        Args:
            param: 已分配的本地目标 Parameter；普通权重为 [R, H]，偏置为 [R]，
                两者 output_dim 均为零。未声明 output_dim 的参数按完整复制处理。
            loaded_weight: checkpoint 读出的原始词表源 tensor。普通输出轴长度
                须等于 org_vocab_size；打包输出轴长度须符合参数声明的打包因子。

        Side Effects:
            原地覆盖 param 的有效原始部分并清零剩余部分，不修改源 tensor。
            无输出轴参数检查完整形状后复制，不执行 TP 切片。
        """
        output_dim = getattr(param, "output_dim", None)  # 参数中的词表切分轴；缺省表示复制参数。
        packed_dim = getattr(param, "packed_dim", None)  # 打包轴决定词表偏移是否需换算存储单位。

        if output_dim is None:
            # 单元素目标允许接收零维源，先统一形状，避免误判为不兼容的参数。
            if (
                loaded_weight.ndim == 0
                and param.data.ndim == 1
                and param.data.numel() == 1
            ):
                loaded_weight = loaded_weight.reshape(1)
            assert param.data.shape == loaded_weight.shape  # 复制参数必须完整匹配，不隐式广播。
            param.data.copy_(loaded_weight)
            return

        start_idx = self.shard_indices.org_vocab_start_index  # 原始 checkpoint 中的有效行起点。
        shard_size = self.shard_indices.org_vocab_end_index - start_idx  # 只复制有效行，排除 padding。

        if packed_dim is not None and packed_dim == output_dim:
            # 词表轴打包时，逻辑 token 行偏移需按参数协议转换为源 tensor 存储偏移。
            packed_factor = (
                param.packed_factor
                if isinstance(param, BasevLLMParameter)
                else param.pack_factor
            )
            # 源须覆盖完整原始词表的打包存储，不能传入已经按 TP 切好的源分片。
            assert loaded_weight.shape[output_dim] == (
                self.org_vocab_size // param.packed_factor
            )
            start_idx = start_idx // packed_factor
            shard_size = shard_size // packed_factor
        else:
            assert loaded_weight.shape[output_dim] == self.org_vocab_size  # 普通源不包含追加区域。

        # narrow 只创建源视图；copy_ 才将当前 rank 的有效数据写入已有本地存储。
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        param[: loaded_weight.shape[0]].data.copy_(loaded_weight)
        # 清除空参数中的未初始化数据；追加区域必须在基础加载完成后另行装填。
        param[loaded_weight.shape[0] :].data.fill_(0)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """将全局 token ID 映射为本地查表贡献，并按配置合并各 rank 的结果。

        Args:
            input_: 任意形状的整数 token ID tensor，调用方须保证 ID 属于有效词表。
                多卡归约时各 rank 必须使用相同形状与相同顺序的输入位置。

        Returns:
            output_parallel: 形状为 input_.shape + [embedding_dim]。多卡开启
                reduce_results 时，各 rank 得到相同的完整 embedding；关闭时
                仅本 rank 拥有的 token 位置保留向量，其余位置为零。
                单卡分支直接返回参数方法的查表结果。

        Side Effects:
            原地清零新生成的局部输出；多卡且 reduce_results=True 时发起组内
            all-reduce，所有组成员必须按一致顺序调用。不修改输入 ID 或权重。
        """
        if self.tp_size == 1:
            # 单卡普通词表的全局 ID 可直接作本地下标；此分支不重映射追加区域。
            return self.quant_method.embedding(self, input_.long())

        # 先消除越界下标，并保留非本地位置的标记，供查表后清零使用。
        masked_input, input_mask = get_masked_input_and_mask(
            input_,
            self.shard_indices.org_vocab_start_index,
            self.shard_indices.org_vocab_end_index,
            self.shard_indices.num_org_vocab_padding,
            self.shard_indices.added_vocab_start_index,
            self.shard_indices.added_vocab_end_index,
        )
        output_parallel = self.quant_method.embedding(self, masked_input.long())  # 暂含安全行的占位值。

        output_dtype = output_parallel.dtype  # 记录返回元素类型；当前普通路径不改变其 dtype。
        # 每个输入位置的标记广播到整个向量；安全行通常非零，必须清除错误贡献。
        output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
        if self.reduce_results:
            # 同一 token 只由一个 rank 提供有效向量，其他 rank 贡献零，故使用求和归约。
            output_parallel = (
                self.parallel_group.all_reduce(output_parallel)
                if self.parallel_group is not None
                else tensor_model_parallel_all_reduce(output_parallel)
            )
        return output_parallel.view(output_dtype)

    def extra_repr(self) -> str:
        """返回 s，供模块打印展示有效词表数、全局/本地容量、隐藏宽度和 TP 大小。"""
        s = f"num_embeddings={self.num_embeddings}"
        s += f", num_embeddings_per_partition={self.num_embeddings_per_partition}"
        s += f", embedding_dim={self.embedding_dim}"
        s += f", org_vocab_size={self.org_vocab_size}"
        s += f", num_embeddings_padded={self.num_embeddings_padded}"
        s += f", tp_size={self.tp_size}"
        return s


class ParallelLMHead(VocabParallelEmbedding):
    """保存按词表行分片的输出投影权重及可选偏置。

    继承词表层的尺寸、分区和加载规则，本地 weight 为 [R, H]，可选 bias 为
    [R]。LogitsProcessor 通过 quant_method.apply 计算 [..., R] 的本地分数，
    再汇集各 rank 的词表列；本类不直接执行完整输出流程，forward 禁止直接调用。
    tie_weights 可使输出头复用输入 embedding 的本地 Parameter，但二者仍执行
    不同运算：输入侧查表，输出侧做隐藏向量与权重行的内积。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        disable_tp: bool = False,
        quant_method: QuantizeMethodBase | None = None,
    ) -> None:
        """创建本地词表投影参数，并按需分配与词表行对齐的偏置。

        Args:
            num_embeddings: 原始与追加有效 token 的全局总数，不含 padding。
            embedding_dim: 输入隐藏宽度，TP 不切分此轴。
            bias: 是否创建本地偏置；参与投影时需由调用方显式传给参数方法。
            params_dtype: 权重与偏置的元素类型；None 使用默认浮点类型。
            org_num_embeddings: 原始有效词表大小，缺省表示没有追加 token。
            padding_size: 词表行数的补齐倍数，两段容量还须满足 TP 整除约束。
            quant_config: 未显式指定方法时用于选择参数创建与投影实现的配置。
            prefix: 本输出头在模型中的完整层路径。
            disable_tp: 是否让本输出头保存完整词表并跳过分片。
            quant_method: 显式参数方法，优先于配置分派；输出头不要求支持查表。

        Side Effects:
            注册尚未装填的 weight 和可选 bias，绑定参数加载回调；不执行投影。
        """
        self.has_bias = bias  # 在父类选择参数方法前暴露输出头是否需要偏置。
        super().__init__(
            num_embeddings, embedding_dim, params_dtype, org_num_embeddings,
            padding_size, quant_config, prefix, disable_tp=disable_tp, quant_method=quant_method,
        )
        self.quant_config = quant_config  # 保留输出头所用配置，实际执行通过 quant_method。
        if bias:
            self._register_bias()
        else:
            self.register_parameter("bias", None)  # 保持稳定的参数属性，禁用时不分配存储。

    def _register_bias(self) -> None:
        """为每个本地词表物理行创建一个偏置元素，并绑定词表分片加载回调。

        Side Effects:
            注册形状为 [num_embeddings_per_partition] 的未装填 bias，包含
            padding 对应的位置。附加 output_dim=0，使其复用原始词表切片规则。
        """
        self.bias = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, dtype=self.params_dtype),
            requires_grad=False,
        )
        # 一维偏置的唯一轴就是词表轴，加载后有效元素与本地 weight 的有效行对齐。
        set_weight_attrs(self.bias, {"output_dim": 0, "weight_loader": self.weight_loader})

    def tie_weights(self, embed_tokens: VocabParallelEmbedding) -> nn.Module:
        """委托当前参数方法，将输出头权重与输入 embedding 建立共享关系。

        Args:
            embed_tokens: 提供本地词表参数的输入层，须与输出头的分片范围和
                隐藏宽度相容；本接口不执行重新分片或 dtype 转换。

        Returns:
            layer: 参数方法返回的共享输出头，普通方法返回 self。

        Side Effects:
            普通方法替换 self.weight 的引用，使两层指向同一个 Parameter；
            输出头已有偏置不参与共享。
        """
        return self.quant_method.tie_weights(self, embed_tokens)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """拒绝直接调用输出头，要求通过 LogitsProcessor 组织投影和词表汇集。

        Args:
            input_: 为 nn.Module 调用形式保留的输入，当前接口不会消费它。

        Raises:
            RuntimeError: 无条件抛出；调用方应使用处理器调用 quant_method.apply。
        """
        raise RuntimeError("LMHead's weights should be used in the sampler.")
