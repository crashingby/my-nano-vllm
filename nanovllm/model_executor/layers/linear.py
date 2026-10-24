"""提供复制、列切分与行切分的线性层及 checkpoint 分片加载接口。

权重布局为 [输出特征, 输入特征]。当前 rank/world_size 和 all_reduce 都使用
默认分布式进程组，不在这里创建独立 TP 组，也不负责启动多进程 Executor。
"""
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    """校验维度能被分片数整除后返回每份大小，不能整除时触发断言。

    Returns:
        partition_size: numerator 除以 denominator 得到的每份整数大小。
    """
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    """创建本 rank 的线性层参数，并把具体子类的加载器挂到参数上。

    input_size/output_size 表示本地尺寸，weight 为 [output_size, input_size]。
    tp_dim 指 checkpoint 沿哪个权重轴切分；基类不实现 forward 或 weight_loader。
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        """分配未初始化的本地权重和可选偏置，记录默认进程组的 rank 与大小。

        Args:
            input_size: 本地输入特征数，对应权重轴 1。
            output_size: 本地输出特征数，对应权重轴 0。
            bias: 是否创建本地输出维度的偏置。
            tp_dim: 子类加载 checkpoint 时使用的切分轴，复制层可以为 None。

        Side Effects:
            创建 nn.Parameter 并绑定 weight_loader 属性，实际数值由权重加载阶段填入。
            调用前必须已经初始化默认分布式进程组。
        """
        super().__init__()
        self.tp_dim = tp_dim  # checkpoint 的切分轴，不是输入 tensor 的 batch 轴。
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))  # 仅分配本地分片。
        # 权重加载器按参数属性分派，子类可以区分复制、分片与打包 checkpoint。
        self.weight.weight_loader = self.weight_loader
        if bias:
            # 偏置也挂载加载器，列切分与行切分对一维参数的处理不同。
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """声明线性层执行入口，子类必须实现，直接调用抛出 NotImplementedError。

        Returns:
            output: 子类线性变换的结果 tensor；基类入口尚未实现，直接调用会抛出异常。
        """
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    """每个 rank 保存完整权重并独立计算的线性层，不进行输出通信。"""
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        """按完整输入与输出尺寸创建复制参数，参数语义与 LinearBase 一致。"""
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """将完整 checkpoint tensor 原地复制到本地参数，不做 rank 切分。"""
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算 x @ weight.T + bias，返回最后一维为完整 output_size 的 tensor。

        Returns:
            output: x @ weight.T + bias 的完整输出，最后一维为 output_size。
        """
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    """沿输出特征切分权重，输入在各 rank 复制，输出保留本地特征分片。

    对 PyTorch 的 [out, in] 权重沿轴 0 切分。forward 不执行 all_gather，
    调用方必须知道返回结果只有全局输出维度的 1/tp_size。
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        """将全局 output_size 均分到默认进程组，input_size 保持完整。"""
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """从完整权重或偏置中选择本 rank 的输出区间，原地填充参数。

        Args:
            param: 当前 rank 的目标参数，轴 0 长度决定本地 shard_size。
            loaded_weight: 未切分的 checkpoint tensor。

        Side Effects:
            用 narrow 取得源切片，再 copy_ 到 param.data，不改写 checkpoint 内容。
        """
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)  # 当前 rank 持有的输出宽度。
        start_idx = self.tp_rank * shard_size  # 完整 checkpoint 中的本地起点。
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """使用完整输入计算本 rank 的输出分片，不拼接其他 rank 的输出。

        Returns:
            output: 本 rank 的列并行输出分片，不拼接其他 rank 的结果。
        """
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    """将多个列并行投影按输出轴打包，例如 MLP 的 gate/up 两个投影。

    各 checkpoint 分别切成本 rank 片段后放入本地打包权重，布局是
    [gate_local, up_local]，而不是把完整 gate/up 拼接后直接均分。
    """
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        """记录各原始投影的全局输出尺寸，并为它们的本地拼接结果分配参数。"""
        self.output_sizes = output_sizes  # 各子投影的全局宽度，加载时分别计算偏移。
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        """按原投影编号，将本 rank 的 checkpoint 分片写入打包参数。

        Args:
            param: 本地打包权重或偏置。
            loaded_weight: 一个原始投影的完整 checkpoint tensor。
            loaded_shard_id: output_sizes 中的投影下标，例如 gate 为 0、up 为 1。

        Side Effects:
            只覆盖该投影对应的本地区间，其他已加载投影保持原值。
        """
        param_data = param.data
        # 前面各投影的本地宽度之和，不能使用完整 checkpoint 中的全局偏移。
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    """打包 Q/K/V 投影，按各自 head 数切分后存为 [Q_local, K_local, V_local]。

    支持 Q 与 KV head 数不同的 GQA；当前要求两种 head 数都能被进程数整除，
    不实现 TP 数大于 KV head 数时的 KV 复制。
    """
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        """按本地 Q/KV head 数计算打包输出尺寸。

        Args:
            hidden_size: 完整输入 hidden state 的特征维度。
            head_size: 每个 Q/K/V head 的通道数。
            total_num_heads: 全局 Query head 数。
            total_num_kv_heads: 全局 KV head 数，为 None 时沿用 Query head 数。
            bias: 是否为打包输出创建同布局偏置。
        """
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)  # 本地 Query head 数。
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        """按 q/k/v 标识加载单个投影的本地 head 分片。

        Args:
            param: 本 rank 的打包权重或偏置。
            loaded_weight: 当前投影未切分的 checkpoint tensor。
            loaded_shard_id: 只能为 q、k、v，决定目标区间及其本地宽度。

        Side Effects:
            原地覆盖目标参数中的对应投影，保留其他两段。
        """
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        # 本地打包布局固定为 Q、K、V；GQA 下 K/V 区间通常比 Q 短。
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    """沿输入特征切分权重，输入已由调用方切分，输出跨 rank 求和。

    权重沿轴 1 切分。forward 不负责拆分完整输入；各 rank 只处理自己的输入分片，
    偏置仅由 rank 0 加一次，避免 all_reduce 后重复累加。
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        """将全局 input_size 均分到各 rank，所有 rank 保持完整 output_size。"""
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """权重按输入轴选择本 rank 区间；一维偏置复制完整内容，原地写入参数。"""
        param_data = param.data
        # 偏置没有输入特征轴，保留完整长度；forward 只在 rank 0 使用一次。
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算本地部分和，在多 rank 时通过默认进程组求和返回完整输出。

        Args:
            x: 最后一维为 input_size/tp_size 的本地输入分片。

        Returns:
            y: 最后一维为完整 output_size 的 tensor，所有参与 rank 获得归约结果。

        Side Effects:
            tp_size 大于 1 时发起 all_reduce，所有 rank 必须按一致顺序参与。
        """
        # 各 rank 贡献同形部分和，只让一个 rank 提供偏置。
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
