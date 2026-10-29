"""线性层及权重计算策略：复制、列并行、打包列并行与行并行。

接口沿用 vLLM 的方法分派，基类直接继承 nn.Module。尺寸配置表示全局维度，
参数只分配本 rank 的分片；默认 forward 返回 (output, output_bias)。
本模块消费已经建立的 TP 组，不负责创建进程或通信组。
"""

from abc import abstractmethod
from collections.abc import Iterable
from typing import Any, TypeVar

import torch
from torch import nn
from typing_extensions import TypeIs

from nanovllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from nanovllm.logger import init_logger
from nanovllm.model_executor.layers.quantization import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from nanovllm.model_executor.layers.utils import dispatch_unquantized_gemm
from nanovllm.model_executor.model_loader.weight_utils import default_weight_loader
from nanovllm.model_executor.parameter import (
    BasevLLMParameter,
    ModelWeightParameter,
    RowvLLMParameter,
)
from nanovllm.model_executor.utils import set_weight_attrs
from nanovllm.utils.math_utils import divide

logger = init_logger(__name__)


# 沿用 vLLM 的按方法名称登记方式，只列入当前实际提供的实现。
WEIGHT_LOADER_V2_SUPPORTED: list[str] = ["UnquantizedLinearMethod"]
_LinearMethodT = TypeVar("_LinearMethodT", bound=QuantizeMethodBase)


def register_weight_loader_v2_supported_method(
    cls: type[_LinearMethodT],
) -> type[_LinearMethodT]:
    """登记能够创建专用参数的方法，供行/列并行层选择 weight_loader_v2。

    未登记的方法使用传统 weight_loader；登记的方法须创建实现 load_* 接口的
    BasevLLMParameter，不能依赖 v2 在执行时回退为普通参数加载。

    Returns:
        cls: 原方法类，便于作为类装饰器使用。

    Side Effects:
        将方法类名加入 WEIGHT_LOADER_V2_SUPPORTED，不构造方法或参数。
    """
    WEIGHT_LOADER_V2_SUPPORTED.append(cls.__name__)
    return cls


def adjust_scalar_to_fused_array(
    param_data: torch.Tensor,
    loaded_weight: torch.Tensor,
    shard_id: int | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把单投影标量定位到融合参数数组中的对应槽位，沿用 vLLM 的加载约定。

    Args:
        param_data: 每个逻辑投影保存一个元素的目标数组。
        loaded_weight: 标量或首维长度为 1 的 checkpoint tensor。
        shard_id: 投影下标，或 QKV 的 q/k/v 标识。

    Returns:
        (param_data, loaded_weight): 对应槽位的目标视图和移除多余首维的源 tensor。
            本函数只调整视图，实际复制由加载器执行。
    """
    qkv_idxs = {"q": 0, "k": 1, "v": 2}
    if isinstance(shard_id, str):
        shard_id = qkv_idxs[shard_id]
    elif not isinstance(shard_id, int):
        raise ValueError(f"Unknown Shard Id {shard_id}")
    if len(loaded_weight.shape) != 0:
        assert loaded_weight.shape[0] == 1
        loaded_weight = loaded_weight[0]
    return param_data[shard_id], loaded_weight


class LinearMethodBase(QuantizeMethodBase):
    """线性层的权重与计算协议，普通精度和量化方法共享 TP 层的外层逻辑。"""

    @abstractmethod
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
        """按 TP 层计算好的尺寸创建本地权重，绑定 checkpoint 加载元信息。

        Args:
            layer: 接收新参数的线性层。
            input_size_per_partition: 本 rank 的输入特征数。
            output_partition_sizes: 各逻辑投影的本地输出宽度，例如 Q/K/V 三段。
            input_size: 全局输入特征数。
            output_size: 全局输出特征数，QKV 复制布局包含副本宽度。
            params_dtype: 参数存储类型。
            extra_weight_attrs: weight_loader 等需绑定到参数的加载元信息。

        Side Effects:
            为 layer 注册参数，偏置仍由外层线性层单独创建。
        """
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """执行本地线性变换，TP 拼接与归约由调用方负责。

        Returns:
            output: 当前 rank 的投影结果，是否含偏置由 bias 决定。
        """
        raise NotImplementedError


class UnquantizedLinearMethod(LinearMethodBase):
    """保存普通浮点权重，构造时绑定 GEMM 实现，执行时不做量化转换。"""

    def __init__(self) -> None:
        """从当前 KernelConfig 选择后端；独立构造时使用 auto。

        配置只在构造期间读取，forward 中直接调用绑定的函数，不增加分派分支。
        """
        # 延迟导入避免总配置持有 QuantizationConfig 时形成模块初始化循环。
        from nanovllm.config import get_current_config_or_none

        config = get_current_config_or_none()
        linear_backend = (
            config.kernel_config.linear_backend if config is not None else "auto"
        )
        self._gemm_impl = dispatch_unquantized_gemm(linear_backend)

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
        """按已经划分的本地尺寸注册浮点 weight，不再执行第二次 TP 切分。

        参数语义沿用 LinearMethodBase；ModelWeightParameter 保存输入/输出轴和
        本层 TP 状态，具体分片复制由参数对象的 load_* 方法完成。

        Side Effects:
            创建未初始化、关闭梯度的权重，设置输入/输出轴及 weight_loader 属性。
        """
        weight_loader = extra_weight_attrs.pop("weight_loader")
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
            tp_rank=layer.tp_rank,
            tp_size=layer.tp_size,
        )
        set_weight_attrs(weight, extra_weight_attrs)  # 已由参数构造函数设置的属性不能覆盖。
        layer.register_parameter("weight", weight)

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """调用绑定的普通精度 GEMM，实现默认的 PyTorch 计算路径。

        Returns:
            output: x @ layer.weight.T + bias 的本地结果。
        """
        return self._gemm_impl(layer, x, layer.weight, bias)


class LinearBase(nn.Module):
    """线性层公共基类，统一尺寸、参数类型、偏置策略和 TP 分片信息。

    普通线性变换为 output = x @ weight.T + bias，PyTorch 权重按
    [输出特征, 输入特征] 存储。复制层、列并行层和行并行层共享这套配置，
    各子类再决定本 rank 保存哪部分参数，以及 forward 是否需要集合通信。
    接口沿用 vLLM 的权重方法分派，直接继承 nn.Module。

    基类构造时保存配置并选择 quant_method，不分配 weight/bias。子类计算本地尺寸，
    交给 quant_method.create_weights 注册权重，并自行创建偏置。普通精度方法使用
    ModelWeightParameter，将子类的加载器绑定到参数上；层计算融合偏移，参数对象
    切分并复制 checkpoint。Loader 随后调用方法的后处理钩子，完成后才能推理。

    input_size/output_size 保存传入的全局尺寸，本地参数形状由切分方式决定。
    例如列并行权重为 [output_size / tp_size, input_size]，行并行权重为
    [output_size, input_size / tp_size]。QKV 层存在 KV head 复制时，output_size
    包含这些副本占用的宽度，其含义由 QKVParallelLinear 进一步说明。

    子类必须实现 weight_loader 与 forward。forward 默认返回
    (output, output_bias)：正常加偏置时 output 已包含偏置，output_bias 为 None；
    skip_bias_add=True 时 output 不加偏置，output_bias 返回偏置供后续融合。
    return_bias=False 则仅返回 output；若同时跳过加偏置，调用方须自行处理偏置。
    未创建偏置参数时，output_bias 始终为 None。

    Attributes:
        input_size: 配置的全局输入特征数，不随本地权重切分而改变。
        output_size: 配置的全局输出特征数，QKV 复制布局可能包含额外副本。
        has_bias: 是否需要创建偏置参数，与执行时是否相加分别控制。
        skip_bias_add: 是否把加偏置留给后续算子，便于融合计算。
        params_dtype: 参数存储类型；未指定时采用构造时的默认浮点类型。
        prefix: 当前层在模型中的完整路径，不改变 nn.Module 注册的参数名称。
        return_bias: 是否返回二元组；True 时的第二项是待处理偏置或 None。
        disable_tp: 是否按完整权重执行；True 时固定 tp_rank=0、tp_size=1。
        tp_rank: 本层切分权重时使用的 TP 组内 rank，不是全局进程 rank。
        tp_size: 本层使用的分片数，决定每份权重及本地特征的尺寸。
        quant_config: 可选的量化配置，负责根据当前层和 prefix 选择方法。
        quant_method: 权重创建、加载后处理及本地计算的策略；未配置量化时使用
            UnquantizedLinearMethod。TP 通信仍由具体线性层的 forward 负责。
        weight: 普通精度方法创建的 ModelWeightParameter，布局为
            [本地输出特征数, 本地输入特征数]，保存加载入口、维度轴和 TP 状态。
        bias: 子类注册的本地偏置，长度等于本地输出特征数；无偏置时为 None。
            列并行偏置随输出切分，行并行偏置保留完整长度。

    Note:
        本类消费已经初始化的 TP 组，不负责创建进程或通信组。显式 tp_rank/tp_size
        仅覆盖分片信息，使用集合通信时须匹配实际 TP 组；disable_tp=True 时
        不查询 TP 组，子类也跳过分片和集合通信。基类不实现计算入口，
        具体层通过 quant_method.apply 执行本地计算。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        tp_rank: int | None = None,
        tp_size: int | None = None,
    ) -> None:
        """保存配置并选择方法；只有未显式提供的 TP 信息才从当前通信组读取。

        Args:
            input_size: 所有 TP rank 合计的输入特征数。
            output_size: 所有 TP rank 合计的输出特征数。
            bias: 是否创建偏置，具体形状由子类的切分方式决定。
            skip_bias_add: 不把偏置加进 output，留给调用方融合到后续算子。
            params_dtype: 参数存储类型，None 使用 torch.get_default_dtype()。
            quant_config: 选择权重与执行方法的配置；None 使用普通精度。
            prefix: 层在模型中的完整路径，仅保存元信息，不改写参数名称。
            return_bias: True 返回 (output, output_bias)，False 只返回 output。
                output_bias 仅在 skip_bias_add=True 时为本层偏置，否则为 None。
                同时设置 skip_bias_add=True、return_bias=False 会省略偏置。
            disable_tp: 强制使用 rank=0、size=1，不切分参数、不发起通信。
            tp_rank: 显式覆盖分片 rank；None 使用当前 TP 组内 rank。
            tp_size: 显式覆盖分片数；None 使用当前 TP 组大小。覆盖值不会
                创建新通信组，使用集合通信时必须与实际 TP 组保持一致。

        """
        super().__init__()
        self.input_size = input_size  # 全局维度，不能用本地权重 shape 反推。
        self.output_size = output_size
        self.has_bias = bias  # 是否创建参数，与 forward 是否加偏置相互独立。
        self.skip_bias_add = skip_bias_add
        self.params_dtype = (
            params_dtype if params_dtype is not None else torch.get_default_dtype()
        )
        self.prefix = prefix  # 与模型路径一致，便于定位层配置。
        self.quant_config = quant_config  # 配置负责为当前层选择方法，不负责 TP 通信。
        self.return_bias = return_bias
        self.disable_tp = disable_tp
        if disable_tp:
            self.tp_rank, self.tp_size = 0, 1  # 禁用后也不查询全局 TP 状态。
        else:
            self.tp_rank = (
                tp_rank if tp_rank is not None else get_tensor_model_parallel_rank()
            )
            self.tp_size = (
                tp_size
                if tp_size is not None
                else get_tensor_model_parallel_world_size()
            )

        if quant_config is None:
            self.quant_method: QuantizeMethodBase = UnquantizedLinearMethod()
        else:
            quant_method = quant_config.get_quant_method(self, prefix)
            if quant_method is None:
                raise ValueError(f"No quantization method for linear layer {prefix!r}")
            self.quant_method = quant_method

    def update_param_tp_status(self) -> None:
        """将参数对象的 TP 状态同步为当前层的配置。

        行/列并行层在参数创建完成后调用；权重后处理若替换了参数，Loader 也会调用。
        disable_tp 和显式 rank/size 覆盖以层配置为准，避免重新加载时选择错误分片。

        Side Effects:
            更新 BasevLLMParameter 的 tp_rank/tp_size，不改变参数内容或存储布局。
            普通偏置和其他 nn.Parameter 没有独立 TP 状态，不参与同步。
        """
        for param in self.parameters():
            if isinstance(param, BasevLLMParameter):
                param.tp_rank = self.tp_rank
                param.tp_size = self.tp_size


class ReplicatedLinear(LinearBase):
    """各 rank 保存完整权重并独立计算，不进行集合通信。"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ) -> None:
        """按完整尺寸分配参数，其余参数沿用 LinearBase 的偏置与返回约定。"""
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )
        self.output_partition_sizes = getattr(self, "output_sizes", [output_size])
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=input_size,
            output_partition_sizes=self.output_partition_sizes,
            input_size=input_size,
            output_size=output_size,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(output_size, dtype=self.params_dtype),
                requires_grad=False,
            )
            # 复制层加载完整偏置；output_dim 只标记其输出特征轴。
            set_weight_attrs(
                self.bias,
                {"output_dim": 0, "weight_loader": self.weight_loader},
            )
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        """校验完整 checkpoint 形状，并原地复制到参数，不做 rank 切分。"""
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param.size() == loaded_weight.size(), (
            f"Tried to load weights of size {loaded_weight.size()}"
            f"to a parameter of size {param.size()}"
        )
        param.data.copy_(loaded_weight)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, nn.Parameter | None]:
        """执行完整投影，按配置把偏置加进输出或留给调用方。

        Returns:
            output 或 (output, output_bias): output 最后一维为 output_size；
                output_bias 仅在 skip_bias_add=True 时为 self.bias，否则为 None。
        """
        bias = None if self.skip_bias_add else self.bias
        output = self.quant_method.apply(self, x, bias)
        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias
    def extra_repr(self) -> str:
        s = f"in_features={self.input_size}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        return s

class ColumnParallelLinear(LinearBase):
    """沿输出特征切分的线性层，输入复制到各 rank，输出可选择 all_gather。

    线性变换为 Y = X A + b，A 按输出维度拆为 [A_1, ..., A_p]，每个 rank
    计算 Y_i = X A_i。PyTorch 将权重保存为 A.T，因此实际沿 [out, in] 的轴 0
    切分。本地权重形状为 [output_size / tp_size, input_size]。

    构造顺序沿用 vLLM：确定 TP 状态和分片尺寸，再由 LinearBase 选择方法，
    随后创建参数、绑定加载器并同步参数 TP 状态。MergedColumn/QKV 预先设置
    output_sizes 时，各子投影分别均分，以保留本地打包边界。

    Args:
        input_size: 完整输入特征数，每个 rank 都接收相同宽度的输入。
        output_size: 全局输出特征数，必须能被本层 tp_size 整除。
        bias: 是否创建随输出特征切分的本地偏置。
        gather_output: 是否拼接各 rank 的输出，False 保留本地分片。
        skip_bias_add: 跳过本层偏置相加，留给后续算子处理。
        params_dtype: 权重和偏置的存储类型，None 使用默认浮点类型。
        quant_config: 权重方法配置，None 使用 UnquantizedLinearMethod。
        prefix: 当前层的完整路径，供配置按层选择方法。
        return_bias: 是否返回 (output, output_bias)，False 时只返回 output。
        disable_tp: 强制使用 rank=0、size=1，完整分配权重且不执行 all_gather。
        tp_rank: 覆盖加载时使用的 TP rank，None 使用当前 TP 组内 rank。
        tp_size: 覆盖分片数，None 使用当前 TP 组大小；覆盖值不创建通信组。

    Note:
        gather_output 只拼接 output，独立返回的 output_bias 始终为本地偏置。
        当前普通精度方法默认登记为 v2；FP8 和在线再量化专用分支尚未接入。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        tp_rank: int | None = None,
        tp_size: int | None = None,
    ) -> None:
        """先确定本地分片，再选择权重方法并创建本 rank 的参数。

        Side Effects:
            访问未显式提供的 TP 组信息，注册权重和可选偏置；不发起集合通信。
        """
        # 方法选择可能读取本地投影尺寸，必须在 LinearBase 构造之前准备好。
        if disable_tp:
            self.tp_rank, self.tp_size = 0, 1
        else:
            self.tp_rank = (
                tp_rank if tp_rank is not None else get_tensor_model_parallel_rank()
            )
            self.tp_size = (
                tp_size
                if tp_size is not None
                else get_tensor_model_parallel_world_size()
            )
        self.input_size_per_partition = input_size  # 列并行保留完整输入特征。
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        # MergedColumn/QKV 每段分别均分，不能只检查总输出宽度。
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(output_size, self.tp_size) for output_size in self.output_sizes
            ]

        super().__init__(
            input_size,
            output_size,
            bias,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        )
        self.gather_output = gather_output

        # 与上游一样在创建参数时选择加载器，而不是在 v2 内按参数类型回退。
        quant_method = self.quant_method
        weight_loader = (
            self.weight_loader_v2
            if quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED
            else self.weight_loader
        )
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=weight_loader,
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.output_size_per_partition, dtype=params_dtype),
                requires_grad=False,
            )
            # 列并行偏置随输出特征切分，加载器据 output_dim 选择分片。
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)
        self.update_param_tp_status()

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        """按输出轴切分并复制传统参数，没有 output_dim 时完整复制。

        param.is_sharded_weight=True 表示源 tensor 已经切成本地分片，避免再次切分。
        标量 checkpoint 先恢复成单元素向量，再严格检查目标形状，禁止广播装填。
        """
        output_dim = getattr(param, "output_dim", None)
        is_sharded_weight = getattr(param, "is_sharded_weight", False)

        param_data = param.data
        if output_dim is not None and not is_sharded_weight:
            shard_size = param_data.shape[output_dim]
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        """把列分片加载交给专用参数，参数自身决定输出轴及 TP 源区间。

        仅供登记支持 v2 的方法创建的参数使用；普通参数应在构造时选择传统加载器。
        """
        if len(loaded_weight.shape) == 0:
            assert loaded_weight.numel() == 1
            loaded_weight = loaded_weight.reshape(1)  # 兼容单元素参数以标量保存。
        param.load_column_parallel_weight(loaded_weight=loaded_weight)

    def forward(
        self,
        input_: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, nn.Parameter | None]:
        """计算本地输出，并按 gather_output 决定是否恢复完整输出。

        Returns:
            output 或 (output, output_bias): output 为本地分片或 rank 顺序的拼接；
                output_bias 仅在 skip_bias_add=True 时为本地 self.bias，否则为 None。

        Side Effects:
            gather_output=True 且 tp_size>1 时所有 TP rank 必须参与 all_gather。
        """
        bias = self.bias if not self.skip_bias_add else None

        output_parallel = self.quant_method.apply(self, input_, bias)

        if self.gather_output and self.tp_size > 1:
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def extra_repr(self) -> str:
        """展示全局输入宽度、本地输出宽度和 TP/拼接配置。

        Returns:
            s: nn.Module.__repr__ 显示的线性层配置摘要。
        """
        s = f"in_features={self.input_size}"
        s += f", output_features={self.output_size_per_partition}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        s += f", gather_output={self.gather_output}"
        return s


class MergedColumnParallelLinear(ColumnParallelLinear):
    """把多个列并行投影合为一次矩阵乘，各投影仍分别按输出轴做 TP 切分。

    例如 gate/up 的本地权重按 [gate_local, up_local] 排列，而不是将完整
    gate/up 拼接后直接均分。forward 继承 ColumnParallelLinear，区别在于
    output_sizes 描述各逻辑投影，以及 checkpoint 按这些边界分别装填。

    Args:
        input_size: 各投影共享的完整输入特征数。
        output_sizes: 各逻辑投影的全局输出宽度，每一段都必须能被 TP size 整除。
        bias: 是否创建同样打包布局的本地偏置。
        gather_output: 是否按 rank 顺序拼接输出；不会重排为全局 gate/up 顺序。
        skip_bias_add: 是否将偏置留给后续算子处理。
        params_dtype: 参数类型，None 沿用默认浮点类型。
        quant_config: 方法配置，None 使用普通精度；具体量化打包和 block scale
            参数尚未实现，因此不包含上游针对这些类型的转换分支。
        prefix: 当前融合层的完整模型路径。
        return_bias: 是否返回 (output, output_bias)，语义沿用父类。
        disable_tp: 完整保存各投影，不再按 rank 切分或执行输出拼接。
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ) -> None:
        """先保存逻辑投影和 TP 状态，再让父类创建融合参数与加载器。"""
        self.output_sizes = output_sizes
        self.tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1
        self.tp_rank = get_tensor_model_parallel_rank() if not disable_tp else 0

        # 总宽度能均分并不代表每个投影都能均分，必须逐段检查。
        assert all(output_size % self.tp_size == 0 for output_size in output_sizes)
        super().__init__(
            input_size=input_size,
            output_size=sum(output_sizes),
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

    def validate_shard_id(self, shard_id: Any) -> TypeIs[int | tuple[int, ...] | None]:
        """校验独立投影、完整融合或连续部分融合的编号。

        Returns:
            is_valid: 输入为合法 int、None 或连续 tuple 时返回 True。

        Raises:
            ValueError: 编号越界、类型错误或下标不连续。
        """
        if isinstance(shard_id, int):
            if shard_id < 0 or shard_id >= len(self.output_sizes):
                raise ValueError(
                    f"Shard id should be between 0 and {len(self.output_sizes) - 1}. "
                    f"Got shard id {shard_id}."
                )
            return True
        if shard_id is None:
            return True
        if isinstance(shard_id, tuple):
            for idx in shard_id:
                if not (0 <= idx < len(self.output_sizes)):
                    raise ValueError(
                        f"Shard id index {idx} should be between 0 and "
                        f"{len(self.output_sizes) - 1}. Got shard id {shard_id}."
                    )
            if len(shard_id) > 1 and any(
                b - a != 1 for a, b in zip(shard_id[:-1], shard_id[1:])
            ):
                raise ValueError(
                    "Shard id with multiple indices should be consecutive. "
                    f"Got shard id {shard_id}."
                )
            return True
        raise ValueError("This line should not be reached")

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        """加载传统参数，先解析融合格式，再按本 rank 选择每个投影的源分片。

        Args:
            param: 本地融合权重、偏置或无输出轴的复制参数。
            loaded_weight: 单个投影或按逻辑投影顺序融合的 checkpoint tensor。
            loaded_shard_id: int 指定单段，None 指全部，连续 tuple 指部分融合投影。

        Side Effects:
            原地覆盖所选投影区间。融合源按各投影的完整宽度拆开；单段加载时
            is_sharded_weight=True 跳过 TP 切分，needs_scalar_to_array=True
            按逻辑投影编号装填标量槽位。
        """
        self.validate_shard_id(loaded_shard_id)

        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)

        if loaded_shard_id is None or isinstance(loaded_shard_id, tuple):
            # 没有输出切分轴的参数按完整内容复制，不解释为融合矩阵。
            if output_dim is None:
                if needs_scalar_to_array:
                    param_data, loaded_weight = adjust_scalar_to_fused_array(
                        param_data, loaded_weight, 0,
                    )
                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return

            output_sizes = (
                self.output_sizes[loaded_shard_id[0] : loaded_shard_id[-1] + 1]
                if loaded_shard_id is not None
                else self.output_sizes
            )
            current_shard_offset = 0
            shard_offsets: list[tuple[int, int, int]] = []
            # 沿用上游的编号与偏移生成顺序，先按完整投影宽度拆分源 tensor。
            for i, output_size in enumerate(output_sizes):
                shard_offsets.append((i, current_shard_offset, output_size))
                current_shard_offset += output_size

            for shard_id, shard_offset, shard_size in shard_offsets:
                loaded_weight_shard = loaded_weight.narrow(output_dim, shard_offset, shard_size)
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        assert loaded_shard_id < len(self.output_sizes)
        if output_dim is not None:
            shard_offset = sum(self.output_sizes[:loaded_shard_id])
            shard_size = self.output_sizes[loaded_shard_id]
            shard_offset //= self.tp_size
            shard_size //= self.tp_size

            is_sharded_weight = getattr(param, "is_sharded_weight", False)
            param_data = param_data.narrow(output_dim, shard_offset, shard_size)
            start_idx = self.tp_rank * shard_size
            if not is_sharded_weight:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        elif needs_scalar_to_array:
            param_data, loaded_weight = adjust_scalar_to_fused_array(
                param_data, loaded_weight, loaded_shard_id,
            )
        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "MergedColumnParallelLinear, assume the weight is "
                    "the same for all partitions."
                )

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def _load_fused_module_from_checkpoint(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        output_sizes: list[int] | None = None,
    ) -> None:
        """拆开磁盘上已融合的权重，逐投影交给 weight_loader_v2 装填。

        Args:
            param: 带输出轴的专用参数。
            loaded_weight: 已按完整投影宽度融合保存的 checkpoint。
            output_sizes: 本次拆分采用的全局投影宽度，None 使用本层全部投影。

        Side Effects:
            按上游顺序从 0 编号，逐投影写入本地融合参数。
        """
        current_shard_offset = 0
        shard_offsets: list[tuple[int, int, int]] = []
        output_sizes = output_sizes or self.output_sizes
        for i, output_size in enumerate(output_sizes):
            shard_offsets.append((i, current_shard_offset, output_size))
            current_shard_offset += output_size

        for shard_id, shard_offset, shard_size in shard_offsets:
            loaded_weight_shard = loaded_weight.narrow(
                param.output_dim, shard_offset, shard_size,
            )
            self.weight_loader_v2(param, loaded_weight_shard, shard_id)

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        """加载专用参数，层解释融合编号，参数对象负责实际分片和复制。"""
        self.validate_shard_id(loaded_shard_id)
        if loaded_shard_id is None or isinstance(loaded_shard_id, tuple):
            if type(param) in (RowvLLMParameter, BasevLLMParameter):
                param.load_merged_column_weight(loaded_weight=loaded_weight)
                return
            output_sizes = (
                [self.output_sizes[idx] for idx in loaded_shard_id]
                if loaded_shard_id
                else None
            )
            self._load_fused_module_from_checkpoint(
                param,
                loaded_weight,
                output_sizes=output_sizes,
            )
            return

        assert loaded_shard_id < len(self.output_sizes)
        shard_offset = sum(self.output_sizes[:loaded_shard_id])
        shard_size = self.output_sizes[loaded_shard_id]
        shard_offset //= self.tp_size
        shard_size //= self.tp_size

        param.load_merged_column_weight(
            loaded_weight=loaded_weight,
            shard_id=loaded_shard_id,
            shard_offset=shard_offset,
            shard_size=shard_size,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Iterable[str]:
        """消费带 shard_id 元信息的权重流，支持本层及同样加载协议的子模块参数。

        Args:
            weights: (参数名, checkpoint tensor) 流；tensor.shard_id 指定逻辑投影，
                没有该属性时按完整融合参数加载。名称相对于当前层。

        Yields:
            name: 本次成功装填的目标参数名称，同一融合参数可随多段加载重复产出。
                本层 bias=None 时跳过名称为 bias 的条目。

        Side Effects:
            迭代期间逐条修改参数，创建生成器本身不会开始装填。
        """
        for name, loaded_weight in weights:
            shard_id = getattr(loaded_weight, "shard_id", None)
            self.validate_shard_id(shard_id)
            param: nn.Parameter
            if "." in name:
                submodule, _, attr = name.rpartition(".")
                param = getattr(self.get_submodule(submodule), attr, self)
            else:
                param = getattr(self, name, self)
            if param is None and name == "bias":
                continue
            param.weight_loader(param, loaded_weight, shard_id)
            logger.debug(
                "Loaded shard %s with shape %s into %s.%s",
                shard_id,
                loaded_weight.shape,
                self.prefix,
                name,
            )
            yield name


class QKVParallelLinear(ColumnParallelLinear):
    """融合 Attention 的 Q/K/V 投影，按完整 head 切分各 rank 的输出。

    本地权重沿输出轴排列为 [Q_local, K_local, V_local]，forward 复用
    ColumnParallelLinear，一次矩阵乘生成三段输出，不执行输出拼接。
    Q head 均分给各 rank；TP 大于 KV head 数时，相邻多个 rank 加载同一
    KV head 的副本。模型与 Attention 后端也须支持相同布局才能实际使用。

    Args:
        hidden_size: 各投影共享的完整输入 hidden state 宽度。
        head_size: 每个 Q/K head 的通道数。
        total_num_heads: 全局 Query head 数，必须能被 TP size 整除。
        total_num_kv_heads: 全局 KV head 数，None 沿用 Query head 数。
            KV 多于 TP 时均分；TP 多于 KV 时要求 TP size 能被 KV 数整除。
        bias: 是否创建同样按 Q/K/V 排列的本地偏置。
        skip_bias_add: 是否将偏置留给后续算子处理。
        params_dtype: 参数类型，None 沿用默认浮点类型。
        quant_config: 权重方法配置；尚未接入量化打包与 block scale 参数类型。
        prefix: 当前融合层的完整模型路径。
        return_bias: 是否返回 (output, output_bias)，语义沿用父类。
        disable_tp: 是否完整保存投影，不再按 rank 切分。
        v_head_size: 每个 V head 的通道数，None 沿用 head_size。
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        v_head_size: int | None = None,
    ) -> None:
        """先确定本地 head 数与 KV 副本数，再让列并行父类创建参数。

        Side Effects:
            未禁用 TP 时查询当前 TP 组；父类注册本地权重与可选偏置，
            并按权重方法绑定传统加载器或 v2 加载器，不发起集合通信。
        """
        self.hidden_size = hidden_size  # 各 rank 保留完整输入特征。
        self.head_size = head_size  # Q/K 的切分粒度是完整 head。
        self.v_head_size = v_head_size if v_head_size is not None else head_size
        self.total_num_heads = total_num_heads  # checkpoint 中未分片的 Query head 数。
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads

        tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            # 每个 rank 至少保存一个 KV head，多个相邻 rank 可加载同一个源 head。
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        input_size = self.hidden_size
        # 父类会逐段除以 tp_size，故这里按包含 KV 副本的总宽度描述三段。
        # checkpoint 的原始宽度仍由 total_num_heads/total_num_kv_heads 决定。
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,  # q_proj。
            self.num_kv_heads * self.head_size * tp_size,  # k_proj。
            self.num_kv_heads * self.v_head_size * tp_size,
        ]
        output_size = sum(self.output_sizes)

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            gather_output=False,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

    def validate_shard_id(self, shard_id: Any) -> TypeIs[str | None]:
        """校验独立 Q/K/V 投影标识，None 表示 checkpoint 已融合三段。

        Returns:
            is_valid: shard_id 为 q、k、v 或 None 时返回 True。

        Raises:
            ValueError: shard_id 不是支持的投影标识。
        """
        if shard_id in {"q", "k", "v"} or shard_id is None:
            return True
        raise ValueError(
            "Shard id for QKVParallelLinear should be 'q', 'k', or 'v', "
            f"got shard id {shard_id}."
        )

    def _get_shard_offset_mapping(self, loaded_shard_id: str) -> int | None:
        """查找本地融合输出中的起始位置，total 表示三段合计的结束位置。

        Returns:
            shard_offset: 以输出元素为单位的本地偏移，未知标识返回 None。
        """
        shard_offset_mapping = {
            "q": 0,
            "k": self.num_heads * self.head_size,
            "v": (self.num_heads + self.num_kv_heads) * self.head_size,
            "total": (self.num_heads + self.num_kv_heads) * self.head_size
            + self.num_kv_heads * self.v_head_size,
        }
        return shard_offset_mapping.get(loaded_shard_id)

    def _get_shard_size_mapping(self, loaded_shard_id: str) -> int | None:
        """查找一个 Q/K/V 投影在本 rank 上的输出宽度。

        Returns:
            shard_size: 本地 head 数乘对应 head 宽度，未知标识返回 None。
        """
        shard_size_mapping = {
            "q": self.num_heads * self.head_size,
            "k": self.num_kv_heads * self.head_size,
            "v": self.num_kv_heads * self.v_head_size,
        }
        return shard_size_mapping.get(loaded_shard_id)

    def _load_fused_module_from_checkpoint(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        """拆开磁盘上已融合的全局 Q/K/V 权重，逐投影交给 v2 加载器。

        Args:
            param: 带 output_dim 的专用参数，目标存储采用本地融合布局。
            loaded_weight: 按全局 [Q, K, V] 顺序保存的 checkpoint tensor。

        Side Effects:
            按三段的全局边界取源视图，分别写入本地参数；源边界不包含 KV 副本。
        """
        shard_offsets = [
            # (投影标识, checkpoint 输出轴偏移, checkpoint 输出宽度)。
            ("q", 0, self.total_num_heads * self.head_size),
            (
                "k",
                self.total_num_heads * self.head_size,
                self.total_num_kv_heads * self.head_size,
            ),
            (
                "v",
                (self.total_num_heads + self.total_num_kv_heads) * self.head_size,
                self.total_num_kv_heads * self.v_head_size,
            ),
        ]

        for shard_id, shard_offset, shard_size in shard_offsets:
            loaded_weight_shard = loaded_weight.narrow(
                param.output_dim, shard_offset, shard_size,
            )
            self.weight_loader_v2(param, loaded_weight_shard, shard_id)

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ) -> None:
        """解释 Q/K/V 融合布局，把 TP 源分片选择和复制交给专用参数。

        Args:
            param: 登记支持 v2 的权重方法创建的专用参数，不回退为传统加载。
            loaded_weight: 单个投影或按全局 [Q, K, V] 保存的融合 checkpoint。
            loaded_shard_id: q/k/v 指定单段，None 表示完整融合内容。

        Side Effects:
            原地装填参数；num_heads 沿用参数接口命名，传入的实际是 KV 副本数。
        """
        self.validate_shard_id(loaded_shard_id)
        if loaded_shard_id is None:
            # 这两种参数没有列切分语义，完整融合内容直接复制。
            if type(param) in (RowvLLMParameter, BasevLLMParameter):
                param.load_qkv_weight(loaded_weight=loaded_weight)
                return
            self._load_fused_module_from_checkpoint(param, loaded_weight)
            return

        assert loaded_shard_id in ["q", "k", "v"]

        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)
        assert shard_offset is not None and shard_size is not None

        param.load_qkv_weight(
            loaded_weight=loaded_weight,
            num_heads=self.num_kv_head_replicas,
            shard_id=loaded_shard_id,
            shard_offset=shard_offset,
            shard_size=shard_size,
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ) -> None:
        """按 output_dim 加载传统参数，分别处理融合源、单段源和复制参数。

        Args:
            param: 本地融合权重、偏置或不随输出轴切分的复制参数。
            loaded_weight: 单个投影或全局 [Q, K, V] 融合 checkpoint。
            loaded_shard_id: q/k/v 指定投影，None 表示源权重已融合。

        Side Effects:
            原地更新目标参数；单段源设置 is_sharded_weight=True 时跳过 TP
            切分，needs_scalar_to_array=True 时按 q/k/v 装填对应的标量槽位。
        """
        self.validate_shard_id(loaded_shard_id)

        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)

        if loaded_shard_id is None:
            # 无输出轴的参数不解释为融合矩阵，直接复制或定位首个标量槽位。
            if output_dim is None:
                if needs_scalar_to_array:
                    param_data, loaded_weight = adjust_scalar_to_fused_array(
                        param_data, loaded_weight, 0,
                    )

                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return
            shard_offsets = [
                # 源偏移使用 checkpoint 的全局 head 数，不包含本地 KV 副本。
                ("q", 0, self.total_num_heads * self.head_size),
                (
                    "k",
                    self.total_num_heads * self.head_size,
                    self.total_num_kv_heads * self.head_size,
                ),
                (
                    "v",
                    (self.total_num_heads + self.total_num_kv_heads) * self.head_size,
                    self.total_num_kv_heads * self.v_head_size,
                ),
            ]
            for shard_id, shard_offset, shard_size in shard_offsets:
                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size,
                )
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        assert loaded_shard_id in ["q", "k", "v"]

        if output_dim is not None:
            # 目标偏移采用本地 head 数，分别定位本 rank 的 Q、K、V 区间。
            if loaded_shard_id == "q":
                shard_offset = 0
                shard_size = self.num_heads * self.head_size
            elif loaded_shard_id == "k":
                shard_offset = self.num_heads * self.head_size
                shard_size = self.num_kv_heads * self.head_size
            elif loaded_shard_id == "v":
                shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size
                shard_size = self.num_kv_heads * self.v_head_size

            is_sharded_weight = getattr(param, "is_sharded_weight", False)
            param_data = param_data.narrow(output_dim, shard_offset, shard_size)
            if loaded_shard_id == "q":
                shard_rank = self.tp_rank
            else:
                shard_rank = self.tp_rank // self.num_kv_head_replicas
            start_idx = shard_rank * shard_size

            if not is_sharded_weight:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        elif needs_scalar_to_array:
            param_data, loaded_weight = adjust_scalar_to_fused_array(
                param_data, loaded_weight, loaded_shard_id,
            )
        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "QKVParallelLinear, assume the weight is the same "
                    "for all partitions."
                )

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Iterable[str]:
        """消费带 q/k/v 标识的权重流，加载本层或使用相同协议的子模块参数。

        Args:
            weights: (参数名, checkpoint tensor) 流；tensor.shard_id 指定投影，
                没有该属性时按完整融合内容加载，参数名相对于当前层。

        Yields:
            name: 成功装填的参数名称，同一参数可随不同投影重复产出；
                本层未创建 bias 时跳过名称为 bias 的条目。

        Side Effects:
            迭代期间逐条修改参数，创建生成器本身不开始加载。
        """
        for name, loaded_weight in weights:
            shard_id = getattr(loaded_weight, "shard_id", None)
            self.validate_shard_id(shard_id)
            param: nn.Parameter
            if "." in name:
                submodule, _, attr = name.rpartition(".")
                param = getattr(self.get_submodule(submodule), attr, self)
            else:
                param = getattr(self, name, self)
            if param is None and name == "bias":
                continue
            param.weight_loader(param, loaded_weight, shard_id)
            logger.debug(
                "Loaded shard %s with shape %s into %s.%s",
                shard_id,
                loaded_weight.shape,
                self.prefix,
                name,
            )
            yield name


class RowParallelLinear(LinearBase):
    """沿输入特征切分权重，各 rank 计算部分和，再按配置归约完整输出。

    数学形式 Y = X A + b 中，A 沿行切分，X 沿最后一维切分，满足
    Y = sum(X_i A_i) + b。PyTorch 保存的是 weight = A.T，因此本地权重
    形状为 [output_size, input_size / tp_size]，实际切分轴是 input_dim。
    各 rank 的输出形状完整，但归约前只包含本地输入分片的贡献。

    Args:
        input_size: 全部 TP rank 合计的输入特征数，必须能被 TP size 整除。
        output_size: 每个 rank 都保留的完整输出特征数。
        bias: 是否创建完整偏置，各 rank 加载相同内容，不随输入切分。
        input_is_parallel: True 表示输入已切分；False 从完整输入选择本 rank 分片。
        skip_bias_add: 是否跳过偏置相加，将完整偏置单独返回以供后续融合。
        params_dtype: 参数类型，None 沿用默认浮点类型。
        reduce_results: 是否对各 rank 的输出求和；False 返回本地部分和，
            此时须关闭 bias 或设置 skip_bias_add=True，由调用方处理归约与偏置。
        quant_config: 权重方法配置，当前尚未接入在线量化与再量化方法。
        prefix: 当前层的完整模型路径，例如 model.layers.0.mlp.down_proj。
        return_bias: 是否返回 (output, output_bias)，语义沿用父类。
        disable_tp: 是否保留完整权重，并跳过 TP 切分和集合通信。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        reduce_results: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ) -> None:
        """先确定 TP 分片尺寸，再选择权重方法并注册本地权重与完整偏置。

        Side Effects:
            未禁用 TP 时查询当前 TP 组；创建参数并绑定与权重方法对应的加载器，
            不加载 checkpoint，也不发起集合通信。
        """
        # 权重方法选择可能读取本地尺寸，因此必须在父类构造之前准备好。
        self.tp_rank = get_tensor_model_parallel_rank() if not disable_tp else 0
        self.tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size  # 每个 rank 都计算所有输出位置。
        self.output_partition_sizes = [output_size]

        super().__init__(
            input_size,
            output_size,
            bias,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        # 与列并行使用相同的登记机制，v2 只接收方法创建的专用参数。
        quant_method = self.quant_method
        weight_loader = (
            self.weight_loader_v2
            if quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED
            else self.weight_loader
        )

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=weight_loader,
        )
        # 未归约的结果仍是部分和，不能让下游把各 rank 的偏置再次累加。
        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError(
                "When not reduce the results, adding bias to the "
                "results can lead to incorrect results"
            )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.output_size, dtype=params_dtype),
                requires_grad=False,
            )
            # 偏置没有输入切分轴，各 rank 都加载完整内容，仅 rank 0 负责相加。
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)
        self.update_param_tp_status()

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        """沿 input_dim 选择本 rank 的源分片，再严格校验形状并原地复制。

        没有 input_dim 的偏置等参数完整复制；is_sharded_weight=True 表示
        checkpoint 已保存本地分片，不再重复切分。标量源先恢复为单元素向量。
        """
        input_dim = getattr(param, "input_dim", None)
        is_sharded_weight = getattr(param, "is_sharded_weight", False)

        param_data = param.data
        if input_dim is not None and not is_sharded_weight:
            shard_size = param_data.shape[input_dim]
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(input_dim, start_idx, shard_size)

        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        """把输入轴分片与原地复制交给专用参数，不回退为传统加载器。

        仅供登记支持 v2 的权重方法使用，普通参数应在构造时绑定传统加载器。
        """
        if len(loaded_weight.shape) == 0:
            assert loaded_weight.numel() == 1
            loaded_weight = loaded_weight.reshape(1)
        param.load_row_parallel_weight(loaded_weight=loaded_weight)

    def forward(
        self,
        input_: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, nn.Parameter | None]:
        """使用本地输入计算部分和，按配置归约并返回输出与待处理偏置。

        Args:
            input_: input_is_parallel=True 时为 [..., input_size / tp_size]；
                False 时为 [..., input_size]，在本进程中取出对应分片。

        Returns:
            output 或 (output, output_bias): output 最后一维为完整 output_size，
                reduce_results=False 时数值仍是部分和；output_bias 仅在
                skip_bias_add=True 时为完整 self.bias，否则为 None。

        Side Effects:
            reduce_results=True 且 tp_size>1 时所有 TP rank 必须参与 all_reduce。
        """
        if self.input_is_parallel:
            input_parallel = input_  # 上游列并行/Attention 已经生成本地输入。
        else:
            split_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size,
            )
            input_parallel = split_input[self.tp_rank].contiguous()

        # 偏置只由 rank 0 加一次，避免归约后被累加 TP size 次。
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self, input_parallel, bias_)

        if self.reduce_results and self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel

        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def extra_repr(self) -> str:
        """展示本地输入宽度、完整输出宽度与 TP 归约配置。

        Returns:
            s: nn.Module.__repr__ 显示的行并行层配置摘要。
        """
        s = f"in_features={self.input_size_per_partition}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        s += f", reduce_results={self.reduce_results}"
        return s
