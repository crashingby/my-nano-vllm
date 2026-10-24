"""在模型 forward 期间共享 Attention 输入和 CUDA Graph 运行信息。

沿用 vLLM 的创建、覆盖与恢复流程，保存静态层注册表、Attention metadata、
KV slot mapping、本轮 Graph 模式和 batch 描述符。Runner 准备输入并选择
运行模式，模型层或 Graph Wrapper 在 with 作用域内读取；本模块只传递引用，
不执行 padding、Graph 分派、捕获或回放，也不推进请求状态。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

# 默认字段值在类定义时求值，因此 Graph 模式枚举必须在运行时导入。
from nanovllm.config.compilation import CUDAGraphMode

if TYPE_CHECKING:
    # 这些对象仅用于类型说明，避免运行时导入配置和 Attention 后端产生循环依赖。
    import torch

    from nanovllm.attention.backend import AttentionMetadata
    from nanovllm.config import NanovllmConfig


@dataclass(frozen=True)
class BatchDescriptor:
    """描述 Graph 对应的 batch 形状，供 Dispatcher 和 Wrapper 作为缓存 key。

    字段沿用 vLLM，当前省略未支持的 LoRA 维度。Graph 路径的描述符保存
    padding 后的执行形状，不记录请求 ID、token 值或历史上下文长度；不同请求只要满足
    同一捕获计划，就可以复用同一个 key。frozen=True 使对象可哈希，且无法
    在进入缓存后修改字段；构造描述符本身不会填充任何输入或 metadata。

    例如三个单 token decode 请求补齐到四个请求槽位后，Runner 可以传入
    BatchDescriptor(num_tokens=4, num_reqs=4, uniform=True)。

    Attributes:
        num_tokens: 执行形状中的 packed token 数；使用 Graph 时包含 padding。
        num_reqs: 执行形状中的请求槽位数；完整 Graph 的 key 通常需要此维度。
            PIECEWISE 的 key 可设为 None，表示不按请求数区分，而非零个请求。
        uniform: 是否所有请求本轮的 query 长度相同；不要求历史上下文长度相同。
            该标记本身不表示 decode，仍需 Dispatcher 结合 query 长度判断。
    """

    num_tokens: int  # Graph 路径含 padding；普通执行的描述符可保留有效 token 数。
    num_reqs: int | None = None  # None 放宽请求维度；具体 key 由 Dispatcher 决定。
    uniform: bool = False  # 描述本轮 query 长度关系，与历史序列长度无关。


@dataclass
class ForwardContext:
    """保存一次模型 forward 共享的模型层引用、Attention 输入和 Graph 信息。

    GPUModelRunner 在调用模型前创建此对象，Attention 层通过
    ``get_forward_context()`` 按自己的完整层名查询数据。当前一个 Worker
    同步执行一次 forward，使用进程内上下文；不支持并发线程交错覆盖。

    Attributes:
        no_compile_layers: 完整层名到模型层实例的静态注册表，沿用 vLLM 命名。
            引用 ``compilation_config.static_forward_context``，不复制层实例。
        attn_metadata: 完整层名到本轮后端 metadata 的映射；profiling 没有
            真实请求元数据时可以为 None。它不是 Scheduler 的请求状态。
        slot_mapping: 完整层名到本轮 packed token 的 KV 写入 slot tensor 的
            映射；无 cache 写入时使用空字典，padding slot 由调用方设为 -1。
        cudagraph_runtime_mode: 本轮具体运行模式，只允许 NONE、PIECEWISE、FULL。
            FULL_DECODE_ONLY 等组合模式属于配置策略，需先由 Dispatcher 解析。
        batch_descriptor: 本轮执行形状或 Graph 缓存 key；原样引用调用方对象。
            普通执行可为 None，Graph 路径由调用方提供或由 set_forward_context()
            根据 num_tokens 创建仅包含 token 数的缺省描述符。

    Side Effects:
        构造对象本身不安装全局上下文。成员保存输入引用，后续 buffer 更新会
        改变这些引用看到的内容，不能将此对象视为独立的历史执行快照。
    """

    no_compile_layers: dict[str, Any]  # 静态模型层按稳定名称定位，与请求槽位无关。
    attn_metadata: dict[str, AttentionMetadata] | None  # 本次 forward 的按层元数据。
    slot_mapping: dict[str, torch.Tensor]  # 本次 forward 的按层 KV 写入坐标。
    # 缺省为普通执行；枚举有效性与后端实际支持范围是两项独立的检查。
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE
    batch_descriptor: BatchDescriptor | None = None  # Graph Wrapper 据此查找捕获条目。

    def __post_init__(self) -> None:
        """沿用 vLLM 的不变量检查，禁止将组合配置策略当作本轮运行模式。

        Raises:
            AssertionError: 传入 FULL_DECODE_ONLY 或 FULL_AND_PIECEWISE，表示
                调用方尚未根据本轮 batch 选择具体运行模式。
        """
        # 只检查运行模式枚举；不替代 Dispatcher 的后端能力与捕获 key 检查。
        assert self.cudagraph_runtime_mode.is_valid_runtime_mode(), (
            f"Invalid cudagraph runtime mode: {self.cudagraph_runtime_mode}"
        )


# 当前 Worker 同步执行模型；只有进入 forward 上下文后，此变量才应当非空。
_forward_context: ForwardContext | None = None


def get_forward_context() -> ForwardContext:
    """取得当前已经安装的模型 forward 上下文。

    Returns:
        forward_context: 当前活动的 ForwardContext 引用，不创建副本。

    Raises:
        AssertionError: 在 ``set_forward_context()`` 或
            ``override_forward_context()`` 安装上下文之前调用时抛出。
    """
    # 模型层在上下文之外读取运行时数据属于内部调用顺序错误，不能返回空对象掩盖。
    assert _forward_context is not None, (
        "Forward context is not set. "
        "Please use set_forward_context() before running the model."
    )
    return _forward_context


def is_forward_context_available() -> bool:
    """判断当前是否存在活动的模型 forward 上下文，不改变任何状态。

    Returns:
        is_available: 当前存在活动 ForwardContext 时为 True，否则为 False。
    """
    return _forward_context is not None


def create_forward_context(
    attn_metadata: dict[str, AttentionMetadata] | None,
    vllm_config: NanovllmConfig,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    slot_mapping: dict[str, torch.Tensor] | None = None,
) -> ForwardContext:
    """创建一次 forward 的数据容器，但不改变当前活动上下文。

    Args:
        attn_metadata: 按完整 Attention 层名组织的本轮 metadata；无真实
            Attention metadata 的 profiling 路径可传 None。
        vllm_config: 当前 Worker 的总配置，提供已经注册的静态模型层。
        cudagraph_runtime_mode: 本轮具体运行模式；直接传给 ForwardContext 校验。
        batch_descriptor: 已确定的执行形状，保留其对象引用；本函数不推导描述符。
        slot_mapping: 按层名组织的 KV 写入坐标；None 表示当前没有写入映射。

    Returns:
        forward_context: 引用原始注册表、metadata 和 slot mapping 的 ForwardContext。
            Graph 模式与描述符原样传递；slot_mapping 为 None 时独立创建空字典。

    Raises:
        AssertionError: 将组合配置模式当作具体运行模式传入时抛出。

    Side Effects:
        不修改输入字典，不复制 tensor，不安装全局上下文。
    """
    # 只在缺省时创建空映射，显式传入的空字典也保留其原始对象身份。
    slot_mapping = {} if slot_mapping is None else slot_mapping
    return ForwardContext(
        # 保留模型层实例的共享引用，不能通过 deepcopy 创建第二套模型状态。
        no_compile_layers=vllm_config.compilation_config.static_forward_context,
        attn_metadata=attn_metadata,  # 保留本轮后端元数据，不复制其中的 tensor。
        cudagraph_runtime_mode=cudagraph_runtime_mode,  # 不在创建上下文时选择执行策略。
        batch_descriptor=batch_descriptor,  # 与 Wrapper 查找条目时使用的 key 一致。
        slot_mapping=slot_mapping,  # Attention 的 KV 写入步骤读取这份共享映射。
    )


@contextmanager
def override_forward_context(
    forward_context: ForwardContext | None,
):
    """临时覆盖活动上下文，并在正常退出或异常退出时恢复外层上下文。

    Args:
        forward_context: 要安装的已有上下文；传 None 可暂时隐藏外层上下文。

    Yields:
        空值；上下文内通过 ``get_forward_context()`` 访问当前对象。

    Side Effects:
        在 with 作用域内替换进程内的活动上下文。嵌套调用退出后恢复父上下文，
        异常继续向上传播，不被此接口吞掉。
    """
    global _forward_context

    # 必须保存完整外层对象；无条件清空会破坏嵌套 forward 的访问环境。
    previous_context = _forward_context
    _forward_context = forward_context  # 当前作用域内的模型层读取新安装的数据。
    try:
        yield
    finally:
        # 模型或后端抛异常时也恢复，避免下一轮误读失败 forward 的 metadata。
        _forward_context = previous_context


@contextmanager
def set_forward_context(
    attn_metadata: dict[str, AttentionMetadata] | None,
    vllm_config: NanovllmConfig,
    num_tokens: int | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    slot_mapping: dict[str, torch.Tensor] | None = None,
):
    """创建并安装本轮 forward 上下文，退出后恢复原来的活动上下文。

    GPUModelRunner 应使用此接口包裹模型调用。这里只管理 Python 运行时引用，
    不替代模型构建期的 ``set_current_config()``，也不写入 KV cache。

    Args:
        attn_metadata: Runner 构造的按层名 metadata；profiling 时可传 None。
        vllm_config: 当前 Worker 的总配置，提供静态模型层注册表。
        num_tokens: 本轮执行的 packed token 数；Graph 路径应传 padding 后的数值。
            Graph 模式非 NONE 且缺少描述符时，用它创建缺省 BatchDescriptor。
            普通执行下不会据此创建描述符；当前不用于 DP 协调或 batch 计时。
        cudagraph_runtime_mode: Dispatcher 选出的具体模式，不是总配置中的组合策略。
        batch_descriptor: 优先使用调用方传入的描述符；省略时可根据 num_tokens
            创建仅有 token 数的描述符，不补算 num_reqs 或 uniform。
        slot_mapping: 本轮按层名组织的物理 KV slot；无写入路径可不提供。

    Yields:
        空值；Attention 层通过 ``get_forward_context()`` 读取运行时数据。

    Side Effects:
        仅在 with 作用域内暴露本轮上下文；退出时恢复外层状态。创建过程中
        不复制模型或输入 tensor，也不增加 CUDA 同步点。

    Raises:
        AssertionError: 运行模式尚未从组合配置策略解析为具体模式。
    """
    # 沿用上游缺省描述符的便利路径；实际 Graph key 应优先由 Dispatcher 提供。
    if cudagraph_runtime_mode != CUDAGraphMode.NONE and num_tokens is not None:
        # 显式描述符可能含 padding 后的请求维度，不能被这里只含 token 数的对象覆盖。
        batch_descriptor = batch_descriptor or BatchDescriptor(num_tokens=num_tokens)
    # 每轮创建独立上下文对象；metadata 与 tensor views 仍引用调用方准备的数据。
    forward_context = create_forward_context(
        attn_metadata=attn_metadata,
        vllm_config=vllm_config,
        slot_mapping=slot_mapping,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        batch_descriptor=batch_descriptor,
    )
    # 覆盖与异常恢复集中在同一个 helper 中，确保两种安装入口具有相同生命周期。
    with override_forward_context(forward_context):
        yield
