# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""按 vLLM 的流程封装 CUDA Graph 捕获、缓存、回放和 batch 统计。

Dispatcher 选择具体运行模式与 BatchDescriptor，Runner 准备 padding 后的
固定地址输入；本模块读取 ForwardContext，首次遇到 key 时捕获，之后回放。
这里不选择档位、不分配持久输入 buffer，也不将本轮数据复制到捕获地址。
当前保留 FULL/PIECEWISE 的 Wrapper 语义；分段编译执行链仍需单独接入。
"""

import dataclasses
import weakref
from collections import Counter
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any, ClassVar
from unittest.mock import patch

import torch

from nanovllm._C import load_extension
from nanovllm.compilation.monitor import validate_cudagraph_capturing_enabled
from nanovllm.config import CUDAGraphMode, NanovllmConfig
from nanovllm.forward_context import (
    BatchDescriptor,
    get_forward_context,
    is_forward_context_available,
)
from nanovllm.logger import init_logger
from nanovllm.platforms import current_platform
from nanovllm.utils.torch_utils import current_stream, weak_ref_tensors

logger = init_logger(__name__)


@dataclasses.dataclass(frozen=True)
class CUDAGraphStat:
    """记录一次 batch 的实际 token 数、执行容量和最终运行模式。

    例如 3 个 decode token 使用容量为 4 的 FULL Graph 时，字段为
    (3, 4, 1, "FULL")。本类不推导 padding，数值由调用方提供。
    frozen=True 使记录可哈希，Logging 可按四个字段完全相同的记录计数。
    """

    num_unpadded_tokens: int  # 本轮实际参与推理的 packed token 数。
    num_padded_tokens: int  # padding 后的执行 token 数；普通执行可与实际数相等。
    num_paddings: int  # 调用方记录的补齐量，通常为上述两个 token 数之差。
    runtime_mode: str  # 本轮具体模式的字符串，而非总配置的组合策略。


class CUDAGraphLogging:
    """暂存 batch 记录，按出现次数汇总成表格，并在日志输出后清空记录。

    只统计调用方 observe() 提交的数据，不自动监听 Wrapper，也不参与
    Graph 分派。generate_metric_table() 只生成文本，log() 才输出并 reset()。
    """

    COLUMN_HEADERS = [
        "Unpadded Tokens",
        "Padded Tokens",
        "Num Paddings",
        "Runtime Mode",
        "Count",
    ]

    def __init__(
        self, cg_mode: CUDAGraphMode, cg_capture_sizes: list[int] | None
    ) -> None:
        """保存日志显示用的配置快照，并创建空的统计窗口。

        Args:
            cg_mode: 配置策略，例如 FULL_DECODE_ONLY；每条记录另存实际模式。
            cg_capture_sizes: 配置中的捕获档位，仅用于显示，不在此创建 Graph。
        """
        self.reset()
        self.cg_mode = str(cg_mode)  # 配置策略与记录中的 runtime_mode 分开显示。
        self.cg_capture_sizes = str(cg_capture_sizes or [])  # 转为文本，不持有可变列表。

        # 表头展示创建 Logging 时的配置，不随之后的 batch 动态改变。
        self.settings_header = (
            "**CUDAGraph Config Settings:**\n\n"
            f"- Mode: {self.cg_mode}\n"
            f"- Capture sizes: {self.cg_capture_sizes}\n\n"
            "**CUDAGraph Stats:**\n\n"
        )

    def reset(self) -> None:
        """丢弃当前统计窗口中的记录，保留创建时的模式和档位配置。"""
        self.stats: list[CUDAGraphStat] = []

    def observe(self, cudagraph_stat: CUDAGraphStat) -> None:
        """追加一条 batch 记录；相同记录暂不合并，生成表格时再统计次数。"""
        self.stats.append(cudagraph_stat)

    def generate_metric_table(self) -> str:
        """返回含配置、batch 记录和 Count 的 Markdown 表格，不改变统计状态。

        四个字段都相同才合并为一行，按 Count 降序排列。比如 observe()
        两次提交 (3, 4, 1, "FULL")，表格该行的 Count 为 2。

        Returns:
            metric_table: 包含配置、批次记录与出现次数的 Markdown 表格字符串，不修改累计统计。
        """
        stats_counts = Counter(self.stats)  # 不可变记录作为 key，避免混合不同 padding 情况。

        # 先按频率排序并转成文本，后续据此计算列宽。
        rows: list[list[str]] = []
        for stat, count in sorted(
            stats_counts.items(), key=lambda item: item[1], reverse=True
        ):
            rows.append(
                [
                    str(stat.num_unpadded_tokens),
                    str(stat.num_padded_tokens),
                    str(stat.num_paddings),
                    stat.runtime_mode,
                    str(count),
                ]
            )

        # 每列宽度同时覆盖表头和所有数据，保持纯文本日志中的列对齐。
        col_widths: list[int] = []
        for i, header_text in enumerate(self.COLUMN_HEADERS):
            max_width = len(header_text)
            for row in rows:
                max_width = max(max_width, len(row[i]))
            col_widths.append(max_width)

        table_header_list = [
            h.ljust(w) for h, w in zip(self.COLUMN_HEADERS, col_widths)
        ]
        table_header = "| " + " | ".join(table_header_list) + " |\n"

        table_separator = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|\n"

        # 数据列与表头按相同列宽填充，Count 是汇总次数，不是请求数。
        data_rows: list[str] = []
        for row in rows:
            formatted_row = [
                str(val).ljust(width) for val, width in zip(row, col_widths)
            ]
            data_rows.append("| " + " | ".join(formatted_row) + " |")

        return (
            self.settings_header
            + table_header
            + table_separator
            + "\n".join(data_rows)
            + "\n"
        )

    def log(self, log_fn: Callable[..., Any] = logger.info) -> None:
        """有记录时输出完整表格，成功后开启新的统计窗口。

        Args:
            log_fn: 接受一个表格字符串的日志接口，默认使用模块级 info。

        Side Effects:
            调用日志接口并清空 stats；日志接口抛异常时保留原记录。
        """
        if not self.stats:
            return  # 空窗口不输出只有表头的日志。
        log_fn(self.generate_metric_table())
        self.reset()


@dataclasses.dataclass
class CUDAGraphEntry:
    """保存一个 BatchDescriptor 对应的捕获状态、输出别名和输入地址。

    新建时 cudagraph 为 None；捕获成功后保存 Graph 和非 owning 输出。
    Graph 存活期间 pool 保留捕获分配；output 不是独立的历史结果快照，
    下一次 replay 或共享 pool 的其他图可能覆盖其中的数据。
    """

    batch_descriptor: BatchDescriptor  # 与 Wrapper 字典中的 key 相同。
    cudagraph: torch.cuda.CUDAGraph | None = None  # None 表示该条目尚未成功捕获。
    output: Any | None = None  # Tensor 或单层 list/tuple，保存非 owning Tensor 别名。

    # 沿用上游，只记录位置参数中 Tensor 的地址；不遍历 kwargs、容器或 metadata。
    # 调试回放时比较地址列表，不能替代 shape、stride 和后端捕获条件的保证。
    input_addresses: list[int] | None = None


@dataclasses.dataclass
class CUDAGraphOptions:
    """控制捕获日志、捕获期间的清理调用和输出强引用的释放时机。

    weak_ref_output=False 时，捕获调用仍返回 owning 输出，供后续子图使用；
    entry.output 缓存始终转成非 owning 别名。PIECEWISE 的非末尾子图必须
    使用 False，最后一个子图才能提前释放 owning 输出。
    """

    debug_log_enable: bool = True  # 是否调用捕获 debug 日志，与输入地址检查开关独立。
    gc_disable: bool = False  # 捕获作用域内临时跳过 gc.collect 和 CUDA empty_cache。
    weak_ref_output: bool = True  # 是否在捕获内部就把局部输出转成非 owning 别名。


class CUDAGraphWrapper:
    """为一个 callable 添加按 batch key 捕获和回放 Graph 的能力。

    初始化时指定 FULL 或 PIECEWISE。调用时读取 ForwardContext：没有上下文、
    模式为 NONE 或与本 Wrapper 不匹配时，直接执行 runnable；模式匹配时，
    按 BatchDescriptor 查缓存，首次捕获，之后只发起 replay 并返回缓存输出。

    Runner 应在构造 Wrapper 前绑定设备，并在回放前原地更新固定地址的输入
    和后端 metadata。Wrapper 信任 Dispatcher 给出的 key，不选择档位、
    不 padding，也不把 args/kwargs 中的新 Tensor 自动复制到旧地址。
    共享 pool 按当前单卡顺序执行使用，不支持多个 Wrapper 并发回放。

    例如 key=(4, 4, True) 首次调用捕获四槽 decode 图。下一轮三个真实请求
    补到四槽后，只要 Runner 更新同一批 buffer，并继续传入该 key，就可
    回放已有图；只传入一个同形状的新 Tensor 并不能替换图记录的地址。
    """

    # 这里只弱引用 Wrapper 对象，避免全局注册表阻止其回收；与 Tensor 别名无关。
    _all_instances: ClassVar[weakref.WeakSet["CUDAGraphWrapper"]] = weakref.WeakSet()

    @classmethod
    def clear_all_graphs(cls) -> None:
        """清空所有仍存活 Wrapper 的 Graph 条目，不销毁 Wrapper 或其 runnable。

        调用方应先完成相关 GPU 工作，且不再使用旧输出别名；此接口不执行同步。
        """
        # 先取得强引用快照，避免遍历期间 WeakSet 因对象回收而改变。
        for instance in list(cls._all_instances):
            instance.clear_graphs()

    def __init__(
        self,
        runnable: Callable[..., Any],
        vllm_config: NanovllmConfig,
        runtime_mode: CUDAGraphMode,
        cudagraph_options: CUDAGraphOptions | None = None,
    ) -> None:
        """保存被包装函数、具体模式和选项，取得平台共享 pool 并创建空缓存。

        Args:
            runnable: 被捕获的模型或模型片段；输出支持 Tensor、单层 list/tuple。
            vllm_config: 当前 Worker 配置，保留上游参数名。
            runtime_mode: 本 Wrapper 对应的具体模式 FULL 或 PIECEWISE。
            cudagraph_options: 捕获选项；None 时创建独立的默认选项对象。

        Side Effects:
            首次取得共享 pool 时可能初始化 CUDA，但不捕获 Graph 或编译扩展。
            将当前实例弱注册到全局集合，供 clear_all_graphs() 统一清理。
        """
        self.runnable = runnable  # 属性透传和普通执行都访问这个原始 callable。
        self.vllm_config = vllm_config  # 保留配置引用，不复制 Worker 配置。
        self.runtime_mode = runtime_mode  # 与每次 ForwardContext 的具体模式比较。
        self.compilation_config = vllm_config.compilation_config

        self.first_run_finished = False  # 沿用上游保留字段；当前流程尚未读写此标记。
        self.is_debugging_mode = True  # 保留当前项目默认：每次回放检查位置参数的地址。
        self._runnable_str = str(runnable) if self.is_debugging_mode else None

        # NONE 路径直接执行模型，不需要创建 Graph Wrapper。
        assert self.runtime_mode != CUDAGraphMode.NONE
        # 与 vLLM 一致，不同档位与 Wrapper 共享 pool；前提是按顺序消费输出、回放。
        self.graph_pool = current_platform.get_global_graph_pool()
        if cudagraph_options is None:
            cudagraph_options = CUDAGraphOptions()
        self.cudagraph_options = cudagraph_options
        # 同一个 Wrapper 中，不同 BatchDescriptor 分别维护捕获状态与固定输出地址。
        self.concrete_cudagraph_entries: dict[BatchDescriptor, CUDAGraphEntry] = {}

        CUDAGraphWrapper._all_instances.add(self)

    def __getattr__(self, key: str) -> Any:
        """Wrapper 没有指定属性时，从原始 runnable 读取；不存在则抛异常。

        Returns:
            attr: 原始 runnable 上 key 对应的属性值或绑定方法；属性不存在时抛出 AttributeError。
        """
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        if self.is_debugging_mode:
            raise AttributeError(
                f"Attribute {key} not exists in the runnable of "
                f"cudagraph wrapper: {self._runnable_str}"
            )
        raise AttributeError

    def unwrap(self) -> Callable[..., Any]:
        """返回原始 callable，调用它会绕过本 Wrapper 的捕获与回放逻辑。

        Returns:
            runnable: 原始可调用对象；直接调用它会绕过本 Wrapper 的捕获与回放。
        """
        return self.runnable

    @property
    def cudagraph_wrapper(self) -> "CUDAGraphWrapper":
        """返回自身，提供统一的 Wrapper 查找入口，不创建新的包装对象。

        Returns:
            cudagraph_wrapper: 当前 Wrapper 自身，不创建新的包装对象。
        """
        return self

    def clear_graphs(self) -> None:
        """丢弃本实例的所有 Graph 条目，后续匹配调用会重新捕获。

        不修改 runnable、模式、pool 句柄或输入 buffer，也不等待 GPU 完成。
        旧输出别名不保活 Graph 分配，清理后不能继续把它们作为有效结果使用。
        """
        self.concrete_cudagraph_entries.clear()

    def __call__(self, *args: Any, **kwargs: Any) -> Any | None:
        """按当前上下文执行普通 forward、首次捕获或已有 Graph 的回放。

        Args:
            args: runnable 的位置参数；回放时 Tensor 必须使用捕获时的地址。
            kwargs: runnable 的关键字参数；只在普通执行或捕获时由 Python 读取。
                回放不会重新执行 runnable，动态值应通过固定地址 buffer 更新。

        Returns:
            output: 普通执行直接返回 runnable 的结果；捕获返回局部 output，是否 owning
                由 weak_ref_output 决定；回放返回 entry.output 中的非 owning 别名。
                返回结果不做 clone，不能作为跨回放长期保存的结果快照。

        Side Effects:
            首次匹配 key 时加载原生扩展并捕获 GPU 工作，成功后缓存 Graph。
            后续匹配 key 时在当前执行路径发起 replay，覆盖已有输出地址。
            捕获失败不标记条目完成，异常继续传播，之后允许重新尝试。

        Notes:
            本接口不会调用 Dispatcher，也不阻止首次 key 在在线阶段出现。
            启动期捕获与在线禁用捕获由 Runner 和 monitor 的开关共同控制。
        """
        if not is_forward_context_available():
            # 上下文之外没有可用的模式与 key，例如独立模型调用；直接执行原函数。
            return self.runnable(*args, **kwargs)

        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode

        if (
            cudagraph_runtime_mode == CUDAGraphMode.NONE
            or cudagraph_runtime_mode != self.runtime_mode
        ):
            # NONE 覆盖 profiling、预热和普通执行；模式不匹配时也直接透传。
            # 嵌套不同模式的 Wrapper 时，只有与本轮模式匹配的实例会捕获或回放。
            return self.runnable(*args, **kwargs)

        assert batch_descriptor is not None  # Graph 模式必须携带可哈希的缓存 key。
        if batch_descriptor not in self.concrete_cudagraph_entries:
            # 创建待捕获条目；只有 cudagraph 非 None 才表示捕获已完成。
            self.concrete_cudagraph_entries[batch_descriptor] = CUDAGraphEntry(
                batch_descriptor=batch_descriptor
            )

        entry = self.concrete_cudagraph_entries[batch_descriptor]
        if entry.cudagraph is None:
            if self.cudagraph_options.debug_log_enable:
                # 分段模式可能为每层创建 Wrapper，调用方可只为首段开启日志。
                # debug_log_enable 控制是否调用 logger，实际显示仍由日志级别决定。
                logger.debug(
                    "Capturing a cudagraph on (%s,%s)",
                    self.runtime_mode.name,
                    entry.batch_descriptor,
                )
            # 只在新增捕获前检查开关；在线禁止捕获时，已有条目的回放仍然允许。
            validate_cudagraph_capturing_enabled()

            # JIT 编译和动态库加载必须在捕获外完成；捕获内只调用已注册的算子。
            load_extension()

            input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            entry.input_addresses = input_addresses  # 只用于回放检查，不持有输入 Tensor。
            cudagraph = torch.cuda.CUDAGraph()
            with ExitStack() as stack:
                if self.cudagraph_options.gc_disable:
                    # 多段捕获时，调用方可在首段完成清理，让其余段跳过重复清理。
                    # 这里只临时替换两项清理入口，不调用 gc.disable，也不清空显存。
                    # ExitStack 在成功或异常退出时都恢复原函数。
                    stack.enter_context(
                        patch("gc.collect", lambda *args, **kwargs: None)
                    )
                    stack.enter_context(
                        patch(
                            "torch.cuda.empty_cache",  # CUDA-only，兼容当前 PyTorch 2.5.1。
                            lambda *args, **kwargs: None,
                        )
                    )

                with torch.cuda.graph(
                    cudagraph,
                    pool=self.graph_pool,
                    stream=current_stream(),
                ):
                    # 捕获期间创建的输出分配由 Graph pool 保留，回放使用相同地址。
                    output = self.runnable(*args, **kwargs)
                    if self.cudagraph_options.weak_ref_output:
                        # 对齐 vLLM：提前释放 owning 输出。PIECEWISE 仅最后一个
                        # 子图可启用，否则下一个子图仍需要该输出的分配保持存活。
                        output = weak_ref_tensors(output)

            # 缓存始终使用非 owning 别名；weak_ref_output 只控制捕获内的转换时机。
            entry.output = weak_ref_tensors(output)
            entry.cudagraph = cudagraph  # 最后发布已完成的 Graph，后续调用进入回放分支。

            # 必须返回局部 output，而非无条件返回缓存别名。weak_ref_output=False
            # 时它仍是 owning Tensor，后续子图捕获需要这份强引用维持分配。
            return output
        if self.is_debugging_mode:
            # 同一 key 只说明执行形状相同；图里的 kernel 仍访问捕获时的地址。
            # 对 kwargs、容器和上下文 metadata 的固定地址保证由 Runner 负责。
            new_input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            assert new_input_addresses == entry.input_addresses, (
                f"Input addresses for cudagraphs are different "
                f"during replay. Expected {entry.input_addresses}, "
                f"got {new_input_addresses}"
            )

        # 回放不会重新调用 runnable，也不会读取新的 Python 参数来替换图内地址。
        entry.cudagraph.replay()
        return entry.output  # 返回 entry.output 中本次 GPU 写入的缓存别名，不创建输出快照。
