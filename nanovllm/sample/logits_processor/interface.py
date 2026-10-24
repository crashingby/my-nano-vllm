"""定义 logits processor 与持久输入批之间的通信协议。

`InputBatch` 会在 GPU 上长期保留一个稠密的请求槽位数组。请求完成、加入，
或为了回收空槽位而移动时，logits processor 同样需要更新自己的逐请求状态。
本模块用 `BatchUpdate` 描述这类变化，但不规定任何具体的 logits 变换策略。
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, TypeAlias

import torch

from nanovllm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from nanovllm.config import NanovllmConfig


class MoveDirectionality(Enum):
    """描述持久 batch 内两个槽位之间的请求移动方式。

    Attributes:
        UNIDIRECTIONAL: 将 ``source_index`` 的请求单向移动到空的
            ``target_index``。源槽位随后不再保存该请求。
        SWAP: 交换两个仍然存活的请求槽位。两个槽位原有的 processor 状态也
            必须一并交换。
    """

    # 单向移动常用于把尾部请求搬进中间的空槽位，保持活跃 batch 紧凑。
    UNIDIRECTIONAL = auto()

    # 交换用于两个存活请求需要调换槽位、但不能先覆盖其中任一个的场景。
    SWAP = auto()


# 被移出持久 batch 的请求槽位下标。
RemovedRequest: TypeAlias = int

# 新写入持久 batch 的请求信息：
# (槽位下标、采样参数、完整 prompt token ids、运行中的 output token ids)。
# output token 列表必须是 Request._output_token_ids 的同一对象，而非副本；这样
# processor 便能在后续 step 中观察到最新生成 token，而不必重新接收整个请求。
AddedRequest: TypeAlias = tuple[int, SamplingParams, list[int], list[int]]

# 持久 batch 中的槽位移动信息：
# (源槽位下标、目标槽位下标、移动方向)。
MovedRequest: TypeAlias = tuple[int, int, MoveDirectionality]


@dataclass(frozen=True)
class BatchUpdate:
    """记录一次 engine step 对持久 batch 组成造成的所有变化。

    一个 logits processor 通常维护和 batch 槽位一一对应的 CPU 或 GPU 状态，
    例如 token 出现次数、最小生成长度，或未来的 grammar 状态。它必须按照
    ``removed -> added -> moved`` 的固定顺序处理本对象，才能与 `InputBatch`
    最终的槽位布局一致。

    Attributes:
        batch_size: 本轮更新完成后持久 batch 中正在使用的槽位数量。
        removed: 本轮移除的旧请求槽位下标。其顺序由 `BatchUpdateBuilder`
            统一整理为降序，便于 InputBatch 从尾部安全回收槽位。
        added: 本轮加入的请求及其初始采样状态。每项的 output token 列表是
            对请求运行中输出列表的引用，不应复制。
        moved: 本轮发生的槽位移动或交换记录。
    """

    # 当前持久 batch 的逻辑大小，即存活请求占据的前缀槽位数。
    batch_size: int

    # processor 必须先处理删除，再写入新请求，最后同步移动状态。
    removed: Sequence[RemovedRequest]
    added: Sequence[AddedRequest]
    moved: Sequence[MovedRequest]


class LogitsProcessor(ABC):
    """作用于 batch logits 的有状态处理器抽象基类。

    具体实现以后可用于 repetition penalty、min tokens、logit bias 等逻辑。
    它们共享的关键要求是：维护的逐请求状态必须通过 `update_state()` 与
    `InputBatch` 的请求槽位布局保持同步。
    """

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        """校验当前处理器需要的采样参数。

        子类可在创建请求时覆盖该方法，并在参数无法支持时抛出 `ValueError`。
        当前基类不额外约束 nano-vLLM 的最小 `SamplingParams`。

        Args:
            sampling_params: 即将绑定到一个生成请求的采样配置。
        """

    @abstractmethod
    def __init__(
        self,
        vllm_config: "NanovllmConfig",
        device: torch.device,
        is_pin_memory: bool,
    ) -> None:
        """创建一个与特定 worker 设备绑定的 processor。

        Args:
            vllm_config: nano-vLLM 的全局配置，供子类读取模型或缓存相关约束。
            device: 保存 processor GPU 状态的目标设备。
            is_pin_memory: CPU 辅助张量是否应使用 pinned memory，以便异步传输。
        """
        raise NotImplementedError

    @abstractmethod
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """对当前 batch 的 logits 应用处理规则。

        实现既可以原地修改 ``logits``，也可以返回一个新张量；返回值必须是
        后续 sampler 应使用的 logits。

        Args:
            logits: 形状通常为 ``[num_reqs, vocab_size]`` 的模型输出 logits。

        Returns:
            logits: 处理后的 logits 张量。
        """
        raise NotImplementedError

    @abstractmethod
    def is_argmax_invariant(self) -> bool:
        """返回该处理器是否不会改变贪心采样的 argmax token。

        若返回 `True`，将来 sampler 可以在处理前完成贪心 token 选择，避免
        为纯贪心请求执行不影响结果的 logits 处理。

        Returns:
            is_argmax_invariant: 处理器不会改变贪心 argmax token 时为 True。
        """
        raise NotImplementedError

    @abstractmethod
    def update_state(self, batch_update: BatchUpdate | None) -> None:
        """在每次模型前向前同步 processor 的持久请求状态。

        Args:
            batch_update: 当本轮 batch 槽位发生增删或移动时为对应的更新对象；
                若仅有 token 继续生成、槽位布局未变，则为 `None`。
        """
        raise NotImplementedError

