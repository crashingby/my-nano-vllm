"""维护 logits processor 所需的持久 batch 更新状态。"""

from collections.abc import Iterable, Iterator
from itertools import chain
from typing import TYPE_CHECKING

from nanovllm.sample.logits_processor.interface import (
    AddedRequest,
    BatchUpdate,
    MovedRequest,
    RemovedRequest,
)

if TYPE_CHECKING:
    from nanovllm.sample.logits_processor.interface import LogitsProcessor


class BatchUpdateBuilder:
    """收集一个 engine step 内的槽位变化并构造 `BatchUpdate`。

    `InputBatch` 需要即时知道最小的空槽位，以便填入新请求或压缩 batch；而
    logits processor 只需在本轮前向前收到一次完整变更。因此本类既提供
    ``peek_removed()`` / ``pop_removed()`` 的即时查询，也在 step 结束时通过
    `get_and_reset()` 交付不可变的更新容器。

    使用约束：
        1. 同一 step 的所有删除必须先调用 `removed_append()` 注册。
        2. 第一次读取 `removed`、`peek_removed()` 或 `pop_removed()` 后，
           不可再追加删除；此时列表已被排序，继续追加会破坏排序不变量。
        3. 不应直接修改 `_removed`、`added` 或 `moved`，只能使用本类接口
           或由 InputBatch 在明确的批更新阶段追加新增/移动记录。
    """

    # 当前 step 中被删除的旧请求槽位；第一次读取时会按降序排序。
    _removed: list[RemovedRequest]

    # 标识 `_removed` 是否已经排序，用于避免重复排序并禁止排序后的追加。
    _is_removed_sorted: bool

    # 当前 step 写入空槽位的新请求及其初始化信息。
    added: list[AddedRequest]

    # 当前 step 为压缩或重排 batch 而产生的请求槽位移动记录。
    moved: list[MovedRequest]

    def __init__(
        self,
        removed: list[RemovedRequest] | None = None,
        added: list[AddedRequest] | None = None,
        moved: list[MovedRequest] | None = None,
    ) -> None:
        """创建一个空的、可跨多个 engine step 重用的更新收集器。

        可选列表参数主要用于单元测试；正常运行时不传，由本类创建独立的空
        列表。传入的空列表也必须保留其对象身份，不能用 ``removed or []``，
        否则调用方无法观察后续追加的内容。

        Args:
            removed: 可选的初始删除槽位列表。
            added: 可选的初始新增请求列表。
            moved: 可选的初始槽位移动列表。
        """

        self._removed = [] if removed is None else removed
        self._is_removed_sorted = False
        self.added = [] if added is None else added
        self.moved = [] if moved is None else moved

        # 即便未来某种模式不需要填充 added/moved，也可据此低成本判断 batch 是否变过。
        self.batch_changed = False

    def _ensure_removed_sorted(self) -> None:
        """将删除槽位按降序排序，且每个 step 至多执行一次。

        降序存储后，列表末尾恰好是数值最小的槽位下标；`pop()` 因而可以在
        O(1) 时间取出最早应复用的空槽位。
        """

        if not self._is_removed_sorted:
            self._removed.sort(reverse=True)
            self._is_removed_sorted = True

    @property
    def removed(self) -> list[RemovedRequest]:
        """返回按降序排列的本轮删除槽位列表。

        访问该属性会冻结本轮的删除注册阶段；之后若仍调用 `removed_append()`，
        会抛出异常以保护排序和槽位复用顺序。
        """

        self._ensure_removed_sorted()
        return self._removed

    def removed_append(self, index: int) -> None:
        """登记一个从持久 batch 移除的请求槽位。

        Args:
            index: 已完成、被抢占或将被替换的请求所在 batch 槽位下标。

        Raises:
            RuntimeError: 删除列表已被读取并排序，不能再安全追加时抛出。
        """

        if self._is_removed_sorted:
            raise RuntimeError(
                "Cannot register a removed request after removed indices have "
                "been read in this step."
            )

        self._removed.append(index)
        self.batch_changed = True

    def has_removed(self) -> bool:
        """返回本轮是否至少登记了一个待回收槽位。"""

        return bool(self._removed)

    def peek_removed(self) -> int | None:
        """查看数值最小的待回收槽位，但不从列表中移除。

        这不是无副作用的只读查询：第一次调用会将 `_removed` 排为降序，并把
        `_is_removed_sorted` 设为 `True`。从这一刻起，本轮删除收集阶段结束，
        不可再调用 `removed_append()`；否则新追加元素会破坏“列表末尾是最小
        槽位”的不变量。

        Returns:
            最小的删除槽位下标；若本轮没有删除请求则返回 `None`。
        """

        if not self.has_removed():
            return None

        self._ensure_removed_sorted()
        return self._removed[-1]

    def pop_removed(self) -> int | None:
        """取走数值最小的待回收槽位，用于立刻安放一个新请求。

        Returns:
            被取走的最小删除槽位下标；若本轮没有删除请求则返回 `None`。
        """

        if not self.has_removed():
            return None

        self._ensure_removed_sorted()
        return self._removed.pop()

    def reset(self) -> bool:
        """清空当前 step 的累计变化，并返回此前是否发生过 batch 变化。

        该方法适用于调用方不需要把变化交给 logits processor 的场景。常规
        前向路径应使用 `get_and_reset()`，以免丢失 processor 的状态同步。

        Returns:
            清空前 `batch_changed` 的值。
        """

        self._is_removed_sorted = False
        self._removed.clear()
        self.added.clear()
        self.moved.clear()

        batch_changed = self.batch_changed
        self.batch_changed = False
        return batch_changed

    def get_and_reset(self, batch_size: int) -> BatchUpdate | None:
        """构造本轮 `BatchUpdate`，并将 builder 重置给下一 engine step。

        返回的 `BatchUpdate` 保留当前三个列表对象；随后 builder 会换用新的
        空列表，而非清空旧列表。这样 processor 在本次前向中仍能读取完整的
        更新内容，builder 则可以安全开始收集下一轮事件。

        Args:
            batch_size: 本轮所有增删和移动处理完后，持久 batch 的逻辑大小。

        Returns:
            有槽位变化时返回 `BatchUpdate`；仅有连续 decode、布局未改变时
            返回 `None`。
        """

        # 新 step 可以重新登记删除；旧 step 的排序状态不应泄漏到下一轮。
        self._is_removed_sorted = False
        self.batch_changed = False

        # 没有任何布局变化时，processor 无须进行状态重排。
        if not any((self._removed, self.added, self.moved)):
            return None

        # BatchUpdate 内的事件顺序由 processor 按 removed -> added -> moved 处理。
        batch_update = BatchUpdate(
            batch_size=batch_size,
            removed=self._removed,
            added=self.added,
            moved=self.moved,
        )

        # 不能 clear()：BatchUpdate 仍引用这些列表，必须把 builder 切换到新列表。
        self._removed = []
        self.added = []
        self.moved = []
        return batch_update


class LogitsProcessors:
    """按是否影响贪心 argmax 对已初始化的 processor 进行分组。

    当前 MVP 尚未注册任何具体 processor，但保留该容器后，采样器可以直接
    复用 vLLM 的两段式执行路径：先处理不改变 argmax 的规则，再只为需要的
    请求运行会改变 token 选择结果的规则。
    """

    def __init__(self, logitsprocs: Iterable["LogitsProcessor"] | None = None) -> None:
        """初始化 processor 分组。

        Args:
            logitsprocs: 已创建的 processor 可迭代对象；省略时创建空容器。
        """

        # 不会改变贪心 argmax 的 processor，例如仅记录状态的实现。
        self.argmax_invariant: list[LogitsProcessor] = []

        # 会改变 token 排名或可选集合的 processor，例如未来的 repetition penalty。
        self.non_argmax_invariant: list[LogitsProcessor] = []

        if logitsprocs is not None:
            # 逐个分类，避免调用方还要理解 sampler 内部的两条执行路径。
            for logitsproc in logitsprocs:
                if logitsproc.is_argmax_invariant():
                    self.argmax_invariant.append(logitsproc)
                else:
                    self.non_argmax_invariant.append(logitsproc)

    @property
    def all(self) -> Iterator["LogitsProcessor"]:
        """按 sampler 的执行分组顺序遍历所有 processor。"""

        return chain(self.argmax_invariant, self.non_argmax_invariant)
