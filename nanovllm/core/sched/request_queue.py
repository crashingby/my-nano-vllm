"""按 FCFS 或优先级维护 Scheduler 等待队列，不分配 KV 或推进 token。"""
import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from nanovllm.request import Request


class SchedulingPolicy(Enum):
    """选择等待队列排序规则，FCFS 按入队顺序，PRIORITY 使用 Request 比较规则。"""

    FCFS = "fcfs"
    PRIORITY = "priority"

class RequestQueue(ABC):
    """等待请求容器的公共约定。

    队列保存 Request 引用，不复制请求或改变生命周期状态。Scheduler 负责状态转换，
    prepend 在优先级实现中仍服从排序键；批量插入不会清空来源队列。
    """

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """按当前策略插入请求引用，不改变请求本身的状态。"""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """移除并返回下一个请求，空队列抛出 IndexError。

        Returns:
            request: 按策略移除的下一个待调度 Request。
        """
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """只读查看下一个请求，空队列抛出 IndexError。

        Returns:
            request: 按策略选中的下一个 Request 引用，不移除队列元素。
        """
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """归还待优先重试的请求；是否直接放到队首由具体策略决定。"""
        pass

    @abstractmethod
    def prepend_requests(self, requests: "RequestQueue") -> None:
        """重新插入来源队列的请求引用，不清空来源队列。"""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """移除指定请求，目标不存在时抛出 ValueError。"""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """移除给定集合中存在的请求，忽略不在本队列中的对象。"""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """返回队列是否仍包含待调度请求。

        Returns:
            has_requests: 队列仍含待调度请求时为 True。
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """返回保存的请求数，单位为请求而非 token。

        Returns:
            num_requests: 队列保存的请求数，单位为请求而非 token。
        """
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """按调度顺序遍历请求，不将它们从队列移除。

        Returns:
            requests_iterator: 按调度顺序读取 Request 引用的迭代器，不移除请求。
        """
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    """使用 deque 保存到达顺序，支持将抢占或暂缓请求重新插入队首。"""

    def add_request(self, request: Request) -> None:
        """将新请求追加到队尾，保持 FCFS 入队顺序。"""
        self.append(request)

    def pop_request(self) -> Request:
        """移除并返回队首请求，空队列由 deque 抛出 IndexError。

        Returns:
            request: 从 FCFS 队首移除的 Request，空队列抛出 IndexError。
        """
        return self.popleft()

    def peek_request(self) -> Request:
        """查看队首请求而不移除，空队列抛出 IndexError。

        Returns:
            request: FCFS 队首的 Request 引用，不移除元素。
        """
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """将待重试请求放回队首，使其先于普通等待请求尝试调度。"""
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """将来源队列逐项插到队首，不删除来源元素。

        deque.extendleft 会反转来源迭代顺序。来源为 [A, B] 时，新增前缀为 [B, A]，
        调用方需要按期望重试顺序组织来源队列。
        """
        self.extendleft(requests)

    def remove_request(self, request: Request) -> None:
        """原地移除指定请求，找不到时抛出 ValueError。"""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """原地过滤待删除请求，并保留剩余请求的相对顺序。"""
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # 先保存剩余请求，再原地重建 deque，Scheduler 持有的队列引用仍然有效。
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """判断是否仍有等待请求，不弹出任何元素。

        Returns:
            has_requests: FCFS 队列非空时为 True。
        """
        return len(self) > 0

    def __len__(self) -> int:
        """返回 deque 中的请求数量。

        Returns:
            num_requests: FCFS 队列当前保存的请求数量。
        """
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """从队首向队尾遍历请求引用，不复制也不移除请求。

        Returns:
            requests_iterator: 从队首到队尾读取原 Request 引用的迭代器。
        """
        return super().__iter__()


class PriorityRequestQueue(RequestQueue):
    """使用最小堆按 Request.__lt__ 选择下一个请求。

    先比较较小的 priority，再比较较早的 arrival_time，平局键由 Request 定义。
    入堆后不应原地修改排序字段，否则需要重新建堆才能保持正确顺序。
    """

    def __init__(self) -> None:
        """创建空堆，元素保留对 Scheduler 权威请求的引用。"""
        # 只有根保证最小，不能将底层数组直接当作完整的调度顺序。
        self._heap: list[Request] = []

    def add_request(self, request: Request) -> None:
        """按请求比较规则入堆，不保证按插入位置排序。"""
        heapq.heappush(self._heap, request)

    def pop_request(self) -> Request:
        """移除并返回当前最优先请求，空堆抛出 IndexError。

        Returns:
            request: 按优先级从堆顶移除的 Request，空堆抛出 IndexError。
        """
        if not self._heap:
            raise IndexError("pop from empty heap")
        return heapq.heappop(self._heap)

    def peek_request(self) -> Request:
        """查看堆顶请求而不移除，空堆抛出 IndexError。

        Returns:
            request: 当前堆顶 Request 的引用，不移除元素。
        """
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    def prepend_request(self, request: Request) -> None:
        """按原优先级重新入堆，不绕过堆的排序规则。"""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """将来源请求逐项入堆，不清空来源或覆盖原有排序键。"""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """删除指定请求后重新建堆，目标不存在时抛出 ValueError。"""
        self._heap.remove(request)
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """过滤给定请求后重新建堆，忽略不存在的目标。"""
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        self._heap = [r for r in self._heap if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """返回堆中是否仍有请求。

        Returns:
            has_requests: 优先级堆非空时为 True。
        """
        return bool(self._heap)

    def __len__(self) -> int:
        """返回堆中的请求数量。

        Returns:
            num_requests: 优先级堆中当前保存的请求数。
        """
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """从堆的浅副本依次弹出请求，原队列保持不变。

        副本只复制引用，排序键仍来自原 Request；遍历期间不应修改请求排序键。

        Yields:
            request: 从堆的浅副本按调度顺序取出的 Request 引用，原队列保持不变。
        """
        # 从副本弹出才能得到全序，同时避免遍历破坏 Scheduler 的等待状态。
        heap_copy = self._heap[:]
        while heap_copy:
            yield heapq.heappop(heap_copy)


def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """按策略创建空等待队列，不支持的策略抛出 ValueError。

    Args:
        policy: 已解析的 FCFS 或 PRIORITY 枚举。

    Returns:
        request_queue: 全新的 RequestQueue 实现，不与其他队列共享元素存储。
    """
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
