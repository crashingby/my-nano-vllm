"""把方法名或序列化 callable 适配为 Worker 的同线程调用。"""

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any
import cloudpickle

def run_method(
    obj: Any,
    method: str | bytes | Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """解析公共 RPC 的目标表示，并在当前线程直接调用。

    Args:
        obj: Worker 或 Wrapper 端点。
        method: 方法名、cloudpickle 序列化函数，或可调用对象。方法名通过 getattr
            得到绑定方法；后两种形式额外把 obj 作为函数第一个参数。
        args: 不含 obj 的业务位置参数。
        kwargs: 直接转交给目标函数的关键字参数。

    Returns:
        result: 原始调用返回值，不额外包装 Future 或 rank 列表。

    Raises:
        NotImplementedError: 方法名无法在 obj 上解析。
        Exception: 反序列化或目标调用的其他异常直接向上传播。

    Side Effects:
        由目标方法决定；本函数不建立线程、通信通道或 GPU 同步点。
    """
    if isinstance(method, bytes):
        # 序列化的是未绑定函数，恢复后显式补入 Worker 端点。
        func = partial(cloudpickle.loads(method), obj)
    elif isinstance(method, str):
        # 通过 getattr 得到的方法已绑定 obj，不能再重复传入端点。
        try:
            func = getattr(obj, method)
        except AttributeError:
            raise NotImplementedError(
                f"Method {method!r} is not implemented."
            ) from None
    else:
        func = partial(method, obj)  # type: ignore
    return func(*args, **kwargs)
