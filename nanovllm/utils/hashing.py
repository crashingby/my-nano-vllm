"""prefix cache 使用的通用 hash 算法。"""
from __future__ import annotations

import hashlib
import pickle
from collections.abc import Callable
from typing import Any

def sha256(input:Any) -> bytes:
    """使用 pickle 序列化后计算 SHA-256 digest。

    Args:
        input: 任意可 pickle 的结构化 Python 对象。

    Returns:
        digest: 固定 32 字节的 SHA-256 digest。
    """
    input_bytes = pickle.dumps(
        input,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    return hashlib.sha256(input_bytes).digest()

def get_hash_fn_by_name(hash_fn_name: str) -> Callable[[Any], bytes]:
    """按配置名称返回 hash 函数。

    Args:
        hash_fn_name: prefix cache 配置中的 hash 算法名称。

    Returns:
        hash_function: 对结构化 Python 对象求 hash 的函数。

    Raises:
        ValueError: 当前 nano-vLLM 尚未支持该算法时抛出。
    """
    if hash_fn_name == "sha256":
        return sha256

    raise ValueError(f"Unsupported hash function: {hash_fn_name}")