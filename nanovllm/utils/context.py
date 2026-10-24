"""保留早期 Attention 路径的进程级上下文接口。

当前主执行链使用 nanovllm.forward_context；此兼容结构不负责异步 batch 快照，
也没有线程或协程隔离。
"""
from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    """描述早期 prefill/decode 路径的长度和分页索引，不持有 KV 数据。

    cu_seqlens_q/k 是带起始零的累计长度；max_seqlen_q/k 是各请求最大长度。
    slot_mapping 把本轮 token 映射到物理 KV 槽位，context_lens 与 block_tables
    供 decode 读取历史。字段为 None 表示调用方尚未提供，并非自动分配 tensor。
    """
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    """返回当前全局 Context 引用，不复制字段或 tensor。

    Returns:
        context: 当前全局 Context 原始引用，不复制字段或 tensor。
    """
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    """用调用方提供的字段替换全局上下文，不自动恢复上一份状态。

    Args:
        is_prefill: 是否按 prefill 路径解释后续长度字段。
        cu_seqlens_q: packed Query 的累计边界。
        cu_seqlens_k: packed Key 的累计边界。
        max_seqlen_q: 单请求最大 Query 长度。
        max_seqlen_k: 单请求最大 Key 长度。
        slot_mapping: 本轮 token 的物理 KV 写入位置。
        context_lens: decode 各请求已有上下文长度。
        block_tables: 请求到物理 KV block 的分页映射。

    Side Effects:
        覆盖进程级 _CONTEXT，tensor 保留原引用，调用方负责使用结束后的清理。
    """
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    """将全局上下文替换为默认空结构，解除旧 tensor 引用，不执行设备同步。"""
    global _CONTEXT
    _CONTEXT = Context()
