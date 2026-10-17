"""Attention 后端与 Runner 共用的分页缓存约定。"""

# 对齐 vLLM：物理 manager block 0 永久保留，padding block table 指向该空块。
NULL_BLOCK_ID = 0
