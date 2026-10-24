"""通过启动阶段的 eager forward 预热当前 FA2 模型执行路径。

沿用 vLLM 的 kernel_warmup(worker) 入口与后端分组检查方式。上游本文件
当前只处理 DeepGEMM / FlashInfer；这里删除这些尚未支持的分支，增加本项目
FA2 + Triton 缓存写入路径的 dummy 预热，不实现 autotune 或持久调优缓存。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nanovllm.config import CUDAGraphMode
from nanovllm.logger import init_logger

if TYPE_CHECKING:
    from nanovllm.attention.backend import AttentionBackend
    from nanovllm.worker.gpu_worker import Worker

logger = init_logger(__name__)


def kernel_warmup(worker: Worker) -> None:
    """在正式 KV cache 分配后，用小型 mixed batch 执行一次真实 FA2 forward。

    由 Worker.compile_or_warm_up_model() 在 CUDA Graph 捕获前调用，即使
    enforce_eager=True 也执行。只为全部采用 FLASH_ATTN 的 Attention groups
    预热；当前没有 DeepGEMM、FlashInfer、pooling 或跨 rank autotune 分支。

    与普通 profile_run() 的区别是 force_attention=True：后者缺省走
    Attention 的占位路径，本入口实际执行 FA2 和 Triton 缓存更新 kernel。
    dummy 的 slot_mapping 全为 -1，页表指向保留的空块，不改写请求 KV。

    Args:
        worker: 已加载模型、初始化正式 KV cache 和 metadata Builder 的 Worker；
            此时 InputBatch 尚无活跃请求，固定 16 token 的 mixed dummy 布局
            必须落在当前 token budget、请求槽位和上下文容量范围内。

    Raises:
        AssertionError: 固定 16 token 超出 Runner 输入容量。
        ValueError: mixed batch 的有效 query 长度超出最大上下文。
        RuntimeError: InputBatch 已有活跃请求，或 Attention 预热执行失败。

    Side Effects:
        原地更新 Runner 的 dummy 输入和 metadata，触发所执行路径上尚未完成的
        JIT 编译及 CUDA 库初始化；不采样、不捕获 Graph，也不推进真实请求。
        这次小 batch 不保证覆盖后续所有形状的 JIT 特化；Graph 每档预热仍由
        Runner._warmup_and_capture() 完成。
    """
    # 与上游的后端检测 helper 对应，只将后端名称替换为当前支持的 FLASH_ATTN。
    def _is_flash_attn_backend(backend: type[AttentionBackend]) -> bool:
        """查询稳定后端名；未提供名称的后端不能进入 FA2 专用预热路径。

        Returns:
            is_flash_attn_backend: 后端稳定名称为 FLASH_ATTN 时为 True；无法取得名称时为 False。
        """
        try:
            return backend.get_name() == "FLASH_ATTN"
        except NotImplementedError:
            return False

    # 只有所有组均使用当前支持的后端才运行；不为未知 hybrid 布局构造 dummy。
    if worker.model_runner.attn_groups and all(
        _is_flash_attn_backend(group.backend)
        for groups in worker.model_runner.attn_groups
        for group in groups
    ):
        logger.info("Warming up FlashAttention kernels.")
        # num_tokens 保留上游 mixed warmup 的固定 16；当前没有自动缩小到配置容量。
        # 调用方需提供足够 token budget 和可容纳 mixed prefill query 的上下文长度。
        num_tokens = min(
            16,
            worker.vllm_config.scheduler_config.max_num_batched_tokens,
            worker.model_runner.max_model_len,
        )
        # is_profile=True 强制 Runner 选择 NONE eager 路径，避免预热时提前捕获 Graph；
        # force_attention=True 保证构造真实 metadata 并执行 FA，而非仅返回占位输出。
        # 不传上游 skip_eplb，因为当前项目没有 EPLB；其余 dummy 控制沿用现有接口。
        worker.model_runner._dummy_run(
            num_tokens=num_tokens,
            is_profile=True,
            force_attention=True,
            create_mixed_batch=True,
        )
