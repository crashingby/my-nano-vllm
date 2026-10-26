"""共用数据准备、Engine 启停和报告保存，让各 benchmark 专注回放与测量。"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import TypeVar

from transformers import AutoTokenizer

from benchmarks.config import BenchmarkConfig
from benchmarks.datasets import SampleRequest, get_dataset
from benchmarks.metrics import BenchmarkStats
from benchmarks.utils import nvtx_range, set_random_seed
from nanovllm import LLM, SamplingParams
from nanovllm.logger import init_logger

logger = init_logger(__name__)

# 保留不同回放函数的具体统计类型，调用方仍能访问延迟报告的逐请求结果。
_StatsT = TypeVar("_StatsT", bound=BenchmarkStats)


def run_benchmark(
    config: BenchmarkConfig,
    replay: Callable[[LLM, list[SampleRequest], SamplingParams, int], _StatsT],
) -> _StatsT:
    """准备统一请求并运行指定回放，结束后打印和按需保存结果。

    Args:
        config: 数据、生成参数、Engine 配置及可选报告路径。
        replay: 使用空闲 Engine 执行请求并返回统计的函数，负责自己的计时边界。

    Returns:
        stats: replay 返回的原始统计对象，保留具体子类类型。

    Side Effects:
        读取数据与 tokenizer、创建 Engine、重置 RNG 并执行请求。Engine 创建
        成功后，正常与异常路径都会关闭它；结果文件在回放计时结束后写入。
    """
    dataset = get_dataset(config.dataset_config)  # 数据格式差异继续由注册适配器处理。
    model_config = config.vllm_config.model_config  # 两种测试使用同一份模型配置。
    logger.info("Loading tokenizer from %s", model_config.model)
    with nvtx_range("load_tokenizer"):
        tokenizer = AutoTokenizer.from_pretrained(model_config.model, use_fast=True)
    logger.info("Loading %s from %s", config.dataset_config.name, dataset.dataset_path)
    with nvtx_range("load_prompts"):
        # 编码和长度处理在请求计时前完成，回放只向 Engine 提交 token IDs。
        requests = dataset.sample(
            tokenizer=tokenizer,
            num_requests=config.num_prompts,
            output_len=config.sampling_params.max_tokens,
            max_model_len=model_config.max_model_len,
        )
    max_active = config.vllm_config.scheduler_config.max_num_seqs  # 闭环回放的并发上限。
    logger.info(
        "Initializing LLM: model=%s num_prompts=%d max_active=%d max_tokens=%d "
        "async_scheduling=%s",
        model_config.model, len(requests), max_active, config.sampling_params.max_tokens,
        config.vllm_config.scheduler_config.async_scheduling,
    )
    with nvtx_range("init_llm"):
        llm = LLM(vllm_config=config.vllm_config)
    try:
        set_random_seed(config.seed)  # 启动预热可能消耗 RNG，正式回放前重置生成初态。
        stats = replay(llm, requests, config.sampling_params, max_active)
        stats.log()  # 排序分位数和打印报告均发生在回放计时之后。
        if config.result_path is not None:
            _save_result(config, stats)
        return stats
    finally:
        llm.close()  # 测量、打印或写文件失败时也释放已创建的执行资源。


def _save_result(config: BenchmarkConfig, stats: BenchmarkStats) -> None:
    """保存配置摘要及统计快照，按需创建报告父目录并覆盖目标 JSON 文件。"""
    # 本函数只从已检查保存选项的分支调用，保护内部路径构造的不变量。
    assert config.result_path is not None
    model_config = config.vllm_config.model_config  # 只导出可序列化的配置摘要。
    scheduler_config = config.vllm_config.scheduler_config
    cache_config = config.vllm_config.cache_config  # 记录缓存开关与接纳策略，避免不同策略混测。
    report = {
        "config": {
            "model": model_config.model,
            "dtype": str(model_config.dtype),
            "enforce_eager": model_config.enforce_eager,
            "async_scheduling": scheduler_config.async_scheduling,
            "max_model_len": model_config.max_model_len,
            "max_active": scheduler_config.max_num_seqs,
            "max_num_batched_tokens": scheduler_config.max_num_batched_tokens,
            "gpu_memory_utilization": cache_config.gpu_memory_utilization,
            "block_size": cache_config.block_size,
            # 当前单 group 的实际调度与 hash 粒度均已校验为最终 block_size。
            "scheduler_block_size": cache_config.block_size,
            "hash_block_size": cache_config.block_size,
            "enable_prefix_caching": cache_config.enable_prefix_caching,
            "prefix_match_unit": cache_config.prefix_match_unit,
            "watermark": scheduler_config.watermark,
            "dataset": asdict(config.dataset_config),
            "num_prompts_requested": config.num_prompts,
            "sampling_params": asdict(config.sampling_params),
            "seed": config.seed,
        },
        "metrics": stats.to_dict(),  # 延迟子类在这里加入分布和逐请求时间戳。
    }
    path = Path(config.result_path).expanduser()  # 仅在调用方指定保存位置后进入本函数。
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Saved benchmark results to %s", path)
