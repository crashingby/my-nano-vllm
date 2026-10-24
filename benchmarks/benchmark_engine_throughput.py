"""回放统一 SampleRequest，测量同步或异步调度下的持续请求吞吐。

数据集格式、chat template 和输入长度处理位于 benchmarks.datasets；
配置与 CLI 位于 benchmarks.config，计数输出位于 benchmarks.metrics。
从项目根目录使用 python -m benchmarks.benchmark_engine_throughput 启动。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from time import perf_counter

# 直接执行脚本时 Python 只加入脚本目录；补入工程根目录以解析公共 benchmark 包。
# 模块方式启动已经能定位包，不修改它的搜索路径。
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.config import BenchmarkConfig, parse_args
from benchmarks.datasets import SampleRequest
from benchmarks.metrics import BenchmarkStats
from benchmarks.runner import run_benchmark as run_engine_benchmark
from benchmarks.utils import nvtx_range
from nanovllm import LLM, SamplingParams
from nanovllm.logger import init_logger

logger = init_logger(__name__)


def _submit_requests(
    llm: LLM,
    requests: list[SampleRequest],
    sampling_params: list[SamplingParams],
    stats: BenchmarkStats,
    max_active: int,
    next_request_idx: int,
    num_active: int,
) -> tuple[int, int]:
    """提交后续请求直到达到活跃上限，返回新的输入下标与活跃数量。

    输入已在计时前编码，回放不再读取原始记录或调用 tokenizer。
    add_request() 成功后才更新计数，Engine 返回的完成结果负责减少活跃数量。

    Returns:
        (next_request_idx, num_active):
            next_request_idx: 下一条尚未提交的输入下标。
            num_active: 本次补充提交后的活跃请求数。
    """
    while next_request_idx < len(requests) and num_active < max_active:
        request = requests[next_request_idx]
        llm.add_request(
            prompt=request.prompt,
            sampling_params=sampling_params[next_request_idx],
            request_id=request.request_id,
        )
        stats.num_submitted += 1
        stats.num_input_tokens += request.prompt_len
        next_request_idx += 1
        num_active += 1
    return next_request_idx, num_active


def replay_requests(
    llm: LLM,
    requests: list[SampleRequest],
    sampling_params: SamplingParams,
    max_active: int,
) -> BenchmarkStats:
    """向已初始化的 Engine 持续补充请求，返回实际计数和推理阶段耗时。

    Args:
        llm: 调用方持有的空闲 Engine；本函数不负责创建或关闭它。
        requests: 预处理后的请求，ID 唯一，输入与目标输出均已校验上下文容量。
        sampling_params: temperature、ignore_eos 等公共生成选项。
        max_active: 回放的最大活跃请求数，应与 Engine 的请求容量一致。

    Returns:
        stats: 实际回放计数、推理阶段耗时和吞吐；正常返回前已排空 Engine。

    Side Effects:
        向 Engine 提交并执行请求，消耗采样随机数，更新 KV 和 prefix cache。
        正常完成后排空在途结果和 Worker 清理通知，收尾调用不纳入吞吐计时。
    """
    if not requests or max_active < 1:
        raise ValueError("Replay requires non-empty requests and positive max_active")
    # 每请求独立保留生成参数，允许将来数据集提供不同的目标输出长度。
    request_sampling_params = [
        replace(sampling_params, max_tokens=request.expected_output_len)
        for request in requests
    ]
    stats = BenchmarkStats(num_prompts=len(requests))
    next_request_idx = 0  # 下一个尚未提交的请求下标。
    num_active = 0  # 已提交且尚未收到完成标记的请求数。
    last_reported_finished = 0  # 进度按完成数量推进，允许一个 step 同时结束多条请求。
    start_time = perf_counter()  # 数据准备、模型加载和采样参数构造已全部完成。

    with nvtx_range("submit_initial_requests"):
        next_request_idx, num_active = _submit_requests(
            llm, requests, request_sampling_params, stats,
            max_active, next_request_idx, num_active,
        )
    with nvtx_range("benchmark_loop"):
        while next_request_idx < len(requests) or not llm.is_finished():
            # engine_core_outputs 可属于旧批次；num_tokens 仅统计本次新提交量。
            engine_core_outputs, num_tokens = llm.step_fn()
            stats.num_engine_steps += 1
            stats.num_scheduled_tokens += num_tokens
            # 增量 token 每轮只累计一次，完成标记单独用于释放回放并发名额。
            for client_outputs in engine_core_outputs.values():
                for output in client_outputs.outputs:
                    stats.num_output_tokens += len(output.new_token_ids)
                    if output.finished:
                        stats.num_finished += 1
                        num_active -= 1
            next_request_idx, num_active = _submit_requests(
                llm, requests, request_sampling_params, stats,
                max_active, next_request_idx, num_active,
            )
            if (
                stats.num_finished - last_reported_finished >= 32
                or stats.num_finished == len(requests)
            ):
                logger.info(
                    "Progress: finished=%d/%d active=%d submitted=%d",
                    stats.num_finished, len(requests), num_active, stats.num_submitted,
                )
                last_reported_finished = stats.num_finished
    stats.elapsed = perf_counter() - start_time
    if num_active != 0 or stats.num_finished != len(requests):
        raise RuntimeError("Engine finished without returning all replayed requests")
    # 异步模式可能仍有提前提交的结果和完成通知，必须排空后才能复用 Engine。
    while llm.has_requests():
        llm.step_fn()
    return stats


def run_benchmark(config: BenchmarkConfig) -> BenchmarkStats:
    """准备统一请求，启动 Engine，再执行一次数据集无关的吞吐回放。

    创建的 Engine 在正常和异常路径都会关闭；返回统计可供其他脚本继续使用。
    原始文件读取、对话格式化和编码全部在模型创建与性能计时之前完成。

    Returns:
        stats: 本次吞吐回放的 BenchmarkStats，Engine 已按公共启动流程关闭。
    """
    return run_engine_benchmark(config, replay_requests)


def main() -> None:
    """解析配置并执行一次 Engine 吞吐测试。"""
    run_benchmark(parse_args())


if __name__ == "__main__":
    main()
