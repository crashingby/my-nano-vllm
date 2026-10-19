"""闭环回放本地同步 Engine，测量 TTFT、TPOT、ITL 和输出 TPS。

当前 Engine 在请求完成时才返回 completion。本脚本保留活跃 Request 引用，
每轮 step() 返回后观察实际输出长度的增长，完成请求即使已从 Scheduler 移除，
也能记录最后一个 token。测量包含 Engine 排队、执行和结果提交的本地耗时。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from time import perf_counter

# 直接执行脚本时补入工程根目录；模块方式启动沿用 Python 的包搜索路径。
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.config import BenchmarkConfig, parse_args
from benchmarks.datasets import SampleRequest
from benchmarks.latency_metrics import LatencyBenchmarkStats, RequestLatency
from benchmarks.runner import run_benchmark as run_engine_benchmark
from benchmarks.utils import nvtx_range
from nanovllm import LLM, SamplingParams
from nanovllm.logger import init_logger
from nanovllm.request import Request

logger = init_logger(__name__)


def _submit_requests(
    llm: LLM,
    requests: list[SampleRequest],
    sampling_params: list[SamplingParams],
    stats: LatencyBenchmarkStats,
    active_requests: dict[str, tuple[Request, RequestLatency]],
    max_active: int,
    next_request_idx: int,
    start_time: float,
) -> int:
    """补充请求至并发上限，保存到达时间及只读 Request 引用，返回下一输入下标。

    arrival_time 在 add_request() 前采集，因此入队开销进入 TTFT。待提交样本在
    benchmark 侧的等待不计入请求延迟；提交成功后才更新活跃集合和工作量计数。
    """
    while next_request_idx < len(requests) and len(active_requests) < max_active:
        sample = requests[next_request_idx]  # 数据已经编码，提交过程无需 tokenizer。
        arrival_time = perf_counter() - start_time  # 与所有 token 观察时间使用同一原点。
        request_id = llm.add_request(
            prompt=sample.prompt,
            sampling_params=sampling_params[next_request_idx],
            request_id=sample.request_id,
        )
        # 当前同步 Engine 的 Scheduler 持有真实 Request；保留引用以跨越完成时的删除。
        request = llm.scheduler.requests[request_id]
        latency = RequestLatency(request_id, sample.prompt_len, arrival_time)
        active_requests[request_id] = (request, latency)  # 只读生成长度，不修改推理状态。
        stats.request_latencies.append(latency)  # 提交顺序用于最终逐请求报告。
        stats.num_submitted += 1
        stats.num_input_tokens += sample.prompt_len
        next_request_idx += 1
    return next_request_idx


def replay_requests(
    llm: LLM,
    requests: list[SampleRequest],
    sampling_params: SamplingParams,
    max_active: int,
) -> LatencyBenchmarkStats:
    """持续补充请求并记录逐 token 时间，返回实际工作量及完整延迟记录。

    Args:
        llm: 已加载模型并完成启动预热的空闲同步 Engine，由调用方负责关闭。
        requests: 已编码且 ID 唯一的请求，每条带有独立目标输出长度。
        sampling_params: temperature、ignore_eos 等公共生成选项。
        max_active: 闭环回放的活跃请求数上限，包含 Engine 中排队的已提交请求。

    Returns:
        包含公共吞吐计数、TPS 和逐请求时间轴的 LatencyBenchmarkStats。

    Side Effects:
        提交并执行请求，更新模型 KV、采样 RNG 和 prefix cache。计时结束后额外
        调用一次 step()，送达最后一批完成通知。不会额外调用 CUDA synchronize。
        每请求每步最多新增一个 token；超出当前能力时抛错，避免制造虚假的 ITL。
    """
    if not requests or max_active < 1:
        raise ValueError("Replay requires non-empty requests and positive max_active")
    if len({request.request_id for request in requests}) != len(requests):
        raise ValueError("Latency replay requires unique request IDs")
    if not llm.is_finished():
        raise ValueError("Latency replay requires an idle Engine")
    # 不同数据集样本可以提供不同输出预算，所有参数构造在计时前完成。
    request_sampling_params = [
        replace(sampling_params, max_tokens=request.expected_output_len)
        for request in requests
    ]
    stats = LatencyBenchmarkStats(num_prompts=len(requests))  # 汇总所有实际提交的请求。
    active_requests: dict[str, tuple[Request, RequestLatency]] = {}  # 仅持有活跃 Request。
    next_request_idx = 0  # 下一个尚未提交的样本。
    last_reported_finished = 0  # 每完成至少 32 条请求时打印一次进度。
    start_time = perf_counter()  # 统一计时原点，包含首批请求的入队开销。
    with nvtx_range("submit_initial_requests"):
        next_request_idx = _submit_requests(
            llm, requests, request_sampling_params, stats, active_requests,
            max_active, next_request_idx, start_time,
        )
    with nvtx_range("benchmark_loop"):
        while active_requests or next_request_idx < len(requests):
            step_outputs, num_tokens = llm.step()
            step_time = perf_counter() - start_time  # 返回后立即取时，同轮 token 共用此值。
            stats.num_engine_steps += 1
            stats.num_scheduled_tokens += num_tokens
            # 先观察所有活跃引用，包含本轮已被 Scheduler 删除的完成请求。
            for request, latency in active_requests.values():
                num_new_tokens = request.num_output_tokens - len(latency.token_timestamps)
                if num_new_tokens == 1:
                    latency.record_token(step_time)  # 最后一个 token 也在完成处理之前记录。
                elif num_new_tokens != 0:
                    raise RuntimeError(
                        f"Expected at most one new token per step for {request.request_id!r}, "
                        f"got {num_new_tokens}"
                    )
            # 记录完本轮 token 后再移除完成请求，空 prefill step 不会制造 token 时间戳。
            for request_id, completion_token_ids in step_outputs:
                state = active_requests.pop(request_id, None)
                if state is None:
                    raise RuntimeError(f"Unexpected completion for {request_id!r}")
                _, latency = state  # 对照完成列表校验实际输出长度，支持 EOS 提前结束。
                latency.finish(step_time, len(completion_token_ids))
                stats.num_finished += 1
                stats.num_output_tokens += len(completion_token_ids)
            # 释放出的容量在下一步执行前补满，后续请求拥有各自真实的提交时间。
            next_request_idx = _submit_requests(
                llm, requests, request_sampling_params, stats, active_requests,
                max_active, next_request_idx, start_time,
            )
            if (
                stats.num_finished - last_reported_finished >= 32
                or stats.num_finished == len(requests)
            ):
                logger.info(
                    "Progress: finished=%d/%d active=%d submitted=%d",
                    stats.num_finished, len(requests), len(active_requests), stats.num_submitted,
                )
                last_reported_finished = stats.num_finished
            if active_requests and llm.is_finished():
                raise RuntimeError("Engine finished without returning all replayed requests")
    stats.elapsed = perf_counter() - start_time  # 包含观察、补充请求和循环内日志的开销。
    if stats.num_finished != len(requests):
        raise RuntimeError("Latency replay did not complete every request")
    llm.step()  # 最终 Worker 清理单独执行，不计入吞吐和请求延迟。
    return stats


def run_benchmark(config: BenchmarkConfig) -> LatencyBenchmarkStats:
    """复用公共启动流程执行一次本地 Engine 延迟测试，并按需保存 JSON 报告。"""
    return run_engine_benchmark(config, replay_requests)


def main() -> None:
    """解析共享 CLI，启动包含 TTFT、TPOT、ITL 与 TPS 的延迟回放。"""
    config = parse_args(description="Benchmark local Engine TTFT, TPOT, ITL and TPS.")
    run_benchmark(config)


if __name__ == "__main__":
    main()
