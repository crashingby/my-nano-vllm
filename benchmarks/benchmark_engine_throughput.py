"""回放统一 SampleRequest，测量同步 Engine 的持续请求吞吐。

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

    Side Effects:
        向 Engine 提交并执行请求，消耗采样随机数，更新 KV 和 prefix cache。
        正常完成后额外下发一次零 token 清理通知，该调用不纳入吞吐计时。
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
    num_active = 0  # 已提交且尚未收到完整 completion 的请求数。
    last_reported_finished = 0  # 进度按完成数量推进，允许一个 step 同时结束多条请求。
    start_time = perf_counter()  # 数据准备、模型加载和采样参数构造已全部完成。

    with nvtx_range("submit_initial_requests"):
        next_request_idx, num_active = _submit_requests(
            llm, requests, request_sampling_params, stats,
            max_active, next_request_idx, num_active,
        )
    with nvtx_range("benchmark_loop"):
        while next_request_idx < len(requests) or not llm.is_finished():
            step_outputs, num_tokens = llm.step()
            stats.num_engine_steps += 1
            stats.num_scheduled_tokens += num_tokens
            # 完成结果包含实际输出；EOS 提前结束时不按目标长度虚增 token 数。
            for _, completion_token_ids in step_outputs:
                stats.num_finished += 1
                stats.num_output_tokens += len(completion_token_ids)
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
    llm.step()  # 下发最后一批 finished_req_ids，回收 Worker 请求镜像。
    return stats


def run_benchmark(config: BenchmarkConfig) -> BenchmarkStats:
    """准备统一请求，启动 Engine，再执行一次数据集无关的吞吐回放。

    创建的 Engine 在正常和异常路径都会关闭；返回统计可供其他脚本继续使用。
    原始文件读取、对话格式化和编码全部在模型创建与性能计时之前完成。
    """
    return run_engine_benchmark(config, replay_requests)


def main() -> None:
    """解析配置并执行一次 Engine 吞吐测试。"""
    run_benchmark(parse_args())


if __name__ == "__main__":
    main()
