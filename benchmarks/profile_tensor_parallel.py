"""通过 Worker RPC 限定多进程 Nsight 采集窗口，分析 TP 的 GPU 通信开销。

复用正式吞吐回放和共享 CLI；预热发生在 profiler 开始前。Nsight 需开启
CUDA graph node tracing，才能看到 decode Graph 内嵌的 NCCL kernel。
采集有额外开销，本入口的 TPS 不用于替代正式 benchmark。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from benchmarks.benchmark_engine_throughput import replay_requests
from benchmarks.config import parse_args
from benchmarks.datasets import SampleRequest
from benchmarks.metrics import BenchmarkStats
from benchmarks.runner import run_benchmark
from benchmarks.utils import nvtx_range
from nanovllm import LLM, SamplingParams

if TYPE_CHECKING:
    from nanovllm.worker.worker_base import WorkerWrapperBase


def _start_worker_profile(worker: WorkerWrapperBase) -> int:
    """在每个 Worker 内等待预热结束，再开启 CUDA profiler 并返回其 rank。"""
    torch.cuda.synchronize()  # 初始化和预热 GPU 工作不能落入正式采集窗口。
    torch.cuda.cudart().cudaProfilerStart()
    return worker.worker.rank


def _stop_worker_profile(worker: WorkerWrapperBase) -> int:
    """等待该 Worker 的 GPU 工作完成后结束采集，保留原有执行结果。"""
    torch.cuda.synchronize()  # 确保最后的 NCCL 和 D2H 已记录，再请求 profiler 停止。
    torch.cuda.cudart().cudaProfilerStop()
    return worker.worker.rank


def main() -> None:
    """在所有 TP Worker 中开启采集，回放同一批数据，并保存诊断 JSON。"""
    config = parse_args(description="Profile TP communication with Nsight Systems.")
    if config.result_path is None:
        raise ValueError("--result-path is required for the profiled workload")

    def replay(
        llm: LLM,
        requests: list[SampleRequest],
        params: SamplingParams,
        max_active: int,
    ) -> BenchmarkStats:
        """只采集正式回放，主进程 NVTX 范围作为离线统计的墙钟边界。"""
        llm.model_executor.collective_rpc(_start_worker_profile)
        try:
            with nvtx_range("tp_profile"):
                return replay_requests(llm, requests, params, max_active)
        finally:
            # replay 正常返回前已排空队列；异常路径同样请求停止各 Worker 的采集。
            llm.model_executor.collective_rpc(_stop_worker_profile)

    run_benchmark(config, replay)


if __name__ == "__main__":
    main()
