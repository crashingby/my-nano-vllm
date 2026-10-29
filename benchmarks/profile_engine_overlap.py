"""为 Nsight 标记跨批次 CPU/GPU 重叠，复用原有吞吐回放，不用于性能打分。

先完成一次真实请求预热，再通过 cudaProfilerStart/Stop 限定采集窗口。
各阶段标签携带同一调度计划的 batch ID，便于区分同批发射与跨批准备。
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch
import weakref

import torch

# 支持从仓库根目录直接执行脚本，与两个正式 benchmark 的入口保持一致。
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_engine_throughput import replay_requests
from benchmarks.config import parse_args
from benchmarks.datasets import SampleRequest
from benchmarks.metrics import BenchmarkStats
from benchmarks.runner import run_benchmark
from benchmarks.utils import nvtx_range, set_random_seed
from nanovllm import LLM, SamplingParams
from nanovllm.core.sched.output import SchedulerOutput
from nanovllm.outputs import AsyncModelRunnerOutput, ModelRunnerOutput


# 在导入时记录观测入口版本，防止运行中改动文件后误认 trace 使用的实现。
_TRACE_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class _OverlapTrace:
    """临时标记单 Worker 的调度、输入准备、计算提交和结果回收阶段。

    标签为 stage|batch_id，NVTX 描述主机调用范围；GPU 工作仍按其原始流执行。
    不在阶段之间插入同步，不改变调度计划、采样结果或已有 CUDA Graph。
    """

    def __init__(self, llm: LLM) -> None:
        """保存计划身份并安装实例级包装，退出时通过 close() 恢复。

        Args:
            llm: 已初始化并完成真实请求预热的空闲 UniProc Engine。

        Side Effects:
            临时替换该 Engine 的调度与 Runner 实例方法，保活本次采集的计划。
            close() 恢复入口；包装期间不增加阶段间的 GPU 同步。
        """
        worker_wrapper = getattr(llm.model_executor, "driver_worker", None)
        if worker_wrapper is None or worker_wrapper.worker is None:
            raise NotImplementedError("Overlap tracing requires UniProcExecutor")
        runner = worker_wrapper.worker.model_runner
        self._stack = ExitStack()  # 管理临时方法替换，不修改类定义或源码文件。
        self._active_batch = 0  # 当前 execute/sample 调用所属的计划编号。
        self._pending_batch = 0  # execute 返回后等待 sample 消费的批次。
        self._plan_ids: dict[int, int] = {}  # 对象身份到本轮 trace 的连续编号。
        self._plans: list[SchedulerOutput] = []  # 保活计划，避免对象 id 被回收后复用。
        self.batches: list[dict[str, Any]] = []  # CPU 分析用布局，不保存 GPU tensor。

        original_schedule = llm.scheduler.schedule

        def schedule() -> SchedulerOutput:
            """为新计划分配编号，标签只覆盖原始 schedule 调用。

            Returns:
                plan: 本轮原始 SchedulerOutput，观测包装不修改其字段。
            """
            batch_id = len(self._plans) + 1
            with nvtx_range(f"schedule|{batch_id}"):
                plan = original_schedule()
            self._plans.append(plan)
            self._plan_ids[id(plan)] = batch_id
            lengths = list(plan.num_scheduled_tokens.values())
            self.batches.append({
                "batch_id": batch_id,
                "num_tokens": plan.total_num_scheduled_tokens,
                "num_reqs": len(lengths),
                "uniform_decode": bool(lengths) and all(n == 1 for n in lengths),
            })
            return plan

        original_execute = runner.execute_model

        def execute(scheduler_output: SchedulerOutput) -> Any:
            """把 Runner 内嵌阶段关联到执行计划，保留原返回值。

            Returns:
                model_output: 原 execute_model 的返回值，保持 None、结果对象或包装对象的原始语义。
            """
            batch_id = self._plan_ids[id(scheduler_output)]
            self._active_batch = self._pending_batch = batch_id
            try:
                with nvtx_range(f"execute|{batch_id}"):
                    return original_execute(scheduler_output)
            finally:
                self._active_batch = 0

        original_sample = runner.sample_tokens

        def sample_tokens() -> ModelRunnerOutput | AsyncModelRunnerOutput:
            """标记采样提交，并给异步输出绑定自己的回收标签。

            Returns:
                output: 原采样输出对象；异步对象额外带有本批回收阶段标签。
            """
            batch_id = self._pending_batch
            self._active_batch = batch_id
            try:
                with nvtx_range(f"sample_tokens|{batch_id}"):
                    output = original_sample()
                if isinstance(output, AsyncModelRunnerOutput):
                    # 使用弱引用，避免输出对象与保存其 bound method 的闭包形成环。
                    original_get = type(output).get_output
                    output_ref = weakref.ref(output)

                    def get_output() -> ModelRunnerOutput:
                        """回收该批结果，事件等待仍由原始实现完成。

                        Returns:
                            model_output: 原 get_output() 回收得到的 ModelRunnerOutput，不改变等待行为。
                        """
                        current_output = output_ref()
                        assert current_output is not None  # Future 保活输出直到回收。
                        with nvtx_range(f"reclaim|{batch_id}"):
                            return original_get(current_output)

                    # 输出对象只回收一次；实例包装不影响其他在途批次的完成事件。
                    output.get_output = get_output
                return output
            finally:
                self._active_batch = 0

        original_update = llm.scheduler.update_from_output

        def update(
            scheduler_output: SchedulerOutput,
            model_runner_output: ModelRunnerOutput,
        ) -> Any:
            """将 CPU 结果提交关联回原计划，不按当前 InputBatch 查批次。

            Returns:
                engine_core_outputs: 原 Scheduler.update_from_output() 返回的 client 增量映射。
            """
            batch_id = self._plan_ids[id(scheduler_output)]
            with nvtx_range(f"update|{batch_id}"):
                return original_update(scheduler_output, model_runner_output)

        try:
            self._replace(llm.scheduler, "schedule", schedule)
            self._replace(llm.scheduler, "update_from_output", update)
            self._replace(runner, "execute_model", execute)
            self._replace(runner, "sample_tokens", sample_tokens)
            # 准备阶段与模型/采样阶段分开，GPU 输入整理 kernel 不计为模型计算。
            for method, stage in (
                ("_update_states", "prep_state"),
                ("_prepare_inputs", "prep_inputs"),
                ("_get_slot_mappings", "prep_slots"),
                ("_build_attention_metadata", "prep_metadata"),
                ("_model_forward", "forward"),
                ("_sample", "sample"),
                ("_bookkeeping_sync", "bookkeeping"),
            ):
                self._wrap_stage(runner, method, stage)
            self._wrap_stage(runner.get_model(), "compute_logits", "logits")
        except Exception:
            self.close()  # 安装失败也恢复已经替换的入口，避免残留半套观测逻辑。
            raise

    def _replace(self, obj: Any, name: str, function: Callable[..., Any]) -> None:
        """替换单个实例的方法，并登记恢复操作。"""
        self._stack.enter_context(patch.object(obj, name, function))

    def _wrap_stage(self, obj: Any, name: str, stage: str) -> None:
        """用携带当前 batch ID 的 NVTX 范围包住原方法，不读取设备结果。"""
        original = getattr(obj, name)

        def measured(*args: Any, **kwargs: Any) -> Any:
            """保留原调用参数、返回值及异常传播，只增加主机范围标记。

            Returns:
                result: 被包装方法的原始返回值，不额外同步或转换。
            """
            with nvtx_range(f"{stage}|{self._active_batch}"):
                return original(*args, **kwargs)

        self._replace(obj, name, measured)

    def close(self) -> None:
        """恢复临时替换的入口，不回收请求或额外同步 GPU。"""
        self._stack.close()


def main() -> None:
    """解析原有 benchmark 参数，执行预热、Nsight 采集与布局元数据导出。"""
    config = parse_args(description="Trace cross-batch CPU/GPU overlap with Nsight.")
    if config.result_path is None:
        raise ValueError("--result-path is required to save batch attribution metadata")

    def replay(
        llm: LLM,
        requests: list[SampleRequest],
        params: SamplingParams,
        max_active: int,
    ) -> BenchmarkStats:
        """先预热真实请求链，再在原吞吐回放外限定 profiler 采集窗口。

        预热不属于正式 benchmark 的协议；其目的是让诊断聚焦稳态重叠。
        长输入可能建立前缀缓存，复现实验应单独说明该影响。

        Args:
            llm: shared runner 已创建的单 Worker Engine。
            requests: 已编码的正式请求，预热只复用其输入内容。
            params: 正式回放的公共采样参数，预热使用独立副本。
            max_active: 正式并发上限，也限制预热请求数。

        Returns:
            stats: 开启 profiler 时得到的诊断统计，不用于正式 TPS 对比。

        Side Effects:
            消耗 GPU 工作并复位 RNG，临时安装 NVTX 包装，写入批次布局 JSON。
            正常或异常退出都恢复实例方法，Engine 关闭仍由 shared runner 负责。
        """
        warmup = requests[:max_active]
        output_len = min(128, config.vllm_config.model_config.max_model_len - max(
            item.prompt_len for item in warmup
        ))
        llm.generate(
            [item.prompt for item in warmup],
            replace(params, max_tokens=output_len, ignore_eos=True),
            use_tqdm=False,
        )
        # 真实请求预热也会消耗随机数，采集前重置所有 Worker 的 RNG。
        set_random_seed(llm, config.vllm_config.model_config.seed)
        tracer = _OverlapTrace(llm)
        started = False
        try:
            torch.cuda.synchronize()  # 只在采集边界等待，阶段内部保持原始异步关系。
            torch.cuda.cudart().cudaProfilerStart()
            started = True
            stats = replay_requests(llm, requests, params, max_active)
        finally:
            try:
                if started:
                    torch.cuda.cudart().cudaProfilerStop()
            finally:
                tracer.close()  # profiler 收尾失败也不能留下实例方法替换。

        # 独立于 profiler 的时间轴保存每批布局，分析时排除零 token 与 prefill。
        trace_path = Path(config.result_path).with_suffix(".trace.json")
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps({
            "measurement": "diagnostic_only",
            "instrumentation_sha256": _TRACE_SOURCE_SHA256,
            "warmup_requests": len(warmup),
            "warmup_output_tokens": output_len,
            "batches": tracer.batches,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return stats

    run_benchmark(config, replay)


if __name__ == "__main__":
    main()
