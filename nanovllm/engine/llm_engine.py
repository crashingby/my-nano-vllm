"""nano-vLLM 的同步 Engine。

LLMEngine 是当前单进程、同步执行 MVP 中的中心编排层：

1. 接收用户 prompt 并构造 Request。
2. 调用 Scheduler 形成一次连续批处理的 SchedulerOutput。
3. 通过 Executor / Worker 将 SchedulerOutput 交给 GPUModelRunner 执行模型。
4. 将执行结果交给 Scheduler.update_from_output()，更新请求并清理结束状态。
5. generate() 累计增量输出，按输入顺序返回完整生成结果。

Engine 通过 Executor 初始化执行端、测量显存并下发 KV 分配方案；Scheduler
负责调度策略和物理 KV block 生命周期，GPUModelRunner 负责 Worker 侧 request
状态、输入张量准备和模型 forward。Engine 不直接维护第二份请求状态。

启动、单步执行和关闭通过 Executor 访问 Worker；逻辑状态处理统一由 Scheduler 负责。
"""

from __future__ import annotations

import atexit
from collections import deque
from concurrent.futures import Future
from typing import cast
from collections.abc import Callable
from dataclasses import replace
from itertools import count
from time import perf_counter

from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm.config import NanovllmConfig
from nanovllm.core.kv_cache_utils import (
    BlockHash,
    get_kv_cache_configs,
    get_request_block_hasher,
    init_none_hash,
)
from nanovllm.core.sched.scheduler import Scheduler
from nanovllm.core.sched.async_scheduler import AsyncScheduler
from nanovllm.core.sched.output import SchedulerOutput
from nanovllm.outputs import ModelRunnerOutput
from nanovllm.distributed import cleanup_dist_env_and_memory
from nanovllm.engine import EngineCoreOutputs
from nanovllm.executor.abstract import Executor
from nanovllm.kv_cache_interface import KVCacheConfig
from nanovllm.logger import init_logger
from nanovllm.request import Request, RequestStatus
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.hashing import get_hash_fn_by_name

logger = init_logger(__name__)


class LLMEngine:
    """同步连续批处理 Engine。

    Attributes:
        vllm_config: 调用方传入的完整配置引用；初始化时回写 EOS、物理 KV block
            数和编译耗时，执行端与 Scheduler 从同一对象读取子配置。
        tokenizer: 将字符串 prompt 编码为 token IDs、将 completion 解码为文本。
        model_executor: 创建和调用 Worker 的执行端；当前选择单卡 UniProcExecutor。
        available_gpu_memory_for_kv_cache: profiling 得到的 KV 字节预算，测量前为 -1。
        request_block_hasher: 所有 Request 共享的完整逻辑 block 哈希构造函数。
        scheduler: 请求队列、调度决策和 Scheduler 侧 KV block 元数据的所有者。
        _request_id_counter: 自动生成 request ID 的单调递增计数器。
        _is_closed: 关闭入口已经开始执行的标记，用于拒绝新请求和重复关闭。
    """
    def __init__(self, vllm_config: NanovllmConfig) -> None:
        """使用完整配置创建 tokenizer、执行端、正式 KV cache 和 Scheduler。

        当前保留单个 LLMEngine 编排同步推理。执行端启动和 KV 初始化顺序参考
        vLLM EngineCore：Executor 构造负责加载权重，显存测量与 KV 规划随后进行，
        正式 KV 初始化和预热结束后，Scheduler 才能使用确定的物理 block 容量。

        Args:
            vllm_config: 调用方已经构造并完成静态校验的总配置。Engine 直接保存
                该对象，并传给 Executor 和 Scheduler，不重建或复制任何子配置。
                每个 Engine 应使用独立配置，避免共享模型层注册表等运行时状态。

        Raises:
            ValueError: 配置与显存预算无法生成有效的 KV 分配方案时抛出。
            NotImplementedError: Executor 不支持配置指定的并行方式时抛出。

        Side Effects:
            读取本地 tokenizer；创建 Worker、初始化设备并加载权重；
            执行显存 profiling，分配正式 KV cache，并按配置预热和捕获 CUDA Graph。
            随后初始化 prefix hash 根节点，创建 Scheduler，注册进程退出时的关闭回调。
            启动结果直接回写传入配置，包括 EOS、KV 容量和编译统计等运行时字段。
        """
        # 沿用 vLLM 的配置传递方式：外层负责组装，Engine 与执行端共享完整配置。
        self.vllm_config: NanovllmConfig = vllm_config
        # executor_class 是执行器类型；先选择类型，使不支持的 TP 配置在加载执行端前报错。
        executor_class = Executor.get_class(self.vllm_config)
        logger.info("Loading tokenizer from %s", self.vllm_config.model_config.model)
        # 保存 tokenizer 的主 EOS，提交请求时再绑定到独立的采样参数。
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.vllm_config.model_config.model,
            use_fast=True,
        )
        self.vllm_config.eos = self.tokenizer.eos_token_id
        logger.info(
            "Initializing %s: tensor_parallel_size=%d eager=%s",
            executor_class.__name__,
            self.vllm_config.parallel_config.tensor_parallel_size,
            self.vllm_config.model_config.enforce_eager,
        )
        # Executor 构造会依次创建 Worker、初始化设备和加载权重，此时尚未分配正式 KV。
        self.model_executor: Executor = executor_class(self.vllm_config)
        # 沿用 EngineCore 的状态名和初值；_initialize_kv_caches() 在 profiling 后更新。
        self.available_gpu_memory_for_kv_cache: int = -1
        # kv_cache_config 是 Scheduler 使用的最终方案；返回前 Worker 已完成 KV 初始化和预热。
        kv_cache_config = self._initialize_kv_caches(self.vllm_config)

        # caching_hash_fn 是配置选择的底层摘要算法；所有请求共享同一个 prefix hash 根。
        caching_hash_fn = get_hash_fn_by_name(
            self.vllm_config.cache_config.prefix_caching_hash_algo,
        )
        init_none_hash(caching_hash_fn)
        # KV 规划可能更新 block_size，必须使用最终大小划分 Request 的完整逻辑 blocks。
        self.request_block_hasher: Callable[[Request], list[BlockHash]] = (
            get_request_block_hasher(
                block_size=self.vllm_config.cache_config.block_size,
                caching_hash_fn=caching_hash_fn,
            )
        )

        self.async_scheduling = (
            self.vllm_config.scheduler_config.async_scheduling
        )
        # 调度器和 Runner 使用同一个异步开关。
        scheduler_cls = AsyncScheduler if self.async_scheduling else Scheduler
        self.scheduler: Scheduler = scheduler_cls(
            vllm_config=self.vllm_config,
            kv_cache_config=kv_cache_config,
            include_finished_set=True,
        )
        # 当前只有单 Worker，异步模式先允许两批工作在途。
        self.batch_queue_size = 2 if self.async_scheduling else 1
        self.batch_queue: (
            deque[tuple[Future[ModelRunnerOutput], SchedulerOutput]] | None
        ) = None

        if self.batch_queue_size > 1:
            self.batch_queue = deque(maxlen=self.batch_queue_size)

        # 自动 ID 只需在当前 Engine 内唯一；调用者也可显式传入 request_id。
        self._request_id_counter = count()
        # close() 必须幂等，因为它既可能被用户调用，也会被 atexit 调用。
        self._is_closed: bool = False
        # 只为已完成初始化的实例注册关闭回调，避免退出时访问尚未创建的执行端。
        atexit.register(self.close)

    def _initialize_kv_caches(
        self,
        vllm_config: NanovllmConfig,
    ) -> KVCacheConfig:
        """测量 KV 字节预算、规划物理 pages，并初始化和预热执行端。

        由 __init__() 在权重加载后调用，流程沿用 vLLM EngineCore 的同名接口。
        当前仅有一个 Worker 和一个 FullAttention group，因此该 Worker 的配置
        可以直接供 Scheduler 使用，不需要跨 Worker 的调度布局投影。

        Args:
            vllm_config: 当前完整配置，提供模型长度、KV 策略和可选 block 数覆盖值。

        Returns:
            Scheduler 与唯一 Worker 共同使用的 KVCacheConfig。num_blocks 包含
            保留空块 0，请求可分配 block 数为 num_blocks - 1。

        Side Effects:
            更新 available_gpu_memory_for_kv_cache、cache_config.num_gpu_blocks
            和 block_size；通过 Executor 分配正式 GPU KV，完成预热和 Graph 捕获，
            并由 Executor 回写 compilation_config.compilation_time。
        """
        # start 记录 profiling、KV 分配和预热这一整个启动阶段的起点，单位为秒。
        start = perf_counter()
        # kv_cache_specs 按 Worker rank 收集 layer_name -> KVCacheSpec；当前列表长度为 1。
        kv_cache_specs = self.model_executor.get_kv_cache_specs()
        # available_gpu_memory 的每一项是对应 Worker 可用于 KV 的字节预算，不是 block 数。
        available_gpu_memory = self.model_executor.determine_available_memory()
        # 当前唯一 Worker 就是 rank 0；保留原始 profiling 预算，便于启动后查看显存规划。
        self.available_gpu_memory_for_kv_cache = available_gpu_memory[0]
        # kv_cache_configs 按 rank 描述实际分配方案；规划器校验预算并应用 block 数覆盖值。
        kv_cache_configs = get_kv_cache_configs(
            vllm_config,
            kv_cache_specs,
            available_gpu_memory,
        )
        # 单卡下 Worker 配置与 Scheduler 布局相同，直接使用同一个配置对象。
        scheduler_kv_cache_config = kv_cache_configs[0]
        # 容量镜像包含空块 0；Scheduler 仍以正式 KVCacheConfig 作为实际分配依据。
        vllm_config.cache_config.num_gpu_blocks = scheduler_kv_cache_config.num_blocks
        # kv_cache_groups 保存共享 block table 的层分组；当前只有一个 FullAttention group。
        kv_cache_groups = scheduler_kv_cache_config.kv_cache_groups
        if kv_cache_groups:
            # 使用规格确定的最终 block 大小，后续 Request hash 与 Scheduler 据此解释 token 区间。
            vllm_config.cache_config.block_size = min(
                group.kv_cache_spec.block_size for group in kv_cache_groups
            )
        # 此接口会先分配和绑定正式 KV，再预热和捕获图，保证 Graph 引用最终 KV 地址。
        self.model_executor.initialize_from_config(kv_cache_configs)
        # elapsed 是整个 KV 启动阶段的墙钟耗时，与 Executor 汇总的模型级编译时间不同。
        elapsed = perf_counter() - start
        logger.info(
            "init engine (profile, create kv cache, warmup model) took %.2f s",
            elapsed,
        )
        return scheduler_kv_cache_config

    def close(self) -> None:
        """等待在途结果，结束剩余请求，再关闭执行端和进程级资源。

        用户调用和 atexit 回调共享该入口。关闭期间只回收已经提交的 Future，
        不再调用 step() 提交新批次，也不向调用方交付生成结果。所有在途结果
        处理完后，释放 Scheduler 请求，再清理 Worker、进程组和空闲设备缓存。

        Raises:
            Exception: 在途结果失败时继续等待其余结果，完成清理后重新抛出首个
                结果异常。执行端或进程组的清理异常也向上传播。

        Side Effects:
            标记 Engine 已关闭，注销 atexit 回调，排空在途队列并取消剩余请求，
            重置全局分布式状态。调用方应在 step() 返回后调用；重复关闭不再执行
            清理，仍被外部引用的 tensor 不会因清空 allocator 缓存而释放。
        """
        # 显式调用与退出回调可能重复进入；关闭流程只执行一次。
        if self._is_closed:
            return
        # 先阻止后续请求和 step()，即使下游清理报错，也不能继续复用部分关闭的执行端。
        self._is_closed = True
        # 注册的 bound method 会持有 self；显式关闭时解除该引用，允许随后删除 Engine。
        atexit.unregister(self.close)
        try:
            pending_error: Exception | None = None
            # 先等待旧批次的 D2H；不能在 GPU 仍引用输出时销毁 Runner 和通信组。
            while self.batch_queue:
                future, _ = self.batch_queue.pop()
                try:
                    future.result()
                except Exception as error:
                    # 一个失败结果不能阻止后续在途批次收尾，保留首个异常供调用方处理。
                    if pending_error is None:
                        pending_error = error
            # 关闭不再继续生成，Worker 随后整体销毁，无需再调度完成通知批次。
            self.scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
            if pending_error is not None:
                raise pending_error
        finally:
            # 任一结果或请求清理报错后，仍须释放执行端与进程级资源。
            try:
                self.model_executor.shutdown()
            finally:
                cleanup_dist_env_and_memory()

    def _get_next_request_id(self) -> str:
        """生成当前 Engine 内尚未使用的自动 request ID。

        Returns:
            格式为 ``request-N`` 的唯一请求 ID。
        """
        while True:
            request_id = f"request-{next(self._request_id_counter)}"
            # 显式 request_id 可能恰好使用相同文本，因此仍检查 Scheduler 权威表。
            if request_id not in self.scheduler.requests:
                return request_id

    def _validate_prompt(self, prompt_token_ids: list[int]) -> None:
        """在请求入队前检查 prompt，并为至少一个生成 token 留出上下文位置。

        Args:
            prompt_token_ids: tokenizer 编码后的 prompt，或调用方直接提供的 token IDs。

        Raises:
            ValueError: prompt 为空，或长度已经达到模型上下文上限。
        """
        if not prompt_token_ids:
            raise ValueError("prompt must contain at least one token")
        # Runner 会将采样 token 写到 prompt 的后一个位置，必须落在固定输入行容量以内。
        max_model_len = self.vllm_config.model_config.max_model_len
        if len(prompt_token_ids) >= max_model_len:
            raise ValueError(
                f"Prompt length ({len(prompt_token_ids)}) must be less than "
                f"max_model_len ({max_model_len}) to leave room for generation"
            )

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        request_id: str | None = None,
        priority: int = 0,
    ) -> str:
        """向 Engine 提交一个新的文本生成请求。

        Args:
            prompt: 原始字符串 prompt，或已由调用者编码好的 token IDs。
            sampling_params: 调用方的采样配置，Engine 复制后绑定本模型 EOS。
            request_id: 可选的调用方请求 ID；未提供时由 Engine 自动生成。
            priority: 预留给未来优先级调度的排序键；当前 Scheduler 仍采用 FCFS。

        Returns:
            最终注册到 Scheduler 的 request ID。

        Raises:
            RuntimeError: Engine 已关闭时抛出。
            ValueError: prompt 为空、request ID 重复等 Request/Scheduler 输入错误时抛出。

        Notes:
            当 ``prompt`` 是 ``list[int]`` 时不复制列表，以保持与 vLLM 类似的轻量
            请求入口；调用者提交后不应原地修改该列表。
        """
        if self._is_closed:
            raise RuntimeError("Cannot add a request to a closed LLMEngine")
        # 字符串由 Engine 统一编码，避免调用者和 tokenizer 使用不同配置。
        prompt_token_ids = (
            self.tokenizer.encode(prompt) if isinstance(prompt, str) else prompt
        )
        self._validate_prompt(prompt_token_ids)
        # 自动 ID 方便 benchmark/example；服务端可传入自己的外部 request ID。
        if request_id is None:
            request_id = self._get_next_request_id()
        # 对齐上游输入处理：复制请求参数，忽略 EOS 时不设置停止用的主 EOS。
        sampling_params = replace(
            sampling_params,
            _eos_token_id=None if sampling_params.ignore_eos else self.vllm_config.eos,
        )
        # Request 保存逻辑 token 序列和状态；物理 KV block 由 KVCacheManager 管理。
        request = Request(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            priority=priority,
            block_hasher=self.request_block_hasher,
        )
        self.scheduler.add_request(request)
        return request_id

    def abort_request(self, request_id: str) -> bool:
        """取消一个仍处于 Engine 中的请求。

        Args:
            request_id: 要取消的请求 ID。

        Returns:
            找到并取消活跃请求时返回 ``True``；请求不存在或已结束时返回 ``False``.
        """
        request = self.scheduler.requests.get(request_id)
        if request is None or request.is_finished():
            return False
        # Scheduler 负责从队列删除、释放 block table，并记录 Worker 清理通知。
        self.scheduler.finish_requests(
            request_ids=[request_id],
            finished_status=RequestStatus.FINISHED_ABORTED,
        )
        return True

    def step(self) -> tuple[dict[int, EngineCoreOutputs], int]:
        """执行一轮同步调度，将结果交给 Scheduler 并返回各 client 的增量输出。

        执行顺序对应 vLLM EngineCore.step()，省略结构化输出和异步取消队列。
        请求进度、前缀缓存、停止判断及资源回收均由 Scheduler 和 Manager 维护。

        Returns:
            第一个元素按 client_index 分组，每条输出只包含本轮新增 token。
            第二个元素是实际调度 token 数，供 benchmark 累计计算工作量。
            空闲时返回 ({}, 0)；零 token 清理批次仍会被送到 Worker。

        Side Effects:
            调度时推进已承诺执行的进度，执行和采样后更新请求状态及完成通知。
            执行失败直接向上传播，当前不支持进度回滚或原地重试。
        """
        if self._is_closed:
            raise RuntimeError("Cannot step a closed LLMEngine")
        # 目前暂时用这个方式使用batch_queue()分支
        if self.batch_queue is not None:
            return self.step_with_batch_queue()
        if not self.scheduler.has_requests():
            return {}, 0

        scheduler_output = self.scheduler.schedule()  # 已保存 forward 之前的输入快照。
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        model_output = future.result()  # 同步消费本批结果，下一阶段再加入在途队列。
        if model_output is None:
            model_output = self.model_executor.sample_tokens()  # 消费当前 forward 的 logits。
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output,
        )
        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens

    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs], int]:
        """提前提交批次，并按提交顺序回收最旧结果。

        每次调用最多提交一个批次。队列尚未填满时优先继续提交；
        队列达到容量或没有新的调度工作时，等待最旧批次并更新 Scheduler。

        Returns:
            第一个元素是本次回收结果产生的增量输出，尚未回收时为空字典。
            第二个元素是本次新提交的 token 数，纯回收调用返回零。

        Side Effects:
            提交 forward 和 sampling，维护在途队列，并在结果回收后更新
            Scheduler。每批 forward 返回后的临时 logits 都会立即交给采样。
        """
        batch_queue = self.batch_queue
        assert batch_queue is not None

        # 本方法返回前始终保留至少一个空位，防止有界 deque 覆盖旧结果。
        assert len(batch_queue) < self.batch_queue_size
        # 返回值统计本次提交量，与本次回收的是哪一批结果分开。
        num_scheduled_tokens = 0
        if self.scheduler.has_requests():
            scheduler_output = self.scheduler.schedule()

            # 当前 UniProcExecutor 在本线程完成 forward 的 Python 调用，
            # GPU 工作可以继续执行；调用错误会在这里立即暴露。
            exec_future = self.model_executor.execute_model(
                scheduler_output,
                non_block=True,
            )
            num_scheduled_tokens = (
                scheduler_output.total_num_scheduled_tokens
            )
            if num_scheduled_tokens > 0:
                # 立即消费本轮 logits 并提交采样，延后的是 CPU 结果回收。
                future = cast(
                    Future[ModelRunnerOutput],
                    self.model_executor.sample_tokens(non_block=True),
                )
            else:
                # 零 token 批次已经完成 Worker 状态清理，不需要采样。
                future = cast(Future[ModelRunnerOutput], exec_future)

            # 左侧加入新批次，右侧回收旧批次，保持 FIFO。
            batch_queue.appendleft((future, scheduler_output))

            if len(batch_queue) < self.batch_queue_size and (
                num_scheduled_tokens > 0
                or self.scheduler.has_requests()
            ):
                # 先填充流水线，本次可以没有用户输出。
                return {}, num_scheduled_tokens

        elif not batch_queue:
            # Scheduler 和在途队列均为空，没有结果需要继续回收。
            return {}, 0
        # 队列已满或没有新的调度工作，主动等待最旧结果。
        future, scheduler_output = batch_queue.pop()
        model_output = future.result()

        # 必须使用该结果对应的原始快照，不能使用刚提交的新批次。
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output,
            model_output,
        )
        return engine_core_outputs, num_scheduled_tokens

    def has_requests(self) -> bool:
        """判断是否还有调度、结果回收或 Worker 清理工作。"""
        return self.scheduler.has_requests() or bool(self.batch_queue)
    def is_finished(self) -> bool:
        """判断用户请求和在途结果是否均已结束。

        Worker 完成通知仍由 has_requests() 判断，保持原有清理语义。
        """
        return self.scheduler.is_finish() and not self.batch_queue

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict[str, str | list[int]]]:
        """同步提交一组请求，持续调度到生成结束，再按输入顺序解码结果。

        先编码并校验整批 prompt，避免后面的无效输入使前面的请求滞留队列。
        各请求可有不同 temperature 和 max_tokens，也可因 EOS 或模型总长度
        上限提前完成。每轮累计 EngineCoreOutputs 的增量 token；中间 prefill
        chunk 没有生成结果时继续执行，取消请求返回此前已产生的 token。

        Args:
            prompts: 字符串 prompts，或每项均为 token IDs 的 prompts。
            sampling_params: 单个共享采样参数，或与 prompts 等长的参数列表。
            use_tqdm: 是否显示进度条。

        Returns:
            与输入 prompts 顺序一致的结果列表；每项包含 ``text`` 和 ``token_ids``。
            文本解码跳过特殊 token，token_ids 保留真实采样结果。

        Raises:
            RuntimeError: Engine 已关闭时抛出。
            ValueError: sampling_params 列表长度与 prompts 不匹配，或 prompt 长度无效。

        Side Effects:
            注册请求并多次调用 step()，更新 KV cache 和请求状态。整批结束后，
            若 Engine 已无其他活跃请求，再发送一次零 token 调度清理 Worker 镜像。
        """
        if self._is_closed:
            raise RuntimeError("Cannot generate with a closed LLMEngine")
        if not prompts:
            return []
        if isinstance(sampling_params, list):
            if len(sampling_params) != len(prompts):
                raise ValueError(
                    "sampling_params must have the same length as prompts"
                )
            sampling_params_list = sampling_params
        else:
            # add_request() 为每个请求复制并绑定采样参数，不修改调用方的共享对象。
            sampling_params_list = [sampling_params] * len(prompts)

        # prompt_token_ids_list 保留每个 prompt 的编码结果；token IDs 输入不重复编码。
        prompt_token_ids_list = [
            self.tokenizer.encode(prompt) if isinstance(prompt, str) else prompt
            for prompt in prompts
        ]
        # 所有输入通过校验后才入队；add_request() 仍独立校验，保护逐请求提交入口。
        for prompt_token_ids in prompt_token_ids_list:
            self._validate_prompt(prompt_token_ids)
        # request_ids 保留输入顺序，完成先后顺序与自动 ID 的字符串排序都不影响最终输出。
        request_ids = [
            self.add_request(prompt=prompt_token_ids, sampling_params=params)
            for prompt_token_ids, params in zip(prompt_token_ids_list, sampling_params_list)
        ]
        # 累计状态只属于本次 generate()，不再从已被 Scheduler 删除的 Request 取结果。
        output_token_ids: dict[str, list[int]] = {req_id: [] for req_id in request_ids}
        finished_request_ids: set[str] = set()  # 完成顺序独立于输入顺序。
        progress_bar = tqdm(
            total=len(request_ids),
            desc="Generating",
            dynamic_ncols=True,
            disable=not use_tqdm,
        )
        try:
            while len(finished_request_ids) < len(request_ids):
                step_start_time = perf_counter()
                engine_core_outputs, num_scheduled_tokens = self.step()
                step_elapsed = perf_counter() - step_start_time
                # 展示本轮 packed token 吞吐；零 token 清理轮与零时间间隔不计算速率。
                if num_scheduled_tokens > 0 and step_elapsed > 0:
                    step_throughput = num_scheduled_tokens / step_elapsed
                    progress_bar.set_postfix(
                        {"Engine": f"{int(step_throughput)}tok/s"}
                    )
                for client_outputs in engine_core_outputs.values():
                    for output in client_outputs.outputs:
                        if output.request_id not in output_token_ids:
                            continue  # 仅收集本次调用提交的请求。
                        output_token_ids[output.request_id].extend(
                            output.new_token_ids
                        )
                        if output.finished:
                            finished_request_ids.add(output.request_id)
                            progress_bar.update(1)
                    # 外部取消可能只有完成通知，保留该请求此前已产生的 token。
                    for req_id in client_outputs.finished_requests or ():
                        if (
                            req_id in output_token_ids
                            and req_id not in finished_request_ids
                        ):
                            finished_request_ids.add(req_id)
                            progress_bar.update(1)

            # 最后一批请求结束后不会自然出现下一轮 forward，显式送达剩余清理通知。
            # 仍有其他活跃请求时，由后续正常 step() 一并处理，不额外推进它们的生成。
            if self.scheduler.is_finish():
                # 所有逻辑请求均已结束，排空迟到结果和 Worker 清理批次。
                while self.has_requests():
                    self.step()
        finally:
            progress_bar.close()
        return [
            {
                "text": self.tokenizer.decode(
                    output_token_ids[request_id], skip_special_tokens=True,
                ),
                "token_ids": output_token_ids[request_id],
            }
            for request_id in request_ids
        ]
