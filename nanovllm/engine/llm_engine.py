"""nano-vLLM 的同步 Engine。

LLMEngine 是当前单进程、同步执行 MVP 中的中心编排层：

1. 接收用户 prompt 并构造 Request。
2. 调用 Scheduler 形成一次连续批处理的 SchedulerOutput。
3. 通过 Executor / Worker 将 SchedulerOutput 交给 GPUModelRunner 执行模型。
4. 在模型成功执行后提交 token 进度、写入 prefix cache。
5. 追加采样 token，判断 EOS / 长度停止条件，并通知 Scheduler 释放资源。

Engine 通过 Executor 初始化执行端、测量显存并下发 KV 分配方案；Scheduler
负责调度策略和物理 KV block 生命周期，GPUModelRunner 负责 Worker 侧 request
状态、输入张量准备和模型 forward。Engine 不直接维护第二份请求状态。

启动、单步执行和关闭通过 Executor 访问 Worker；请求结果处理暂保留在本类。
"""

from __future__ import annotations

import atexit
from collections.abc import Callable
from dataclasses import fields
from itertools import count
from time import perf_counter
from typing import Any, cast

from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm.config import (
    ModelConfig,
    NanovllmConfig,
    SchedulerConfig,
)
from nanovllm.core.kv_cache_utils import (
    BlockHash,
    get_kv_cache_configs,
    get_request_block_hasher,
    init_none_hash,
)
from nanovllm.core.sched.output import SchedulerOutput
from nanovllm.core.sched.scheduler import Scheduler
from nanovllm.distributed import cleanup_dist_env_and_memory
from nanovllm.executor.abstract import Executor
from nanovllm.kv_cache_interface import KVCacheConfig
from nanovllm.logger import init_logger
from nanovllm.outputs import ModelRunnerOutput
from nanovllm.request import Request, RequestStatus
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.hashing import get_hash_fn_by_name

logger = init_logger(__name__)


class LLMEngine:
    """同步连续批处理 Engine。

    Attributes:
        vllm_config: 全局推理配置；初始化时回写物理 KV block 数和编译耗时。
        tokenizer: 将字符串 prompt 编码为 token IDs、将 completion 解码为文本。
        model_executor: 创建和调用 Worker 的执行端；当前选择单卡 UniProcExecutor。
        available_gpu_memory_for_kv_cache: profiling 得到的 KV 字节预算，测量前为 -1。
        request_block_hasher: 所有 Request 共享的完整逻辑 block 哈希构造函数。
        scheduler: 请求队列、调度决策和 Scheduler 侧 KV block 元数据的所有者。
        _request_id_counter: 自动生成 request ID 的单调递增计数器。
        _is_closed: 关闭入口已经开始执行的标记，用于拒绝新请求和重复关闭。
    """
    def __init__(self, model: str, **kwargs: Any) -> None:
        """创建配置、tokenizer、执行端和正式 KV cache，再初始化 Scheduler。

        当前保留单个 LLMEngine 编排同步推理。执行端启动和 KV 初始化顺序参考
        vLLM EngineCore：Executor 构造负责加载权重，显存测量与 KV 规划随后进行，
        正式 KV 初始化和预热结束后，Scheduler 才能使用确定的物理 block 容量。

        Args:
            model: 本地 Hugging Face 模型目录。
            **kwargs: ``NanovllmConfig`` 的子配置，或为兼容旧入口保留的
                ``enforce_eager``、``max_model_len``、``max_num_batched_tokens``、
                ``max_num_seqs`` 等扁平参数。

        Raises:
            TypeError: kwargs 包含 NanovllmConfig 不支持的字段，或重复传入 ``model``
                时抛出。
            ValueError: 显式 model_config 的路径与 model 不一致，或配置、显存预算
                无法生成有效的 KV 分配方案时抛出。
            NotImplementedError: Executor 不支持配置指定的并行方式时抛出。

        Side Effects:
            读取本地模型配置与 tokenizer；创建 Worker、初始化设备并加载权重；
            执行显存 profiling，分配正式 KV cache，并按配置预热和捕获 CUDA Graph。
            随后初始化 prefix hash 根节点，创建 Scheduler，注册进程退出时的关闭回调。
        """
        # NanovllmConfig 的字段由 Engine 直接转发；hf_config 等派生状态没有入口。
        config_field_names = {
            config_field.name
            for config_field in fields(NanovllmConfig)
            if config_field.init
        }
        # 保留旧 LLM(model, enforce_eager=..., max_num_seqs=...) 调用方式；它们会
        # 在此边界被翻译成对应子配置，内部代码不再读取旧顶层字段。
        model_config_field_names = {
            config_field.name
            for config_field in fields(ModelConfig)
            if config_field.init and config_field.name != "model"
        }
        scheduler_config_field_names = {
            config_field.name
            for config_field in fields(SchedulerConfig)
            if config_field.init
        }
        supported_kwargs = (
            config_field_names
            | model_config_field_names
            | scheduler_config_field_names
        )
        unknown_kwargs = set(kwargs) - supported_kwargs
        if unknown_kwargs:
            unknown_names = ", ".join(sorted(unknown_kwargs))
            raise TypeError(f"Unsupported Engine arguments: {unknown_names}")

        if "model" in kwargs:
            raise TypeError("model must be passed as the first positional argument")
        # 显式传入子配置时，禁止同时传对应的旧扁平字段，避免两个来源相互矛盾。
        model_config = cast(ModelConfig | None, kwargs.pop("model_config", None))
        scheduler_config = cast(
            SchedulerConfig | None,
            kwargs.pop("scheduler_config", None),
        )
        legacy_model_kwargs = {
            field_name: kwargs.pop(field_name)
            for field_name in model_config_field_names & set(kwargs)
        }
        legacy_scheduler_kwargs = {
            field_name: kwargs.pop(field_name)
            for field_name in scheduler_config_field_names & set(kwargs)
        }
        if model_config is not None and legacy_model_kwargs:
            conflicting_names = ", ".join(sorted(legacy_model_kwargs))
            raise TypeError(
                "model_config cannot be combined with legacy model arguments: "
                f"{conflicting_names}"
            )
        if scheduler_config is not None and legacy_scheduler_kwargs:
            conflicting_names = ", ".join(sorted(legacy_scheduler_kwargs))
            raise TypeError(
                "scheduler_config cannot be combined with legacy scheduler "
                f"arguments: {conflicting_names}"
            )
        # 旧入口没有传子配置时，按旧字段构造相应的 vLLM 风格子配置。
        if model_config is None:
            model_config = ModelConfig(model=model, **legacy_model_kwargs)
        elif model_config.model != model:
            raise ValueError(
                "The positional model path must match model_config.model, got "
                f"{model!r} and {model_config.model!r}."
            )
        if scheduler_config is None:
            scheduler_config = SchedulerConfig(**legacy_scheduler_kwargs)

        # NanovllmConfig 只聚合已经完成各自校验的子配置及跨领域配置。
        self.vllm_config: NanovllmConfig = NanovllmConfig(
            model_config=model_config,
            scheduler_config=scheduler_config,
            **cast(dict[str, object], kwargs),
        )
        # executor_class 是执行器类型；先选择类型，使不支持的 TP 配置在加载执行端前报错。
        executor_class = Executor.get_class(self.vllm_config)
        logger.info("Loading tokenizer from %s", self.vllm_config.model_config.model)
        # EOS 必须在创建 Scheduler 前写入 NanovllmConfig，因为 Scheduler 保存 EOS 配置。
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
        # Scheduler 的 CPU block 池与 Worker 的 GPU pages 使用同一方案，包含保留空块 0。
        self.scheduler: Scheduler = Scheduler(
            vllm_config=self.vllm_config,
            kv_cache_config=kv_cache_config,
        )
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
        """关闭执行端，再清理当前 Engine 使用的进程组和空闲设备缓存。

        用户调用和 atexit 回调共享该入口。沿用 vLLM EngineCore.shutdown() 的
        核心顺序：先由 Executor / Worker 停止模型执行并清理模型、KV 引用，再
        销毁进程内的 TP、WORLD 和默认 process group，最后回收空闲 allocator 缓存。
        当前按同进程中一个 Engine 管理推理资源的生命周期组织关闭。

        当前 Scheduler 只管理 CPU 请求队列和 KV block 元数据，没有 connector 或
        事件发布器等需显式停止的外部资源，因此不额外增加 Scheduler.shutdown()。

        Raises:
            Exception: 执行端或进程组清理失败时继续向上传播。关闭开始后仍拒绝
                新请求，重复 close() 不重试已经进入的关闭流程。

        Side Effects:
            标记 Engine 已关闭，注销 atexit 回调，委托执行端清理并重置全局分布式
            状态。未完成请求不再继续生成；仍被其他对象引用的 tensor 不会因清空
            allocator 缓存而释放。调用方应在当前同步 step() 返回后调用。
        """
        # 显式调用与退出回调可能重复进入；关闭流程只执行一次。
        if self._is_closed:
            return
        # 先阻止后续请求和 step()，即使下游清理报错，也不能继续复用部分关闭的执行端。
        self._is_closed = True
        # 注册的 bound method 会持有 self；显式关闭时解除该引用，允许随后删除 Engine。
        atexit.unregister(self.close)
        # Worker 先同步并清理执行资源，避免仍有模型任务依赖即将销毁的通信组。
        self.model_executor.shutdown()
        # 与 EngineCore 相同，由 Engine 层收尾进程级分布式状态和空闲设备缓存。
        cleanup_dist_env_and_memory()

    def _get_next_request_id(self)->str:
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
            request_id:str|None=None,
            priority:int = 0
        )->str:
        """向 Engine 提交一个新的文本生成请求。

        Args:
            prompt: 原始字符串 prompt，或已由调用者编码好的 token IDs。
            sampling_params: 本请求独占的采样配置。
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
        prompt_token_ids = (self.tokenizer.encode(prompt) if isinstance(prompt,str) else prompt)
        self._validate_prompt(prompt_token_ids)
        # 自动 ID 方便 benchmark/example；服务端可传入自己的外部 request ID。
        if request_id is None:
            request_id = self._get_next_request_id()
        # Request 保存逻辑 token 序列和状态；物理 KV block 由 KVCacheManager 管理。
        request = Request(
            request_id= request_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            priority=priority,
            block_hasher=self.request_block_hasher,
        )
        self.scheduler.add_request(request)
        return request_id
    def abort_request(self,request_id:str)->bool:
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
    def _cache_computed_blocks(
        self,
        scheduler_output:SchedulerOutput,
    )->None:
        """将本轮 forward 成功写入的完整 KV blocks 提交至 prefix cache。

        Args:
            scheduler_output: 刚刚成功执行完成的调度输出。

        Side Effects:
            通过 KVCacheManager 为已完整计算的逻辑 blocks 建立 hash 到物理 block 的
            prefix-cache 映射。
        """
        # 只有实际进入 forward 的请求才可能写入新的 KV 内容。
        for request_id in scheduler_output.num_scheduled_tokens:
            request = self.scheduler.requests.get(request_id)
            if request is None:
                raise RuntimeError(
                    f"Scheduled request disappeared before cache commit: "
                    f"{request_id}"
                )
            # commit_scheduled_tokens 后，该数字正好表示本轮 forward 后的 KV 进度。
            self.scheduler.kv_cache_manager.cache_blocks(
                request=request,
                num_tokens=request.num_computed_tokens
            )
    def _get_finished_status(
        self,
        request:Request,
        output_token_id:int,
    )->RequestStatus|None:
        """根据追加后的输出 token 判断请求是否应结束。

        Args:
            request: 已追加本轮采样 token 的 Request。
            output_token_id: 本轮刚生成并写入 Request 的 token ID。

        Returns:
            请求应结束时返回对应 ``FINISHED_*`` 状态；否则返回 ``None``.

        Side Effects:
            请求结束时写入 ``request.stop_reason``。
        """
        # 总序列长度达到模型上下文上限后，不能再把后续 token 写入 KV cache。
        if request.num_tokens >= self.vllm_config.model_config.max_model_len:
            request.stop_reason = "length"
            return RequestStatus.FINISHED_LENGTH_CAPPED
        # max_tokens 只限制 completion，不包含 prompt token。
        if request.num_output_tokens >= request.max_tokens:
            request.stop_reason = "length"
            return RequestStatus.FINISHED_LENGTH_CAPPED
        # ignore_eos=True 时仍保留 EOS token，但继续将它作为普通 token 生成。
        if not request.ignore_eos and output_token_id == self.scheduler.eos:
            request.stop_reason = output_token_id
            return RequestStatus.FINISHED_STOPPED

        return None
    def _process_model_output(
        self,
        model_runner_output:ModelRunnerOutput,
    )->list[tuple[str,list[int]]]:
        """将采样结果写回 Request，并收集本轮完成的请求。

        Args:
            model_runner_output: 按 req_ids 行序对齐的采样结果。中间 chunk 的行
                是空列表，普通采样行只含一个 token ID。

        Returns:
            本轮结束请求的 ``(request_id, completion_token_ids)`` 列表。

        Raises:
            RuntimeError: ModelRunner 为不存在请求或未结束 chunked prefill 的请求返回
                采样 token 时抛出，表示 Engine/Worker 协议不一致。
        """
        finished_outputs: list[tuple[str, list[int]]] = []
        assert len(model_runner_output.sampled_token_ids) == len(model_runner_output.req_ids)
        # 输出行与请求 ID 快照对齐；空行表示本轮只有 KV 写入，没有有效生成 token。
        for request_id in model_runner_output.req_ids:
            req_index = model_runner_output.req_id_to_index[request_id]
            sampled_ids = model_runner_output.sampled_token_ids[req_index]
            if not sampled_ids:
                continue
            assert len(sampled_ids) == 1  # 当前不支持投机解码或多分支采样。
            output_token_id = sampled_ids[0]
            request = self.scheduler.requests.get(request_id)
            if request is None:
                raise RuntimeError(
                    f"ModelRunner returned a token for unknown request: "
                    f"{request_id}"
                )
            # 中间 prefill chunk 只能写 KV，不能从不完整 prompt 的 logits 采样。
            if request.is_prefill_chunk:
                raise RuntimeError(
                    "ModelRunner sampled an unfinished prefill request: "
                    f"{request_id}"
                )
            # 新 token 尚未进入 KV；它会在下一次 step 作为待计算 token 被调度。
            request.append_output_token_ids(output_token_id)
            finished_status=self._get_finished_status(
                request=request,
                output_token_id=output_token_id,
            )
            if finished_status is None:
                continue
            # 先复制 completion；finish_requests 之后 Scheduler 可删除其权威 Request。
            finished_outputs.append(
                (request_id, request.output_token_ids.copy())
            )
            # Scheduler 统一处理队列移除、KV block 引用释放和 Worker 清理通知。
            self.scheduler.finish_requests(
                request_ids=[request_id],
                finished_status=finished_status,
            )
        return finished_outputs
    def step(self) -> tuple[list[tuple[str, list[int]]], int]:
        """执行一次同步的调度、模型推理和结果提交。

        沿用 vLLM EngineCore.step() 的执行顺序：检查是否还有请求或清理通知，
        构造 SchedulerOutput，通过 Executor 执行模型，取得 Future 的结果，
        必要时采样，最后提交 Scheduler 侧状态。当前结果提交仍由本类的
        commit/cache/output 三个步骤组织，对应上游 Scheduler.update_from_output()。

        Returns:
            一个二元组：
            - 第一个元素是本轮结束请求的 ``(request_id, completion_token_ids)``。
            - 第二个元素是本轮实际执行的总 token 数，始终非负。仅处理 Worker
              清理通知时返回 ``([], 0)``，并不意味着跳过了执行端调用。

        Raises:
            RuntimeError: Engine 已关闭，或请求与模型输出不符合执行协议。
            Exception: 执行或采样失败时继续向上传播，不提交本轮计算进度。

        Side Effects:
            Scheduler 分配 KV blocks；Worker 更新请求镜像并执行模型与采样。
            成功取得结果后，提交计算进度、缓存完整 KV blocks、追加生成 token，
            并根据停止条件释放请求资源。当前没有执行失败后的自动回滚或重试。

        Notes:
            连续批处理中一个 batch 可同时含有 prefill、chunked prefill、decode 和
            恢复请求，因此不再使用旧实现“正数代表 prefill、负数代表 decode”的
            返回约定。non_block=True 只使用 Executor 的 Future 返回协议；当前
            UniProcExecutor 在同线程完成调用，不产生调度与执行重叠。
        """
        if self._is_closed:
            raise RuntimeError("Cannot step a closed LLMEngine")
        # 与 EngineCore 相同，待清理的已结束请求也算工作，不能只检查 waiting/running。
        if not self.scheduler.has_requests():
            return [], 0

        # scheduler_output 固化本轮输入、KV 映射和生命周期事件，进度仍是 forward 前的值。
        scheduler_output = self.scheduler.schedule()
        # future 是单结果 Future。即使总 token 数为零，也要让 Runner 应用结束/抢占通知；
        # Runner 会在同步状态后返回空输出，不执行模型 forward。
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        # model_output 为 None 表示 forward 已完成、logits 待采样；空输出对象则无需采样。
        model_output = future.result()
        if model_output is None:
            # 当前仅有 temperature sampling，没有 grammar bitmask 或异步 abort 队列。
            model_output = self.model_executor.sample_tokens()

        # 对应上游 update_from_output() 的状态提交：执行和采样成功后才推进已计算 token 数；
        # 零 token 调度也须提交，以轮换已经成功送达 Worker 的 finished_req_ids。
        self.scheduler.commit_scheduled_tokens(scheduler_output)
        # 已完成的 KV 进度先进入 prefix cache，不能把下面新生成、尚未 forward 的 token 算进去。
        self._cache_computed_blocks(scheduler_output)
        # finished_outputs 收集本轮完成的请求；新产生的结束通知由下一次调度下发给 Worker。
        finished_outputs = self._process_model_output(model_output)
        return finished_outputs, scheduler_output.total_num_scheduled_tokens
    def is_finished(self):
        """判断 Engine 是否已经没有等待或运行中的请求。"""
        return self.scheduler.is_finish()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict[str, str | list[int]]]:
        """同步提交一组请求，持续调度到生成结束，再按输入顺序解码结果。

        先编码并校验整批 prompt，避免后面的无效输入使前面的请求滞留队列。
        各请求可有不同 temperature 和 max_tokens，也可因 EOS 或模型总长度
        上限提前完成。中间 prefill chunk 没有生成结果时继续执行下一轮。

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
            # 同一个 SamplingParams 可安全供多个不可变语义的 Request 共享。
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
        # 集合用于筛选本次调用的结果；列表仍用于最终按输入顺序组装返回值。
        request_id_set = set(request_ids)
        completed_token_ids: dict[str, list[int]] = {}
        progress_bar = tqdm(
            total=len(request_ids),
            desc="Generating",
            dynamic_ncols=True,
            disable=not use_tqdm,
        )
        try:
            while len(completed_token_ids) < len(request_ids):
                step_start_time = perf_counter()
                step_outputs, num_scheduled_tokens = self.step()
                step_elapsed = perf_counter() - step_start_time
                # 展示本轮 packed token 吞吐；零 token 清理轮与零时间间隔不计算速率。
                if num_scheduled_tokens > 0 and step_elapsed > 0:
                    step_throughput = num_scheduled_tokens / step_elapsed
                    progress_bar.set_postfix(
                        {"Engine": f"{int(step_throughput)}tok/s"}
                    )
                for request_id, token_ids in step_outputs:
                    # Engine 可能同时服务 generate() 外部提交的请求；这里只收集本次
                    # generate 调用自身的输出。
                    if request_id in request_id_set:
                        completed_token_ids[request_id] = token_ids
                        progress_bar.update(1)

            # 最后一批请求结束后不会自然出现下一轮 forward，显式送达剩余清理通知。
            # 仍有其他活跃请求时，由后续正常 step() 一并处理，不额外推进它们的生成。
            if self.scheduler.is_finish() and self.scheduler.has_requests():
                self.step()
        finally:
            progress_bar.close()
        return [
            {
                "text": self.tokenizer.decode(
                    completed_token_ids[request_id], skip_special_tokens=True,
                ),
                "token_ids": completed_token_ids[request_id],
            }
            for request_id in request_ids
        ]
