"""GPU ModelRunner 的最小 vLLM V1 风格接口。

GPUModelRunner 是 Worker 内真正持有模型、KV cache、Worker 侧请求镜像和
固定容量 InputBatch 的对象。它不拥有 Scheduler 的逻辑 Request，也不决定
请求调度顺序；它只消费 SchedulerOutput 并执行模型。
"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from contextlib import contextmanager
import time
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, cast
import gc
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from nanovllm.config import (
    CUDAGraphMode,
    NanovllmConfig,
    get_layers_from_nanovllm_config,
    set_current_config,
)
from nanovllm.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
)
from nanovllm.sample.sampler import Sampler,SamplingMetadata
from nanovllm.model_executor.layers.attention import Attention
from nanovllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from nanovllm.core.sched.output import SchedulerOutput
from nanovllm.worker.gpu_input_batch import CachedRequestState, InputBatch
from nanovllm.worker.ubatch_utils import UBatchSlices
from nanovllm.worker.utils import (
    AttentionGroup,
    bind_kv_cache,
    prepare_kernel_block_sizes,
)
from nanovllm.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    CommonAttentionMetadata,
)
from nanovllm.sample.logits_processor import LogitsProcessors
from nanovllm.utils.buffer import CpuGpuBuffer
from nanovllm.utils.mem_utils import DeviceMemoryProfiler, format_gib
from nanovllm.utils.math_utils import cdiv, round_up
from nanovllm.forward_context import BatchDescriptor, set_forward_context
from nanovllm.compilation.cuda_graph import CUDAGraphStat, CUDAGraphWrapper
from nanovllm.cudagraph_dispatcher import CudagraphDispatcher
from nanovllm.attention.backends.utils import NULL_BLOCK_ID
from nanovllm.platforms import current_platform
from nanovllm.model_executor.model_loader import get_model_loader
from nanovllm.model_executor.layers.rotary_embedding import get_rope
from nanovllm.logger import init_logger
from nanovllm.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    ModelRunnerOutput,
    SamplerOutput,
)
from nanovllm.compilation.monitor import (
    set_cudagraph_capturing_enabled,
    validate_cudagraph_capturing_enabled,
)
from nanovllm.distributed.parallel_state import graph_capture
logger = init_logger(__name__)

class AsyncGPUModelRunnerOutput(AsyncModelRunnerOutput):
    """提交采样结果的独立流回传，在回收时交付 CPU token 列表。

    每个实例对应一个 batch，持有该批请求行快照、采样 tensor 和完成事件。
    后续 batch 可以改变 InputBatch，但不能改变这里保存的输出归属。
    """

    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampled_token_ids: torch.Tensor,
        invalid_req_indices: list[int],
        async_output_copy_stream: torch.cuda.Stream,
        vocab_size: int,
    ):
        """安排 D2H 和完成事件，保留回收前仍会被使用的设备存储。

        Args:
            model_runner_output: 本批请求 ID 和行号快照，token 列表尚未填入。
            sampled_token_ids: GPU 上形状为 [num_reqs, 1] 的采样结果。
            invalid_req_indices: 中间 prefill chunk 等应丢弃候选的输出行号。
            async_output_copy_stream: Runner 复用的 GPU 到 CPU 拷贝流。
            vocab_size: 词表容量；当前单 token 回收路径保留但不读取此参数。

        Side Effects:
            在拷贝流等待采样所在流，提交 CPU 拷贝并记录事件；不在这里
            等待 GPU 完成。调用方必须通过 get_output() 等待后再读取结果。
        """
        # 输出归属与无效行均属于本批次，不能在回收时查询已经变化的 InputBatch。
        self._model_runner_output = model_runner_output
        self._invalid_req_indices = invalid_req_indices

        # 每批独立事件只保护该批 D2H；设备 tensor 引用保证拷贝源保持存活。
        self.async_copy_ready_event = torch.Event()
        self._sampled_token_ids = sampled_token_ids
        self.vocab_size = vocab_size
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(async_output_copy_stream):
            # 这是设备侧依赖：采样完成后才能拷贝，不会让 CPU 等待采样结果。
            async_output_copy_stream.wait_stream(default_stream)
            self.sampled_token_ids_cpu = self._sampled_token_ids.to(
                "cpu", non_blocking=True
            )
            self.async_copy_ready_event.record()
    def get_output(self) -> ModelRunnerOutput:
        """等待本批回传结束，过滤无效行，并填入可供 Scheduler 消费的结果。

        Returns:
            output: 保存本批请求映射和真实 CPU token 列表的 ModelRunnerOutput。

        Side Effects:
            阻塞 CPU 至本批完成事件，随后释放持有的 GPU tensor 引用。
            当前只支持每行一个采样 token；该对象应只回收一次，重复读取由
            外层 AsyncOutputFuture 的结果缓存处理。
        """
        max_gen_len = self.sampled_token_ids_cpu.shape[-1]
        # CPU 从这里开始才可以安全读取 non_blocking D2H 的目标内存。
        self.async_copy_ready_event.synchronize()

        del self._sampled_token_ids
        if max_gen_len == 1:
            valid_sampled_token_ids = self.sampled_token_ids_cpu.tolist()
            # 中间 prefill 也执行了采样，但不能把这些候选作为新增输出返回。
            for i in self._invalid_req_indices:
                valid_sampled_token_ids[i].clear()
        # 上游采样断言限定单 token，尚未实现多候选/投机解码的回收分支。
        output = self._model_runner_output
        output.sampled_token_ids = valid_sampled_token_ids
        return output

class ExecuteModelState(NamedTuple):
    """保存 forward 与采样之间必须短暂保留的运行时状态。

    vLLM 将 ``execute_model()`` 与 ``sample_tokens()`` 拆开。前者执行模型、
    获得 logits，但不立刻把 token 返回 Engine；后者使用保存的 logits 和当前
    InputBatch 采样，并构造 ``ModelRunnerOutput``。

    Attributes:
        scheduler_output: 产生本次 forward 的调度计划。采样时需要它判断哪些
            request 本轮真正完成了 prefill，哪些 request 应当产生下一个 token。
        logits: 模型对需要采样位置计算出的 logits，形状通常为
            ``[num_sample_reqs, vocab_size]``。
    """
    # 原计划用于判断采样有效性；logits 必须在下一次 forward 前被消费。
    scheduler_output: SchedulerOutput
    logits: torch.Tensor

class GPUModelRunner:
    """执行单个 TP rank 的模型前向、KV cache 写入和 token 采样。

    当前只支持：

    - Qwen3 decoder-only text generation；
    - 单 KV cache group 的 full attention；
    - 同步或异步调度的连续批处理；
    - 单 token 采样；
    - 单机或未来 TP 下的同构模型分片。

    明确暂不支持：

    - PP、DP、DCP；
    - 多模态、encoder-decoder、pooling；
    - LoRA、MoE routed experts；
    - 投机解码、结构化输出、logprobs；
    - Mamba、MLA、sliding-window attention；
    - KV transfer connector。

    Attributes:
        vllm_config: 当前 Worker 使用的完整 nano-vLLM 配置。
        cache_config: 用户指定的静态 KV cache 策略。
        compilation_config: 模型构建期注册 Attention module 的静态上下文。
        device: 当前 TP rank 绑定的 CUDA device。
        max_num_tokens: 单轮连续批处理允许执行的最大 token 数。
        max_num_reqs: 单轮 batch 可容纳的最大 request 数。
        sampler: 根据 logits 和 request temperature 采样下一个 token 的模块。
        model: 已加载权重的执行入口，启用完整 Graph 时为 CUDAGraphWrapper；
            ``load_model()`` 前尚未创建。
        model_memory_usage: 模型权重加载带来的 PyTorch 显存增量，单位为字节。
        kv_cache_config: Engine profiling 后为当前 rank 下发的 KV cache 分配计划。
        kv_caches: 按 Attention 层顺序保存的 combined KV cache view 列表；
            它们与实际分配的 raw tensor 共享底层存储。
        requests: Worker 侧 request mirror。它不是 Scheduler 的权威 Request，
            只保存模型输入、输出 token 和 block table 等执行所需状态。
        input_batch: 固定容量的持久 batch。连续相邻 step 通常只有少量 request
            变化，因此避免每轮重新分配 CPU/GPU 输入 tensor。
        cudagraph_dispatcher: 后端能力解析后保存合法 keys，选择运行模式和补齐容量。
        cudagraph_batch_sizes: 升序候选捕获尺寸；实际捕获计划以 Dispatcher 为准。
        execute_model_state: forward 完成后、采样完成前的短生命周期状态。
        use_async_scheduling: 是否保留 GPU 采样结果并延后 CPU 输出回收。
        async_output_copy_stream: 异步结果 D2H 使用的独立流。
        prepare_inputs_event: 标记上一轮输入准备已消费 CPU staging buffer；
            记录点位于模型 forward 之前，不代表整批计算或输出回传完成。
    """
    def __init__(
        self,
        vllm_config: NanovllmConfig,
        device: torch.device,
    ) -> None:
        """保存配置并创建不依赖模型权重的运行时状态。

        此时不构造模型、不加载权重、不分配正式 KV cache。模型结构与权重由
        ``load_model()`` 负责，KV cache 由 Engine profiling 后调用
        ``initialize_kv_cache()`` 负责。

        Args:
            vllm_config: 当前 Engine 的完整配置。
            device: 当前 Worker 使用的 CUDA device。
        """
        # ModelRunner 与 Worker 共享同一份配置对象，不保存会失去同步的副本。
        self.vllm_config = vllm_config
        # 模型与调度配置拥有各自的生命周期，Runner 后续按领域读取它们。
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        # 下方局部别名只缩短初始化时的配置访问，不创建新的配置副本。
        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        # device 是本 TP rank 实际执行所有 CUDA 运算的设备。
        self.device = device
        # 在分配 staging buffer 前读取平台策略，统一传给 InputBatch 和 CpuGpuBuffer。
        self.pin_memory = current_platform.is_pin_memory_available()
        # 配置已解析运行精度，不再从 HF 原始配置推断参数和激活精度。
        self.dtype = model_config.dtype
        # 当前 KV 与模型使用相同精度，长度上限还会在显存规划阶段更新。
        self.max_model_len = model_config.max_model_len
        # 单步调度总 token 预算，用于固定大小输入 buffer 的后续分配。
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        # 持久 InputBatch 最多同时保存的 request 数。
        self.max_num_reqs = scheduler_config.max_num_seqs
        # 当前没有 spec decode，每个 uniform decode 请求只调度一个 query token。
        self.uniform_decode_query_len = 1

        # 保存本 rank 的 query head 数，Attention metadata 按本地分片解释。
        self.num_query_heads = model_config.get_num_attention_heads(parallel_config)

        # 采样器本身不持有请求状态；temperature 等请求级数据属于 InputBatch。

        # 控制 token 占位、GPU 结果复用、拷贝流和输入 buffer 保护的整条分支。
        self.use_async_scheduling = self.scheduler_config.async_scheduling
        self.sampler = Sampler()

        # KV cache 规划与真实 tensor 都必须在 profiling 后才能确定。
        # kv_cache_config 在 initialize_kv_cache() 时安装，此处尚无正式分配计划。
        self.kv_caches: list[torch.Tensor] = []
        # 外层按逻辑 KV group 编号索引，内层按后端和规格组织 AttentionGroup。
        self.attn_groups: list[list[AttentionGroup]] = []
        # Worker 侧持久请求镜像。其 key 与 Scheduler 的 request_id 完全一致。
        self.requests: dict[str, CachedRequestState] = {}
        placeholder_block_size = (
            self.cache_config.block_size
        )
        # 正式 KV 计划未知时先按配置建立单 group 的尺寸约定，初始化后重新核对。
        self._init_block_sizes = [placeholder_block_size]
        self._init_kernel_block_sizes = [placeholder_block_size]
        # 先分配请求工作区；正式 KV 规格要求不同尺寸时会重建空 InputBatch。
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_size=placeholder_block_size,
            kernel_block_size=placeholder_block_size,
            logitsprocs=LogitsProcessors()
        )
        # 独立拷贝流允许结果回传与后续 batch 的计算重叠。
        self.async_output_copy_stream: torch.cuda.Stream | None = None
        # 输入事件保护跨批复用的 CPU staging，区别于每批输出的 D2H 完成事件。
        self.prepare_inputs_event: torch.Event | None = None
        if self.use_async_scheduling:
            self.async_output_copy_stream = torch.cuda.Stream()
            self.prepare_inputs_event = torch.Event()
        # 沿用上游：保存升序候选尺寸；后端解析后再生成实际合法的捕获 keys。
        self.cudagraph_batch_sizes = (
            sorted(self.compilation_config.cudagraph_capture_sizes)
            if self.compilation_config.cudagraph_capture_sizes
            and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            else []
        )

        # 本轮 packed token IDs 的 CPU/GPU 固定 buffer。
        self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        # 每个 packed token 在所属 request 的逻辑位置。
        self.positions = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=self.device
        )
        # 每个 request 在 packed token 轴上的起始偏移前缀和。
        self.query_start_loc = self._make_buffer(
            self.max_num_reqs + 1, dtype=torch.int32
        )
        # 本轮 forward 后每个 request 的总长度。
        self.seq_lens = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        # CPU 长度上界复用固定 storage，metadata 构建时不从 GPU 同步取最大值。
        self.optimistic_seq_lens_cpu = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device="cpu",
            pin_memory=self.pin_memory,
        )
        # 执行前计划中的逻辑进度；异步时其依赖由同一计算流保证，CPU 不等完成。
        self.num_computed_tokens = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        # packed token -> request 槽位的映射，用于批量生成位置和元数据。
        self.req_indices = self._make_buffer(self.max_num_tokens, dtype=torch.int64)
        # 当前请求槽位到上一轮采样行号的映射；-1 表示没有可复用的上一轮结果。
        self.prev_positions = self._make_buffer(self.max_num_reqs, dtype=torch.int64)
        # 每个 request 本轮执行多少 token。
        self.num_scheduled_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )
        # 复用 arange，快速构造 positions/query offsets。
        arange_size = max(self.max_num_reqs + 1, self.max_num_tokens)
        self.arange_np = np.arange(arange_size, dtype=np.int64)
        self.query_pos = self._make_buffer(arange_size, dtype=torch.int64)
        self._arange_scratch = np.empty(arange_size, dtype=np.int64)

        # 构造时只有空 keys；Attention 后端能力解析完成后才能启用 Graph 分派。
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)

        # execute_model() 返回 None 后，在 sample_tokens() 前保存 logits。
        self.execute_model_state: ExecuteModelState | None = None

    def update_max_model_len(self, max_model_len: int) -> None:
        """更新显存规划后允许使用的逻辑上下文长度。

        Args:
            max_model_len: Worker 规划后确定的 token 长度上限。

        Side Effects:
            只更新 Runner 的长度标量，不重新分配已有输入工作区。
        """
        self.max_model_len = max_model_len

    def get_model(self) -> nn.Module:
        """返回当前已加载的原始语言模型，绕过 CUDA Graph 包装。

        Returns:
            model: 已构造且权重已加载到当前 CUDA device 的模型。

        Raises:
            ValueError: 在 ``load_model()`` 前调用时抛出。
        """
        if not hasattr(self, "model"):
            raise ValueError("Cannot get model before model has been initialized")
        # 权重、模型属性和 compute_logits 属于原模型，forward 入口保存在 self.model。
        if isinstance(self.model, CUDAGraphWrapper):
            return self.model.unwrap()
        return self.model
    
    def load_model(self, load_dummy_weights: bool = False) -> None:
        """构造 Qwen3 模型、加载 Hugging Face 权重并记录权重显存。

        实现时应使用 ``set_current_config(self.vllm_config)`` 包裹模型构造，使每个
        Attention 向 ``static_forward_context`` 注册稳定层名。随后通过项目已有
        ``load_model()`` 工具加载权重，并计算 ``model_memory_usage``。

        Side Effects:
            创建 ``self.model``；填充静态 Attention 注册表；将权重放入 GPU。
            启用 FULL 类策略时包装执行入口，get_model() 仍返回原模型。
        """
        logger.info(
            "Starting to load model %s...",
            self.model_config.model,
        )
        try:
            # profiler 只度量模型加载的显存净增量，KV 分配仍在后续规划阶段。
            with DeviceMemoryProfiler() as m:
                time_before_load = time.perf_counter()
                model_loader = get_model_loader(self.load_config)  # 根据加载策略选择权重入口。
                self.model = model_loader.load_model(
                    vllm_config=self.vllm_config,model_config=self.model_config
                )
                time_after_load = time.perf_counter()
            self.model_memory_usage = m.consumed_memory
        except torch.cuda.OutOfMemoryError as e:
            msg = (
                "Failed to load model - not enough GPU memory. "
                "Try lowering --gpu-memory-utilization to free memory for weights, "
                "increasing --tensor-parallel-size, or using --quantization. "
                "See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
                "for more tips."
            )
            combined_msg = f"{msg} (original error: {e})"
            logger.error(combined_msg)
            raise e
        logger.info(
            "Model loading took %s GiB memory and %.6f seconds",
            format_gib(self.model_memory_usage),
            time_after_load - time_before_load,
        )
        # 与上游相同，在加载完成后包装整个模型。后端尚未解析时先按配置创建；
        # 后续若降级为 NONE，ForwardContext 的 NONE 路径仍直接执行原模型。
        cudagraph_mode = self.compilation_config.cudagraph_mode
        if cudagraph_mode is not None and cudagraph_mode.has_full_cudagraphs():
            self.model = CUDAGraphWrapper(
                self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL,
            )
    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """返回当前模型所有 Attention layer 的 KV cache 存储规格。

        直接读取模型构造期注册的 AttentionLayerBase 对象，并调用各层的
        get_kv_cache_spec()；不依赖 Qwen3 的 module 遍历顺序或旧缓存辅助入口。

        Returns:
            kv_cache_spec: 从完整 Attention 层名到 ``KVCacheSpec`` 的映射。
        """
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        # 注册表保留稳定完整层名，后续规划、分配和 forward 都通过同一 key 连接。
        layer_type = cast(type[Any], AttentionLayerBase)
        attn_layers = get_layers_from_nanovllm_config(self.vllm_config,layer_type)
        for layer_name,attn_module in attn_layers.items():
            # 不需要独立 KV cache 的层可以返回 None，不为它创建空分配描述。
            if spec := attn_module.get_kv_cache_spec(self.vllm_config):
                kv_cache_spec[layer_name] = spec
        return kv_cache_spec
    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        is_profiling: bool = False,
    ) -> None:
        """按 KV group、后端、规格与 Query head 数组织 Attention 层。

        Args:
            kv_cache_config: Engine 规划的 KV cache groups 与各层规格。
            is_profiling: 是否为显存 profiling 初始化；保留上游入口，当前没有
                profiling 专用的编译和尺寸策略。

        Side Effects:
            填充 attn_groups，解析 Graph 策略并初始化 Dispatcher keys；
            此时尚未创建 metadata Builder 或实际捕获任何 Graph。
        """
        assert not self.attn_groups, "Attention groups are already initialized"
        # 层名和后端均从模型构建期间注册的 Attention 层读取。
        for kv_group_id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            kv_cache_spec = kv_cache_group.kv_cache_spec
            if not isinstance(kv_cache_spec, FullAttentionSpec):
                raise NotImplementedError("Only full decoder attention is supported")

            layers = get_layers_from_nanovllm_config(
                self.vllm_config,
                Attention,
                kv_cache_group.layer_names,
            )
            missing_layers = set(kv_cache_group.layer_names) - set(layers)
            if missing_layers:
                raise ValueError(
                    f"Attention layers are not registered: {sorted(missing_layers)}"
                )

            # 相同后端、KV 规格与本地 Q head 数的层可共用 metadata Builder。
            group_layers: dict[
                tuple[type[AttentionBackend], KVCacheSpec, int], list[str]
            ] = {}
            for layer_name in kv_cache_group.layer_names:
                layer = layers[layer_name]
                group_key = (layer.get_attn_backend(), kv_cache_spec, layer.num_heads)
                group_layers.setdefault(group_key, []).append(layer_name)

            self.attn_groups.append(
                [
                    AttentionGroup(
                        backend=group_backend,
                        layer_names=layer_names,
                        kv_cache_spec=group_spec,
                        kv_cache_group_id=kv_group_id,
                    )
                    for (group_backend, group_spec, _), layer_names
                    in group_layers.items()
                ]
            )

        # Builder 类可以在实例构造前查询能力；先确定最终策略，再创建 Builder，
        # 避免 Builder 保存仍未降级的 use_full_cuda_graph 配置快照。
        attention_backends = [
            {group.backend for group in groups} for groups in self.attn_groups
        ]
        self._check_and_update_cudagraph_mode(
            attention_backends, kv_cache_config.kv_cache_groups,
            is_profiling=is_profiling,
        )

    def _check_and_update_cudagraph_mode(
        self,
        attention_backends: list[set[type[AttentionBackend]]],
        kv_cache_groups: list[KVCacheGroupSpec],
        is_profiling: bool = False,
    ) -> None:
        """按各后端最低完整捕获能力解析策略，然后初始化合法 Graph keys。

        与 vLLM 一致，能力通过 Builder.get_cudagraph_support() 查询。当前 FA2
        声明 UNIFORM_BATCH，因此一般 FULL 会解析为 FULL_DECODE_ONLY。

        Args:
            attention_backends: 每个 KV group 使用的后端集合。
            kv_cache_groups: 与后端集合对齐的 KV 规格。
            is_profiling: 保留上游参数；当前配置解析器没有专用 profiling 分支。

        Side Effects:
            回写 compilation_config.cudagraph_mode，更新捕获尺寸快照，建立
            Dispatcher 的合法 key 集合；不预热、不捕获、不修改请求数据。
        """
        assert len(attention_backends) == len(kv_cache_groups)
        # 多种后端共用捕获入口，整体能力由最弱的一种限制，不能取最大支持级别。
        min_cg_support = AttentionCGSupport.ALWAYS
        min_cg_attn_backend = None
        for attn_backend_set, kv_cache_group in zip(attention_backends, kv_cache_groups):
            for attn_backend in attn_backend_set:
                builder_cls = attn_backend.get_builder_cls()  # 能力由 metadata Builder 声明。
                cg_support = builder_cls.get_cudagraph_support(
                    self.vllm_config, kv_cache_group.kv_cache_spec,
                )
                if cg_support.value < min_cg_support.value:
                    min_cg_support = cg_support
                    min_cg_attn_backend = attn_backend.__name__

        # 当前配置解析器不支持 spec decode、TP 对齐或 profiling 特例，裁剪其参数。
        cudagraph_mode = self.compilation_config.resolve_cudagraph_mode_and_sizes(
            min_cg_support, min_cg_attn_backend,
        )
        self.cudagraph_dispatcher.initialize_cudagraph_keys(
            cudagraph_mode, self.uniform_decode_query_len,
        )
        self.cudagraph_batch_sizes = (
            sorted(self.compilation_config.cudagraph_capture_sizes)
            if cudagraph_mode != CUDAGraphMode.NONE else []
        )

    @staticmethod
    def _is_uniform_decode(
        max_num_scheduled_tokens: int,
        uniform_decode_query_len: int,
        num_tokens: int,
        num_reqs: int,
        force_uniform_decode: bool | None = None,
    ) -> bool:
        """按上游的最大 query 长度和总数关系判断是否为 uniform decode。

        当前 query 长度固定为 1：最大调度量为 1 且总 token 数等于请求数时
        返回 True。不比较历史上下文长度，也不读取请求的 prefill/decode 标签。
        force_uniform_decode 供后续 dummy/capture 路径显式指定 batch 类型。

        Returns:
            uniform_decode: 输入满足 uniform decode 条件时为 True；显式覆盖参数优先。
        """
        return (
            (max_num_scheduled_tokens == uniform_decode_query_len)
            and (num_tokens == max_num_scheduled_tokens * num_reqs)
        ) if force_uniform_decode is None else force_uniform_decode

    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        num_reqs: int,
        num_scheduled_tokens_np: np.ndarray,
        max_num_scheduled_tokens: int,
        use_cascade_attn: bool = False,
        force_eager: bool = False,
        force_uniform_decode: bool | None = None,
    ) -> tuple[
        CUDAGraphMode, BatchDescriptor, bool, torch.Tensor | None, CUDAGraphStat | None,
    ]:
        """选择本轮运行模式和补齐容量，保留 vLLM 的五项返回结构。

        Args:
            num_tokens: 真实 packed token 数，不含 padding。
            num_reqs: 真实请求数，不含空槽位。
            num_scheduled_tokens_np: 按请求排列的调度量；保留上游参数，当前
                没有 DP/microbatch 分派分支，判断只使用最大值和总量。
            max_num_scheduled_tokens: 本轮单请求最大 query 长度。
            use_cascade_attn: 为 True 时禁止 FULL；当前后端不执行 cascade。
            force_eager: 为 True 时只允许 NONE，用于后续预热或 profiling。
            force_uniform_decode: dummy/capture 可覆盖自动判断，普通执行传 None。

        Returns:
            按返回顺序包含以下各项：
                cudagraph_mode: 当前批次实际使用的具体 CUDA Graph 模式。
                batch_descriptor: 执行容量与布局的描述符，本方法不修改输入 buffer 或补齐 metadata。
                should_ubatch: 是否拆分 microbatch；当前没有 DBO，固定为 False。
                num_tokens_across_dp: 跨 DP token 数；当前没有 DP，固定为 None。
                cudagraph_stats: 可选 Graph 指标；当前未启用，固定为 None。
        """
        # uniform 指每请求本轮 query 数相同，不要求历史 KV 长度相同。
        uniform_decode = self._is_uniform_decode(
            max_num_scheduled_tokens, self.uniform_decode_query_len,
            num_tokens, num_reqs, force_uniform_decode,
        )
        # Dispatcher 返回 (cudagraph_mode, batch_descriptor)，后者描述运行容量；实际 padding 在执行入口完成。
        cudagraph_mode, batch_descriptor = self.cudagraph_dispatcher.dispatch(
            num_tokens=num_tokens,
            uniform_decode=uniform_decode,
            valid_modes={CUDAGraphMode.NONE} if force_eager else None,
            invalid_modes={CUDAGraphMode.FULL} if use_cascade_attn else None,
        )
        return cudagraph_mode, batch_descriptor, False, None, None

    def initialize_metadata_builders(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int],
    ) -> None:
        """按最终 kernel page 大小为每个 AttentionGroup 创建 Builder。

        Args:
            kv_cache_config: 当前 Runner 的 KV cache 分配计划。
            kernel_block_sizes: 与逻辑 KV groups 对齐的 kernel page 大小。

        Side Effects:
            每个 AttentionGroup 持有一个元数据 Builder。
        """
        assert len(self.attn_groups) == len(kv_cache_config.kv_cache_groups)
        for kv_group_id, groups in enumerate(self.attn_groups):
            for group in groups:
                group.create_metadata_builders(
                    self.vllm_config,
                    self.device,
                    kernel_block_sizes[kv_group_id],
                )

    def initialize_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int],
    ) -> dict[str, torch.Tensor]:
        """创建、重解释并绑定所有 Attention layer 的 KV cache。

        方法结构对齐 vLLM：

        1. ``_allocate_kv_cache_tensors()`` 按物理分配计划创建 raw storage；
        2. ``_reshape_kv_cache_tensors()`` 按每个 group 的 spec 创建模型可用 view；
        3. ``bind_kv_cache()`` 将 layer-name 映射绑定给 Attention 和 Runner。

        Args:
            kv_cache_config: 当前 TP rank 的最终 KV cache 分配计划。
            kernel_block_sizes: 与 ``kv_cache_groups`` 顺序一致的 kernel page 大小。

        Returns:
            kv_caches: 从完整 Attention layer 名称到 combined KV cache tensor 的映射。
        """
        # 首先按 KVCacheTensor.size 分配原始字节存储。
        kv_cache_raw_tensors = self._allocate_kv_cache_tensors(
            kv_cache_config,
        )
        # 然后根据每个 group 的 KVCacheSpec 将 raw storage 解释为 K/V cache view。
        kv_caches = self._reshape_kv_cache_tensors(
            kv_cache_raw_tensors,
            kernel_block_sizes,
        )
        # 最后把按层组织的 cache 同时交给 Runner 和静态注册的 Attention modules。
        bind_kv_cache(
            kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_caches,
        )
        return kv_caches
    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        is_profiling: bool = False,
    ) -> None:
        """按后端分组、kernel page 选择、Builder、InputBatch 的顺序创建 KV cache。

        Args:
            kv_cache_config: Engine 根据显存 profiling 规划的分配方案。
            is_profiling: 是否创建临时 KV cache；当前不改变后端与布局选择。

        Side Effects:
            保存配置副本，创建 AttentionGroup、Builder 和各层 KV cache。
        """
        # Runner 持有独立配置，后续调整不回写 Engine 侧的分配方案。
        self.kv_cache_config = deepcopy(kv_cache_config)
        # TODO:initialize_attn_backend 已接入；后续补充按层选择多个后端。
        self.initialize_attn_backend(self.kv_cache_config, is_profiling)

        # 后端选择真实的 kernel page 大小，当前 FlashAttention 通常使用 256。
        kernel_block_sizes = prepare_kernel_block_sizes(
            self.kv_cache_config,
            self.attn_groups,
        )
        self._kernel_block_sizes = kernel_block_sizes

        # Builder、InputBatch 的 block table 与真实 cache view 必须使用同一规格。
        self.initialize_metadata_builders(
            self.kv_cache_config,
            kernel_block_sizes,
        )
        self.may_reinitialize_input_batch(
            self.kv_cache_config,
            kernel_block_sizes,
        )
        self.initialize_kv_cache_tensors(
            self.kv_cache_config,
            kernel_block_sizes,
        )
    def profile_run(self) -> None:
        """执行一次代表最大 batch 的 dummy forward，用于 Worker 显存 profiling。

        调用时位于模型加载后、正式 KV cache 分配前。dummy run 必须覆盖最大 token
        预算和合理的 request 数，使 Worker 能测得模型运行时峰值 activation 显存。

        它不应使用正式 Scheduler 的请求，不应修改 ``self.requests``，也不应将
        dummy KV 内容提交到 prefix cache。
        """
        hidden_states, last_hidden_states = self._dummy_run(
            self.max_num_tokens, is_profile=True
        )
        # 没有pp
        output = self._dummy_sampler_run(last_hidden_states)
        self._sync_device()
        del hidden_states, output
        gc.collect()

    @torch.inference_mode()
    def profile_cudagraph_memory(self) -> int:
        """在正式 KV cache 分配前估算模型 CUDA Graph 的显存需求。

        Worker 在 load_model() 和 profile_run() 之后调用。沿用 vLLM：创建
        临时 KV cache，使用独立 Graph pool 捕获每种模式最大的两档，再以
        首次捕获开销加上其余图的增量估算整个捕获计划。当前只执行 FULL
        decode，没有 PIECEWISE、encoder Graph 或编译捕获计数器。

        Returns:
            total_estimate: Graph 计划的估算显存字节数；计划为空时返回 0。
                第二档增量至少按每图 1 MiB 计算，以覆盖 driver 开销。这是估算，
                不是对所有形状逐一捕获后的精确值。

        Side Effects:
            临时绑定 Attention 的 KV cache，修改 dummy 输入与 metadata。
            捕获阶段成功或失败后均关闭新增捕获，沿用上游的全局 Wrapper
            注册表清图并恢复原 pool，清空 Dispatcher keys，释放临时 KV cache。
            此入口按单个 Engine 的启动期调用；后续必须重新调用
            initialize_kv_cache()，再通过 capture_model() 捕获正式图。

        Raises:
            RuntimeError: dummy 预热或 CUDA Graph 捕获失败；捕获异常继续传播。
            ValueError: 临时 KV cache 规划失败。
        """
        # 先按模型规格创建临时 KV；后端能力解析同时生成本轮捕获计划。
        with set_current_config(self.vllm_config):
            self._init_minimal_kv_cache_for_profiling()

        # 当前没有 compilation_counter，省略上游捕获计数器的保存与恢复。
        # capture_descs 是按模式组织、组内按 token 数降序排列的描述符列表。
        capture_descs = self.cudagraph_dispatcher.get_capture_descs()
        # decoder_graphs 是计划中 decoder 图的总数，不是实际采样的图数。
        decoder_graphs = sum(len(descs) for _, descs in capture_descs)
        # 当前没有 encoder Graph，因此总数直接取 decoder_graphs。
        total_graphs = decoder_graphs
        # 没有任何图时，沿用上游在切换 pool 和进入捕获 guard 前释放临时 KV。
        if total_graphs == 0:
            logger.debug("No CUDA graphs will be captured, skipping profiling")
            self._cleanup_profiling_kv_cache()
            return 0

        # graph_groups 保存各模式图数和最大容量，供本轮 profiling 日志展示。
        graph_groups = [
            *(
                f"{mode.name}={len(descs)} (largest={descs[0].num_tokens})"
                for mode, descs in capture_descs
                if descs
            ),
        ]
        logger.info("Profiling CUDA graph memory: %s", ", ".join(graph_groups))

        # profiling_pool 是临时独立 pool，避免采样分配影响正式共享 pool 的布局。
        profiling_pool = current_platform.graph_pool_handle()
        # original_pools 按 Wrapper 对象身份保存原 pool，finally 按同一个 id 恢复。
        original_pools: dict[int, Any] = {}
        # all_wrappers 沿用上游全局注册表；当前没有 BreakableCUDAGraphWrapper。
        all_wrappers = list(CUDAGraphWrapper._all_instances)
        # instance 是当前切换 pool 的 Wrapper，既保留原句柄，也统一使用临时 pool。
        for instance in all_wrappers:
            original_pools[id(instance)] = instance.graph_pool
            instance.graph_pool = profiling_pool

        # shared_memory_estimate 保存各模式首次捕获开销，per_graph_estimate 保存余下图开销。
        shared_memory_estimate: dict[CUDAGraphMode, int] = {}
        per_graph_estimate: dict[CUDAGraphMode, int] = {}

        # 与上游一致，这个 guard 负责捕获阶段的清理，不吞掉捕获异常。
        try:
            # 仅在这段启动窗口允许 Wrapper 创建临时图。
            set_cudagraph_capturing_enabled(True)
            with self._freeze_gc(), graph_capture(device=self.device):
                # 同步并释放空闲缓存后读取基线，避免之前的异步工作干扰样本。
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                # mode 是执行模式，descs 是正式捕获计划中的全部形状。
                for mode, descs in capture_descs:
                    # profile_descs 只取最大的两档，避免在预算阶段捕获整个计划。
                    profile_descs = descs[:2]
                    # mem_samples 保存两次捕获前后的空闲显存差，单位为字节。
                    mem_samples: list[int] = []
                    # i 表示采样档位；desc 是本次预热与捕获使用的形状 key。
                    for i, desc in enumerate(profile_descs):
                        # mem_before 是预热和捕获前的空闲显存，不包含临时 KV 大小。
                        mem_before = torch.cuda.mem_get_info()[0]
                        # 最大 FULL 档模拟较长 KV；其他档使用普通 dummy 长度。
                        self._warmup_and_capture(
                            desc,
                            cudagraph_runtime_mode=mode,
                            profile_seq_lens=(
                                min(
                                    self.max_model_len,
                                    self.max_num_tokens // desc.num_tokens,
                                )
                                if mode == CUDAGraphMode.FULL and i == 0
                                else None
                            ),
                        )
                        # capture 异步工作完成后才能采样本档新增占用。
                        torch.cuda.synchronize()
                        free_after = torch.cuda.mem_get_info()[0]
                        mem_samples.append(mem_before - free_after)
                    # first_capture 直接使用首个显存差样本，保持上游的估算口径。
                    first_capture = mem_samples[0]
                    # 第二档近似每图增量；只有一档时剩余图数为零。
                    per_graph = max(
                        mem_samples[1] if len(mem_samples) > 1 else 0,
                        1 << 20,
                    )
                    shared_memory_estimate[mode] = first_capture
                    per_graph_estimate[mode] = per_graph * (len(descs) - 1)
                    logger.debug(
                        "Estimated %s CUDA graph memory: "
                        "%.2f MiB first-capture + (%d-1) × %.2f MiB per-graph",
                        mode.name,
                        first_capture / (1 << 20),
                        len(descs),
                        per_graph / (1 << 20),
                    )
        finally:
            # 失败也关闭新增捕获；临时图属于 profiling，不能留给正式推理回放。
            set_cudagraph_capturing_enabled(False)
            CUDAGraphWrapper.clear_all_graphs()
            # 重新枚举注册表，只恢复采样前保存过原 pool 的 Wrapper。
            all_wrappers = list(CUDAGraphWrapper._all_instances)
            for instance in all_wrappers:
                if id(instance) in original_pools:
                    instance.graph_pool = original_pools[id(instance)]
            # 临时 keys 不能直接用于正式布局；initialize_kv_cache() 会重新生成它们。
            for key_set in self.cudagraph_dispatcher.cudagraph_keys.values():
                key_set.clear()
            self.cudagraph_dispatcher.keys_initialized = False
            self._cleanup_profiling_kv_cache()
        # decoder_estimate 将共享 pool 的首次占用取最大值，再累加其余图的独立开销。
        decoder_estimate = max(shared_memory_estimate.values(), default=0) + sum(
            per_graph_estimate.values()
        )
        # 没有独立 encoder pool，所以 total_estimate 直接取 decoder_estimate。
        total_estimate = decoder_estimate
        logger.info(
            "Estimated CUDA graph memory: %.2f GiB total",
            total_estimate / (1 << 30),
        )
        return int(total_estimate)

    def _init_minimal_kv_cache_for_profiling(self) -> None:
        """复用正式 KV 初始化流程，临时分配用于 Graph 显存采样的小型 cache。

        临时覆盖 num_gpu_blocks_override，并在配置生成后恢复用户值；初始化
        失败也不会遗留该覆盖。dummy 的页表只读取 block 0，slot_mapping 全为
        -1，不会分配或写入真实请求。与上游按最大捕获尺寸分配的思路一致，
        本项目额外计入保留空块；临时配置不回填 CacheConfig.num_gpu_blocks，
        该字段由 Worker 在正式 KV cache 初始化时记录最终规划结果。
        """
        # 延迟导入规划工具，保持 Runner 初始化与 Engine 配置规划的模块边界。
        from nanovllm.core.kv_cache_utils import (
            get_kv_cache_config_from_groups,
            get_kv_cache_groups,
        )

        # 读取模型构造期间注册的层，并合并当前唯一 FullAttention KV group。
        kv_cache_spec = self.get_kv_cache_spec()
        kv_cache_groups = get_kv_cache_groups(self.vllm_config, kv_cache_spec)
        # min_blocks 先沿用上游按最大捕获容量计算的临时 block 数。
        min_blocks = self.compilation_config.max_cudagraph_capture_size or 1
        # 本项目的 planner 要求一个 null block 和至少一个可分配 block，额外计入空块。
        min_blocks += 1
        # saved_override 保存用户配置，不能让 profiling 的小容量影响正式 KV 规划。
        saved_override = self.cache_config.num_gpu_blocks_override
        self.cache_config.num_gpu_blocks_override = min_blocks
        # 正常路径的初始化顺序与上游一致；finally 额外保证规划失败也恢复用户覆盖值。
        try:
            # available_memory=0 表示这里按显式覆盖生成计划，而不是用正式显存预算。
            minimal_config = get_kv_cache_config_from_groups(
                self.vllm_config, kv_cache_groups, available_memory=0,
            )
        finally:
            self.cache_config.num_gpu_blocks_override = saved_override
        # 同一初始化入口建立后端、Dispatcher、Builder 和实际 KV tensor 绑定。
        self.initialize_kv_cache(minimal_config, is_profiling=True)
        logger.debug("Initialized minimal KV cache for CUDA graph profiling")

    def _cleanup_profiling_kv_cache(self) -> None:
        """同步 GPU 并释放临时 KV tensor、层绑定、Builder 和缓存配置。

        调用方先清理临时 Graph，避免图仍保留 KV 地址；本方法不释放模型权重、
        持久输入 buffer 或 InputBatch，也不处理当前不存在的量化 scale cache。
        """
        # KV 地址不再被 GPU 使用后才能释放所有 view 对原始 storage 的引用。
        torch.cuda.synchronize()
        if hasattr(self, "kv_caches") and self.kv_caches:
            # i 是缓存列表下标；先逐项解除引用，再清空列表，沿用上游清理步骤。
            for i in range(len(self.kv_caches)):
                self.kv_caches[i] = None  # type: ignore[assignment]
            self.kv_caches.clear()
        # 清空 AttentionGroup 同时释放其中的 metadata Builder。
        if hasattr(self, "attn_groups"):
            self.attn_groups.clear()
        # 配置只在初始化后存在；部分初始化失败也允许进入此清理入口。
        if hasattr(self, "kv_cache_config"):
            delattr(self, "kv_cache_config")
        self.cache_config.num_gpu_blocks = None
        # profiling 不回填正式 num_gpu_blocks 镜像，此处也不覆盖该运行时统计字段。
        # Attention 自身也持有 KV view，仅清理 Runner 列表不能释放 storage。
        for layer in self.compilation_config.static_forward_context.values():
            if hasattr(layer, "kv_cache"):
                # kv_cache 是旧层绑定；空占位保留其 Tensor / list 表示方式。
                kv_cache = layer.kv_cache
                layer.kv_cache = (
                    torch.tensor([]) if isinstance(kv_cache, torch.Tensor) else []
                )
        # 回收不可达对象并归还临时 pool / KV 分配的空闲 allocator 缓存。
        gc.collect()
        torch.cuda.empty_cache()
        logger.debug("Cleaned up profiling KV cache and CUDA graphs")

    def capture_model(self) -> int:
        """在启动阶段按 Dispatcher 的形状计划预热并捕获模型 CUDA Graph。

        调用时模型与正式 KV cache 已加载，Attention 能力解析和 Dispatcher
        keys 初始化已完成，InputBatch 尚无活跃请求。当前 FA2 的计划仅包含
        FULL uniform decode，各模式内按 token 数降序捕获，使较小图有机会
        复用较大图建立的 Graph pool 空间。

        本方法组织 stream 环境、捕获开关和显存统计；_warmup_and_capture()
        通过 _dummy_run() 准备输入，实际 torch.cuda.graph() 在模型 Wrapper
        内部执行。图覆盖模型 forward，不包含词表投影与 sampler。

        Returns:
            cuda_graph_size: 捕获前后 GPU 空闲显存的差值，单位为字节；禁用 Graph 时返回 0。
                此值包含 Graph pool 与相关 CUDA 开销，不能视为图对象的精确大小。

        Side Effects:
            更新持久 dummy 输入与 metadata，向模型 Wrapper 缓存捕获条目，
            多次同步 GPU 并清理 allocator 的空闲缓存。退出捕获窗口后关闭
            全局新增捕获开关；已有图仍可回放。部分图捕获后发生异常时不回滚
            已缓存条目，异常继续传播。

        Raises:
            AssertionError: 尚未完成后端解析和 Dispatcher keys 初始化。
            RuntimeError: 捕获环境、dummy 执行或模型捕获失败。
        """
        # NONE 策略不需要捕获，也不创建捕获 stream 或读取 CUDA 显存。
        if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "ensure `cudagraph_mode` was not manually set to `NONE`"
            )
            return 0
        # 捕获计划依赖最终后端能力；未初始化的空 key 集合不能当成捕获已完成。
        assert self.cudagraph_dispatcher.keys_initialized, (
            "Initialize attention backends and cudagraph keys before capture_model"
        )
        # start_time 是整个启动捕获过程的墙钟起点，包含预热和同步的耗时。
        start_time = time.perf_counter()
        # 只有启动窗口允许 Wrapper 新建图；正常在线执行应只使用已捕获的条目。
        set_cudagraph_capturing_enabled(True)
        # finally 保证预热或捕获失败后也关闭开关，避免后续调用意外新增图。
        try:
            # _freeze_gc 处理 Python GC；graph_capture 准备共同使用的 CUDA stream，
            # 此处并不直接开始 CUDA Graph 的 kernel 记录。
            with self._freeze_gc(), graph_capture(device=self.device):
                # 等待此前 GPU 工作完成，再归还 allocator 空闲缓存，建立显存基线。
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                # start_free_gpu_memory 是捕获前空闲字节数，正式 KV cache 已包含在占用中。
                start_free_gpu_memory = torch.cuda.mem_get_info()[0]
                # runtime_mode 是具体执行模式，batch_descs 是该模式下的降序形状列表。
                for runtime_mode, batch_descs in self.cudagraph_dispatcher.get_capture_descs():
                    self._capture_cudagraphs(
                        batch_descriptors=batch_descs,
                        cudagraph_runtime_mode=runtime_mode,
                    )
                    # 一个模式的捕获结束后同步，保持共享 pool 的顺序执行约定。
                    torch.cuda.synchronize()
                # 所有捕获工作完成后再采样，避免把未完成的异步分配算作最终状态。
                torch.cuda.synchronize()
                # end_free_gpu_memory 是图及其内存池保留后剩余的空闲字节数。
                end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        finally:
            set_cudagraph_capturing_enabled(False)
        # 退出捕获 stream 环境后再次同步，并释放不再引用的 allocator 空闲缓存。
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # 上游 lock_workspace() 只锁 WorkspaceManager 的可增长缓冲区，
        # 不锁 PyTorch allocator、Graph pool 或 Runner 输入。当前 FA2/Triton
        # 路径没有使用此管理器，因此不移植一个没有实际使用者的锁接口。
        # end_time 是捕获与收尾结束的时刻，elapsed_time 的单位为秒。
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        # cuda_graph_size 表示本轮捕获的显存变化，不等于各图显存大小简单相加。
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # 输出启动耗时与显存变化；图共享 pool，所以按整轮差值汇总更直观。
        logger.info_once(
            "Graph capturing finished in %.0f secs, took %.2f GiB",
            elapsed_time,
            cuda_graph_size / (1 << 30),
        )
        return cuda_graph_size
    
    def _warmup_and_capture(
        self,
        desc: BatchDescriptor,
        cudagraph_runtime_mode: CUDAGraphMode,
        profile_seq_lens: int | None = None,
        num_warmups: int | None = None,
    ) -> None:
        """为一个形状先执行 eager 预热，再触发一次指定模式的图捕获。

        Args:
            desc: Dispatcher 提供的形状，num_tokens 为执行容量，uniform 指示
                dummy 是否按固定 decode query 长度构造请求。
            cudagraph_runtime_mode: 本次捕获的具体运行模式；当前使用 FULL。
            profile_seq_lens: 可选的模拟 KV 长度，透传给 dummy 的预热和捕获调用。
            num_warmups: 请求的预热次数；None 时读取配置。当前至少执行一次，
                因为尚无上游 kernel_warmup()，FA/Triton 首次初始化不能进入捕获。

        Side Effects:
            反复原地准备同一容量的输入。预热使用 NONE，最后一次调用采用
            指定模式，由模型 Wrapper 创建或回放对应 key 的图，不执行采样。
        """
        # 未指定次数时，沿用全局 CompilationConfig 的启动预热设置。
        if num_warmups is None:
            num_warmups = self.compilation_config.cudagraph_num_of_warmups
        # 负次数属于非法配置；零则表示没有额外预热，但仍需完成一次首次初始化。
        if num_warmups < 0:
            raise ValueError("num_warmups must be non-negative")
        num_warmups = max(1, num_warmups)
        # force_attention 在 eager 下也执行 FA，保证 FULL 图所需 kernel 已经预热。
        force_attention = cudagraph_runtime_mode == CUDAGraphMode.FULL
        # 每次预热都用 NONE，避免 Wrapper 提前捕获；FULL 的预热使用 capture
        # metadata，使 max_seq_len 等 Python 上界与正式捕获一致。
        for _ in range(num_warmups):
            self._dummy_run(
                desc.num_tokens,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                force_attention=force_attention,
                uniform_decode=desc.uniform,
                profile_seq_lens=profile_seq_lens,
                is_graph_capturing=force_attention,
            )
        # 这一次必须采用目标模式，不能继续传 NONE；is_graph_capturing 让
        # Builder 准备最大上下文上界，Wrapper 才会真正进入捕获分支。
        self._dummy_run(
            desc.num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            uniform_decode=desc.uniform,
            profile_seq_lens=profile_seq_lens,
            is_graph_capturing=True,
        )

    def _capture_cudagraphs(
        self,
        batch_descriptors: list[BatchDescriptor],
        cudagraph_runtime_mode: CUDAGraphMode,
    ) -> None:
        """按已有的降序描述符列表捕获同一模式的所有图，并显示启动进度。

        Args:
            batch_descriptors: Dispatcher 的捕获计划，当前 FULL decode 描述符
                按 num_tokens 从大到小排列；本方法保留调用方提供的顺序。
            cudagraph_runtime_mode: FULL 或 PIECEWISE 等具体模式，不能为 NONE
                或组合配置策略。当前配置解析器只会提供 FULL 计划。

        Side Effects:
            每个形状调用 _warmup_and_capture()，完成后同步 GPU；更新 Wrapper
            中的图缓存。tqdm 仅包装列表迭代，不修改描述符或 Dispatcher keys。
        """
        # 此入口负责实际捕获，NONE 或组合策略表示上游没有正确解析执行模式。
        assert (
            cudagraph_runtime_mode != CUDAGraphMode.NONE
            and cudagraph_runtime_mode.is_valid_runtime_mode()
        ), f"Invalid cudagraph runtime mode: {cudagraph_runtime_mode}"
        # 空计划不预热、不显示进度，也不触发 GPU 同步。
        if not batch_descriptors:
            return
        # uniform_decode 从该模式的首个描述符取得，仅用于进度条的批次类型展示。
        uniform_decode = batch_descriptors[0].uniform
        # 当前只运行单卡，无需上游的 rank 0 判断；配置控制是否显示进度条。
        batch_descriptors = tqdm(
            batch_descriptors,
            disable=not self.load_config.use_tqdm_on_load,
            desc="Capturing CUDA graphs ({}, {})".format(
                "decode" if uniform_decode else "mixed prefill-decode",
                cudagraph_runtime_mode.name,
            ),
        )
        # batch_desc 是本次要准备的完整形状 key，顺序沿用大图先捕获的计划。
        for batch_desc in batch_descriptors:
            self._warmup_and_capture(
                batch_desc,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
            )
            # 同一个 pool 的图顺序执行，避免上一档 GPU 工作未结束就进入下一档。
            torch.cuda.synchronize()

    @staticmethod
    @contextmanager
    def _freeze_gc() -> Iterator[None]:
        """在捕获前清理 Python 垃圾，并保留上游可选的 GC 冻结与恢复结构。

        当前尚无 VLLM_ENABLE_CUDAGRAPH_GC 环境配置，should_freeze 固定为 False，
        因而实际只执行进入前的 gc.collect()，捕获期间仍允许普通循环 GC。
        这里保留分支便于后续接入策略；gc.freeze() 处理 Python 的 GC 跟踪集合，
        与 CUDA Graph 的捕获开关及 GPU 内存池没有关系。

        Yields:
            空值；调用方在 with 作用域中预热和捕获模型。

        Side Effects:
            收集进入前不可达的 Python 对象。如果后续启用冻结策略，退出时
            无论是否发生异常都会 unfreeze 并再次 collect，不吞掉异常。
        """
        # 先释放不可达的 Python 对象，减少历史临时引用对捕获显存基线的影响。
        gc.collect()
        # should_freeze 是是否冻结现有 GC 跟踪对象的策略，当前保持 GC 开启。
        should_freeze = False
        # 可选冻结分支把已有对象移入永久集合，减少长时间预热中的 GC 扫描开销。
        if should_freeze:
            gc.freeze()
        try:
            yield
        finally:
            # 只有本次执行过 freeze 才恢复；捕获失败也执行相同的 GC 收尾。
            if should_freeze:
                gc.unfreeze()
                gc.collect()
    def shutdown(self) -> None:
        """释放模型与 KV 引用，并清理注册层和共享 RoPE 缓存。

        Engine 应先回收在途 Future 再关闭 Worker。本方法解除 Python 引用，
        不负责继续生成、向用户交付结果或等待各批输出事件。
        """
        # 清理 KV 绑定和临时 profiling 存储，避免模型层继续持有设备 tensor。
        self._cleanup_profiling_kv_cache()
        self.compilation_config.static_forward_context.clear()
        self.model = None  # type: ignore[assignment]
        get_rope.cache_clear()

    def _get_or_create_async_output_copy_stream(self) -> torch.cuda.Stream:
        """获取异步输出拷贝流，首次使用时创建并保存。

        Returns:
            stream: 本 Runner 复用的异步输出拷贝 stream，首次使用时创建。
        """
        stream = self.async_output_copy_stream
        if stream is None:
            # 同一个 Runner 复用该流，避免每轮采样重复创建。
            stream = torch.cuda.Stream()
            self.async_output_copy_stream = stream
        return stream
    def _update_states(
        self,
        scheduler_output: SchedulerOutput,
    ) -> Callable | None:
        """更新 Worker 请求镜像和持久 InputBatch。

        沿用官方实现的处理顺序：清理完成请求，移出未调度请求，
        建立新请求镜像，更新运行中或恢复请求，最后重新入批并刷新 metadata。

        连续执行的异步请求通过 GPU 采样结果准备下一轮输入；重新入批的
        异步请求则通过 Scheduler 提供的完整 token 快照恢复 CPU 输入。

        Args:
            scheduler_output: 本轮调度快照，包含请求进度、block table 更新、
                新请求、恢复请求和完成通知。

        Returns:
            post_forward_callback: 当前不支持投机解码，不需要 forward 后的延迟修正回调，返回 None。

        Side Effects:
            修改 requests、InputBatch 请求槽位、CPU token 行、block table
            和采样 metadata，并提交相应 metadata 的设备拷贝。
        """
        # ------------------------------------------------------------------
        # 第一阶段：先完成全部删除。
        #
        # InputBatch.add_request() 可能调用 BatchUpdateBuilder.pop_removed()
        # 复用空槽位；一旦 pop 触发 removed 列表排序，本轮就不能再登记删除。
        # 因此必须保证所有 remove_request() 都发生在任意 add_request() 之前。
        # ------------------------------------------------------------------

        # 先删除已完成请求的 Worker 镜像。
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)

        # 再从持久 batch 中移除完成请求。
        # 若同一 ID 同时出现在完成集合和新请求中，后续会重新建立镜像。
        for req_id in scheduler_output.finished_req_ids:
            self.input_batch.remove_request(req_id)

        # 本轮真正要进入模型 forward 的 request 集合。
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        # resumed request 的 block table 已在抢占时失效。即使它恰好仍留在旧 batch 中，
        # 也必须先移除，之后通过完整 block table 作为“新 batch 成员”重新加入。
        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
        # InputBatch 只保存本轮实际执行的请求。未调度请求可能只是暂时没有 token budget，
        # 也可能已被抢占；两种情况下都保留 self.requests 中的 Worker mirror。
        #
        # ``scheduled_req_ids - resumed_req_ids`` 的写法与 vLLM 一致：恢复请求必须先
        # 被移出旧槽位，随后走 add_request() 的完整重建路径。
        unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)

        for req_id in unscheduled_req_ids:
            self.input_batch.remove_request(req_id)
        # 收集稍后需要写入 InputBatch 的请求。必须等所有删除完成后再统一 add。
        reqs_to_add: list[CachedRequestState] = []

        # ------------------------------------------------------------------
        # 第二阶段：建立首次下发的新请求镜像。
        # ------------------------------------------------------------------
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            # 当前 MVP 没有 vLLM streaming session 的“同 req_id 重置请求”语义。
            # 若需要支持该 feature，应单独实现 _update_streaming_request()。
            if req_id in self.requests:
                raise ValueError(
                    "Received a scheduled new request that already exists in the "
                    f"Worker request cache: req_id={req_id!r}."
                )
            # NewRequestData 已经携带完整 prompt、完整 block table 和 prefix-cache
            # 命中的已计算长度，因此 Worker 能直接建立可执行镜像。
            req_state = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                sampling_params=new_req_data.sampling_params,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
            )
            # Worker mirror 从此保留到 request finished；即使中途被抢占，也不能删除。
            self.requests[req_id] = req_state

            # 新请求目前不可能已位于 InputBatch，稍后统一写入空槽位。
            reqs_to_add.append(req_state)
        # ------------------------------------------------------------------
        # 第三阶段：同步已有 running / resumed 请求。
        # ------------------------------------------------------------------
        req_data = scheduler_output.scheduled_cached_reqs

        # 当前 MVP 没有 PP；该字段保留在 SchedulerOutput 中只是为了未来接口兼容。
        assert not req_data.new_token_ids

        # CachedRequestData 的各列严格按 req_ids 下标对齐。
        for req_idx, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            # 本轮 forward 前 Scheduler 看到的已计算 token 数。
            num_computed_tokens = req_data.num_computed_tokens[req_idx]
            # 普通 running request 仅携带新增 block；恢复请求则携带完整 block table。
            new_block_ids = req_data.new_block_ids[req_idx]
            # 该 request 是否要从 PREEMPTED 状态恢复。
            resumed_from_preemption = req_id in req_data.resumed_req_ids
            # 本轮执行前的逻辑输出长度；异步时包含尚未回收的输出占位。
            num_output_tokens = req_data.num_output_tokens[req_idx]
            # 若请求尚在持久 batch 中，取得其当前物理槽位；被抢占或上轮未调度的
            # 请求此前已经从 InputBatch 移除，因此这里会得到 None。
            req_index = self.input_batch.req_id_to_index.get(req_id)
            
            # 更新缓存状态
            req_state.num_computed_tokens = num_computed_tokens
            # 对齐 Scheduler 已经丢弃的输出后缀；原地截断以保留共享列表引用。
            if num_output_tokens < len(req_state.output_token_ids):
                del req_state.output_token_ids[num_output_tokens:]

                # 当前总长度由 prompt 长度和 output 列表长度推导，
                # 无需额外更新官方实现中的 num_tokens_no_spec 数组。
            
            # 更新 Worker mirror 中的逻辑 block table。
            if not resumed_from_preemption:
                # 普通 RUNNING 请求只收到本轮新分配的 block 后缀。
                if new_block_ids is not None:
                    req_state.block_ids.extend(new_block_ids)
            else:
                # 恢复请求曾被抢占，旧物理 block 已释放；必须以完整 table 覆盖。
                assert req_index is None
                assert new_block_ids is not None
                req_state.block_ids = new_block_ids

            if req_index is None:
                # 请求此前未在持久 batch 中，可能被暂停，也可能被抢占后恢复。
                if self.use_async_scheduling and num_output_tokens > 0:
                    # 旧 Worker 输出列表可能含有 -1，必须从真实历史快照恢复。
                    resumed_token_ids = req_data.all_token_ids[req_id]
                    req_state.output_token_ids = resumed_token_ids[
                        -num_output_tokens:
                    ]

                # 当前没有活跃槽位引用该列表，add_request() 会建立新的引用，
                # 同时重建完整 CPU token 行、block table 和采样参数。
                reqs_to_add.append(req_state)
                continue
            # 请求持续留在 InputBatch 中，只更新会随调度轮次变化的 metadata。
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                num_computed_tokens
            )
            # InputBatch 的 block table 是 Worker mirror block_ids 的 staging 副本；
            # 普通请求只需追加本轮新增的物理 block。
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(
                    new_block_ids,
                    req_index,
                )

        # ------------------------------------------------------------------
        # 第四阶段：在全部删除完成后，统一把本轮执行请求放入持久 batch。
        # ------------------------------------------------------------------
        for request in reqs_to_add:
            # add_request() 会写入 prompt/output token 行、完整 block table、temperature
            # 和 num_computed_tokens，并通过 BatchUpdateBuilder 记录新增槽位。
            self.input_batch.add_request(request)
        # 消除删除导致的槽位空洞。必须位于所有 add_request() 之后，使新请求优先复用
        # 最小空槽位，而不是无意义地搬运仍会被覆盖的旧行。
        self.input_batch.condense()
        # 将 temperature 和未来 logits processor 所需 BatchUpdate 同步到 GPU。
        # 即使本轮没有新增/删除，也必须调用：它负责消费并重置 Builder 的 step 边界。
        self.input_batch.refresh_metadata()
        # 当前无投机解码、Mamba hybrid state 或异步修正逻辑。
        return None

    def _prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[torch.Tensor]:
        """将持久 InputBatch 转换为本轮 packed forward 输入和 attention 上下文。

        该方法是 SchedulerOutput 与模型 forward 之间的关键适配层。它不重新创建
        request 状态，也不决定调度；它只读取已经由 ``_update_states()`` 对齐的
        InputBatch，将不同 request 的本轮 token 拼接为一个 packed token 轴。

        当前实现支持同一轮混合：

        - 新请求或 chunked prefill：一个 request 本轮执行多个 token；
        - 普通 decode：一个 request 本轮执行一个 token；
        - prefix-cache 命中：``num_computed_tokens > 0``；
        - 被抢占后恢复：已由 ``_update_states()`` 重建 block table。

        当前统一走 paged varlen attention。

        Args:
            scheduler_output: 当前 step 的不可变调度快照。
            num_scheduled_tokens: 与 ``input_batch.req_ids`` 对齐的一维 int32 数组；
                每项表示对应 request 本轮执行的 token 数。

        Returns:
            logits_indices: 每条 request 最后一个 query token 在 packed hidden-state tensor 中的索引。
                ``execute_model()`` 应使用它执行：

                ``sample_hidden_states = hidden_states[logits_indices]``。

        Raises:
            ValueError: SchedulerOutput、InputBatch 与传入 token 数数组不一致时抛出。

        Side Effects:
            更新 Runner 的 CPU/GPU packed 输入 buffer、positions、长度元数据、
            block-table slot mapping，准备后续 _build_attention_metadata() 所需的基础张量。
        """
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # 尽早启动 CPU -> GPU block-table 拷贝，使其能与后续 CPU 上的 gather 计算重叠。
        self.input_batch.block_table.commit_block_table(num_reqs)
        # 例如请求本轮长度 [2, 5, 3]，生成：
        # [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]。
        req_indices = np.repeat(
            self.arange_np[:num_reqs],
            num_scheduled_tokens,
        )
        # 计算每条 request 的 packed query 终点，以及每个 packed token 在 query 内的偏移。
        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # self.query_pos.np[:10]: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens = self._get_cumsum_and_arange(
            num_scheduled_tokens,
            self.query_pos.np,
        )
        # 一个 packed token 的逻辑位置等于：
        #
        # 已写入 KV 的 token 数 + 该 token 在本轮 query chunk 内的相对位置。
        positions_np = (
            self.input_batch.num_computed_tokens_cpu[req_indices]
            + self.query_pos.np[: cu_num_tokens[-1]]
        )
        # token_ids_cpu 是 [request_slot, max_model_len] 的二维矩阵。转换为一维索引后，
        # 一次 index_select 即可取出所有 request 的不连续 query token。
        token_indices = (
            positions_np
            + req_indices * self.input_batch.token_ids_cpu.shape[1]
        )
        token_indices_tensor = torch.from_numpy(token_indices)
        # 把本轮实际执行的 token 收集到固定容量的 packed CPU staging buffer。
        torch.index_select(
            self.input_batch.token_ids_cpu_tensor.flatten(),
            0,
            token_indices_tensor,
            out=self.input_ids.cpu[:total_num_scheduled_tokens],
        )
        # Query 的起始边界带一个起始零，末项等于本批真实 packed token 数。
        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
        # 空请求后缀重复最后一个终点，保证 FlashAttention 读取的前缀和单调不减。
        self.query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
        self.query_start_loc.copy_to_gpu()

        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        self._compute_prev_positions(num_reqs)

        self.num_computed_tokens[:num_reqs].copy_(
                self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
                non_blocking=True,
            )

        # 将每个 packed token 归属的 request 槽位上传，用于在 GPU 上批量计算位置。
        self.req_indices.np[:total_num_scheduled_tokens] = req_indices
        self.req_indices.copy_to_gpu(total_num_scheduled_tokens)
        req_indices_gpu = self.req_indices.gpu[:total_num_scheduled_tokens]
        # query_pos 是每个 token 在“本轮 query chunk”中的相对位置，类型保持 int64，
        # 以便与 RoPE positions 和 PyTorch embedding 索引保持兼容。
        self.query_pos.copy_to_gpu(total_num_scheduled_tokens)
        # 每条 request 本轮调度 token 数也保存到固定 GPU buffer；seq_lens 计算和未来
        # CUDA Graph 固定缓冲复用都会依赖它。
        self.num_scheduled_tokens.np[:num_reqs] = num_scheduled_tokens
        self.num_scheduled_tokens.copy_to_gpu(num_reqs)
        num_scheduled_tokens_gpu = self.num_scheduled_tokens.gpu[:num_reqs]
        # 按 request gather 已计算长度，再加 query 内位置，得到每个 packed token 的 RoPE
        # position 和 slot-mapping 逻辑 position。
        self.positions[:total_num_scheduled_tokens] = (
            self.num_computed_tokens[req_indices_gpu].to(torch.int64)
            + self.query_pos.gpu[:total_num_scheduled_tokens]
        )

        # seq_lens 是 forward 后完整 KV 长度；当前 varlen attention 用它建立 K/V 分段，
        # 未来 decode-only attention 路径也可将它作为 context_lens 使用。
        self.seq_lens[:num_reqs] = (
            self.num_computed_tokens[:num_reqs]
            + num_scheduled_tokens_gpu
        )

        # 未使用后缀置零，避免后续 CUDA Graph 或调试工具错误读取上轮残留长度。
        self.seq_lens[num_reqs:].fill_(0)
        
        # 根据“逻辑位置 + request block-table 行”生成每个 packed token 的物理 KV slot。
        # Attention.forward() 随后使用该 mapping 将本轮 K/V 写入正确的 cache page。
        self.input_batch.block_table.compute_slot_mapping(
            num_reqs=num_reqs,
            query_start_loc=self.query_start_loc.gpu[:num_reqs + 1],
            positions=self.positions[:total_num_scheduled_tokens],
        )
        # 把 self.input_ids 拷贝到GPU 
        self._prepare_input_ids(
            scheduler_output,
            num_reqs,
            total_num_scheduled_tokens,
            cu_num_tokens,
        )

        # 每段 query 的最后一个 token 都是本轮产生下一个 token logits 的位置。
        # chunked prefill 的中间 chunk 也会得到一个 logits，但 sample_tokens() 会依据
        # Scheduler 的 request 进度决定是否忽略它。
        return (self.query_start_loc.gpu[1:num_reqs + 1] - 1)

    def _build_attention_metadata(
        self,
        num_tokens: int,
        num_reqs: int,
        max_query_len: int,
        num_tokens_padded: int | None = None,
        num_reqs_padded: int | None = None,
        for_cudagraph_capture: bool = False,
        slot_mappings: dict[int, torch.Tensor] | None = None,
    ) -> tuple[dict[str, AttentionMetadata], CommonAttentionMetadata | None]:
        """把本轮 packed 输入转换为各 Attention 层的后端元数据。

        _prepare_inputs() 已填好长度 buffer 与 BlockTable；_get_slot_mappings()
        已提供每个 KV group 的物理写入位置。真实请求在有效前缀，padding 请求
        使用空 query、零序列长度与 NULL_BLOCK_ID，补齐 token 的 slot 为 -1。

        Args:
            num_tokens: 本轮真实执行的 packed token 数。
            num_reqs: 本轮有效请求数；启动阶段也可以是 dummy 请求数。
            max_query_len: 单个请求本轮最多执行的 token 数。
            num_tokens_padded: 模型执行的 token 容量；None 时使用真实数量。
            num_reqs_padded: 后端执行的请求容量；None 时使用真实数量。
            for_cudagraph_capture: 为 True 时使用捕获 Builder 入口，并把
                max_seq_len 固定为 max_model_len，覆盖后续增长的 KV 上下文。
            slot_mappings: KV group 编号到 packed token 物理 slot 的映射。
                None 时从持久 BlockTable 构造；映射必须覆盖补齐后的 token 容量。

        Returns:
            (attn_metadata, common_attn_metadata):
                attn_metadata: 按完整 Attention 层名索引的后端元数据，零 token 时为空字典。
                common_attn_metadata: 供投机解码使用的公共元数据；当前不支持该功能，固定为 None。

        Side Effects:
            原地清理固定输入及 metadata 的 padding 后缀，不替换 GPU storage。
            有真实 InputBatch 时更新持久 CPU 序列长度；空 batch 的 dummy 路径
            直接使用调用方已准备的 optimistic_seq_lens_cpu。dummy 的有效
            query/seq_lens/block table 应由调用方预先准备。
        """
        if not self.kv_cache_config.kv_cache_groups:
            return {}, None
        assert len(self.kv_cache_config.kv_cache_groups) == 1
        # 启动时 dummy 不创建真实请求；只有在线 batch 才校验请求数量一致。
        if not for_cudagraph_capture and self.input_batch.num_reqs:
            assert num_reqs == self.input_batch.num_reqs
        num_tokens_padded = (
            num_tokens if num_tokens_padded is None else num_tokens_padded
        )
        num_reqs_padded = num_reqs if num_reqs_padded is None else num_reqs_padded
        assert 0 < num_tokens <= num_tokens_padded <= self.max_num_tokens
        assert 0 < num_reqs <= num_reqs_padded <= self.max_num_reqs
        # 有效边界由 prepare/dummy 路径生成，不能把未准备的长度误当成 padding。
        assert int(self.query_start_loc.np[num_reqs]) == num_tokens

        # 保持 query 边界非递减：padding 请求贡献零个 query。补齐的模型 token
        # 行不必加入有效 query 前缀和，随后只对真实请求的 hidden states 采样。
        self.query_start_loc.cpu[num_reqs + 1:].fill_(num_tokens)
        self.query_start_loc.copy_to_gpu(num_reqs_padded + 1)
        self.seq_lens[num_reqs:].zero_()
        self.num_computed_tokens[num_reqs:].zero_()
        self.optimistic_seq_lens_cpu[num_reqs:].zero_()
        self.num_scheduled_tokens.cpu[num_reqs:].zero_()
        self.num_scheduled_tokens.gpu[num_reqs:].zero_()
        self.input_batch.num_computed_tokens_cpu_tensor[num_reqs:].zero_()
        self.input_batch.num_prompt_tokens_cpu_tensor[num_reqs:].zero_()
        # 模型 embedding/RoPE 会执行容量对应的全部行，避免读取上轮 token/position。
        self.input_ids.cpu[num_tokens:num_tokens_padded].zero_()
        self.input_ids.gpu[num_tokens:num_tokens_padded].zero_()
        self.positions[num_tokens:num_tokens_padded].zero_()

        if for_cudagraph_capture:
            # Python 标量在 replay 时不会重算，不能用短 dummy 的 KV 长度选择 kernel。
            max_seq_len = self.max_model_len
        else:
            # 在线请求从调度进度计算长度；启动 dummy 没有真实进度，沿用其
            # 提前准备的 CPU 长度，尤其不能覆盖 profile_seq_lens 指定的上下文。
            if self.input_batch.num_reqs:
                np.add(
                    self.input_batch.num_computed_tokens_cpu[:num_reqs],
                    self.num_scheduled_tokens.np[:num_reqs],
                    out=self.optimistic_seq_lens_cpu.numpy()[:num_reqs],
                )
            max_seq_len = int(self.optimistic_seq_lens_cpu[:num_reqs].max())

        block_table = self.input_batch.block_table
        block_table.block_table.cpu[num_reqs:].fill_(NULL_BLOCK_ID)
        block_table.num_blocks_per_row[num_reqs:] = 0
        block_table_tensor = block_table.get_device_tensor(num_reqs_padded)
        block_table_tensor[num_reqs:num_reqs_padded].fill_(NULL_BLOCK_ID)

        if slot_mappings is None:
            slot_mappings, _ = self._get_slot_mappings(
                num_tokens_padded, num_reqs_padded, num_tokens,
            )
        assert slot_mappings is not None
        assert slot_mappings[0].numel() >= num_tokens_padded
        slot_mapping = slot_mappings[0][:num_tokens_padded]
        slot_mapping[num_tokens:num_tokens_padded].fill_(-1)
        common_metadata = CommonAttentionMetadata(
            query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],
            query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],
            seq_lens=self.seq_lens[:num_reqs_padded],
            num_reqs=num_reqs_padded,
            num_actual_tokens=num_tokens_padded,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            block_table_tensor=block_table_tensor,
            slot_mapping=slot_mapping,
        )

        # 同一 AttentionGroup 内的层复用 Builder 产出的同一个 metadata 对象。
        attn_metadata: dict[str, AttentionMetadata] = {}
        for group in self.attn_groups[0]:
            builder = group.get_metadata_builder()
            if for_cudagraph_capture:
                group_metadata = builder.build_for_cudagraph_capture(common_metadata)
            else:
                group_metadata = builder.build(
                    common_prefix_len=0, common_attn_metadata=common_metadata,
                )
            for layer_name in group.layer_names:
                attn_metadata[layer_name] = group_metadata
        return attn_metadata, None

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        """同步调度状态、执行 packed forward，并暂存下一步采样需要的 logits。

        prefill、decode 和 chunked prefill 共用同步 packed token 输入。
        当前 FA2 策略下，Dispatcher 为满足 uniform 单 token decode 条件、
        且匹配合法 Graph key 的 batch 选择 FULL；模型 Wrapper 按补齐后的
        固定形状回放启动时捕获的图。Graph 禁用、非 uniform batch 或没有
        匹配的合法 key 时，使用 NONE 模式执行 eager forward。两条路径都
        只选取真实请求最后一个 query 的 hidden state 来计算采样 logits。

        Args:
            scheduler_output: 本轮调度快照，包含请求状态更新、KV block 变化，
                以及每个请求本轮需要执行的 token 数。

        Returns:
            model_output: 没有调度 token 时，返回空 ModelRunnerOutput。
                成功执行 forward 时，返回 None，调用方随后必须调用 sample_tokens()。

        Raises:
            RuntimeError: 上一轮 logits 尚未消费，或执行时尚未初始化 KV cache。
            AssertionError: 持久 batch 与调度结果违反内部执行协议。

        Side Effects:
            更新 Worker 请求镜像、InputBatch 和模型输入缓冲区；模型 forward
            写入各层 KV cache；成功后设置 execute_model_state。
        """
        # 上一轮 logits 必须先完成采样，才能更新 batch 或复用输入缓冲区。
        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with self.synchronize_input_prep():
            self._update_states(scheduler_output)
            if not num_scheduled_tokens:
                return EMPTY_MODEL_RUNNER_OUTPUT
            # 当前 InputBatch 只维护一张 BlockTable，因此只接受一个 KV group。
            assert len(self.kv_cache_config.kv_cache_groups) == 1, (
                "execute_model() requires exactly one KV cache group."
            )
            # _update_states() 已经完成删除、添加和 condense；这里读取稳定槽位顺序。
            num_reqs = self.input_batch.num_reqs
            req_ids = self.input_batch.req_ids
            # 必须使用 InputBatch 的槽位顺序，不能直接使用调度字典的 values()。
            # token 行、block table、temperature 和 logits 都按这个顺序对齐。
            tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
            num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
            # Attention kernel 需要单条请求的最大 query 长度；直接从 CPU 数组计算。
            max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())
            
            num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens
            # 整理 packed token IDs、positions、长度和 block-table slot mapping。
            # logits_indices 是每条请求最后一个 query token 在 packed hidden states 中的位置。
            logits_indices = self._prepare_inputs(
                scheduler_output,
                num_scheduled_tokens_np,
            )
            (
                cudagraph_mode,
                batch_desc,
                _,
                _,
                _,
            ) = self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                max_num_scheduled_tokens=max_num_scheduled_tokens,
            )
            num_tokens_padded = batch_desc.num_tokens
            num_reqs_padded = (
                batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
            )
            pad_attn = cudagraph_mode == CUDAGraphMode.FULL
            # group 映射用于构造 metadata；layer 映射用于 forward context。
            slot_mappings_by_group, slot_mappings_by_layer = (
                self._get_slot_mappings(
                    num_tokens_padded=num_tokens_padded
                    if pad_attn
                    else num_tokens_unpadded,
                    num_reqs_padded=(
                        num_reqs_padded if pad_attn else num_reqs
                    ),
                    num_tokens_unpadded=num_tokens_unpadded,
                    ubatch_slices=None,
                )
            )
            # 当前不使用投机解码，所以只消费按 Attention 层名组织的 metadata。
            attn_metadata, _ = self._build_attention_metadata(
                num_tokens=num_tokens_unpadded,
                num_tokens_padded=num_tokens_padded if pad_attn else None,
                num_reqs=num_reqs,
                num_reqs_padded=num_reqs_padded if pad_attn else None,
                max_query_len=max_num_scheduled_tokens,
                slot_mappings=slot_mappings_by_group,
            )
            # 固定缓冲区只取本轮有效前缀，避免把上轮残留 token 送入模型。
            input_ids = self.input_ids.gpu[:num_tokens_padded]
            positions = self.positions[:num_tokens_padded]

        # Attention 层通过运行时上下文取得本轮 metadata 和 KV 写入位置。
        # 上下文必须在正常退出和异常退出时都恢复，避免污染下一轮执行。
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens_padded,
            cudagraph_runtime_mode=cudagraph_mode,
            batch_descriptor=batch_desc,
            slot_mapping=slot_mappings_by_layer,
        ):
            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
            )
        hidden_states=model_output
        sample_hidden_states = hidden_states[logits_indices]
        logits = self.model.compute_logits(sample_hidden_states)
        # 当前采样只需要调度进度和 logits，不需要保留整批 hidden states。
        self.execute_model_state = ExecuteModelState(
            scheduler_output=scheduler_output,
            logits=logits,
        )
        # 保持 vLLM V1 的 forward / sampling 分离协议。
        return None
    @torch.inference_mode()
    def sample_tokens(self) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        """消费最近一次 forward 的 logits，执行采样并交付结果。

        执行顺序与 vLLM 保持一致：消费临时执行状态，执行采样，
        清除上一轮 GPU token 引用，完成 bookkeeping，最后构造输出。

        Returns:
            根据调用方式返回以下分支之一：
                output: 同步模式下已填入 CPU token 列表的 ModelRunnerOutput。
                async_output: 异步模式下的 AsyncGPUModelRunnerOutput，调用 get_output() 后取得实际结果。

        Raises:
            RuntimeError: 没有待消费的 forward 结果时抛出。

        Side Effects:
            清空 execute_model_state，更新 Worker 输出列表和 CPU token 行。
            异步模式追加占位 token，并保存本轮 GPU 结果供下一轮输入准备使用。
        """
        if self.execute_model_state is None:
            raise RuntimeError(
                "sample_tokens() requires a pending execute_model() result."
            )
        # 当前临时状态只保留 scheduler_output 和 logits，未支持的字段已裁剪。
        scheduler_output, logits = self.execute_model_state
        # 与 vLLM 一致，先消费短暂状态，防止同一份 logits 被重复采样。
        self.execute_model_state = None
        # Sampler 只生成设备端候选，不在这里转换成 CPU token 列表。
        sampler_output = self._sample(logits)

        # 上一轮 GPU token 已在本轮输入准备中使用完毕。
        # bookkeeping 随后会保存当前轮结果，供下一轮输入准备使用。
        self.input_batch.prev_sampled_token_ids = None

        # 同步模式回写真实 token；异步模式回写占位并保存 GPU 结果。
        (
            valid_sampled_token_ids,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = self._bookkeeping_sync(scheduler_output, sampler_output)

        # 与 vLLM 一样，在采样与状态回写完成后统一构造 ModelRunnerOutput。
        output = ModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
        )
        # 同步模式的 output 已包含 bookkeeping 取得的 CPU 结果，直接返回即可。
        if not self.use_async_scheduling:
            return output
        # 异步包装器在独立拷贝流上提交 D2H，保存完成事件和设备 tensor 引用。
        async_output = AsyncGPUModelRunnerOutput(
            model_runner_output=output,
            sampled_token_ids=sampler_output.sampled_token_ids,
            invalid_req_indices=invalid_req_indices,
            async_output_copy_stream=self._get_or_create_async_output_copy_stream(),
            vocab_size=self.input_batch.vocab_size,
        )
        return async_output

    def _sample(self, logits: torch.Tensor) -> SamplerOutput:
        """使用与 logits 行顺序一致的 InputBatch metadata 调用 Sampler。

        Returns:
            sampler_output: 采样得到的 SamplerOutput，使用与 logits 行顺序一致的 metadata。
        """
        return self.sampler(logits, self.input_batch.sampling_metadata)

    def _bookkeeping_sync(
        self,
        scheduler_output: SchedulerOutput,
        sampler_output: SamplerOutput,
    ) -> tuple[
        list[list[int]],
        list[str],
        dict[str, int],
        list[int],
    ]:
        """过滤无效采样行，并更新 Worker 的输出状态。

        同步模式取得真实 CPU token；异步模式保存 GPU token 和请求映射，
        并通过 -1 占位维护 Worker 的逻辑输出长度。

        Args:
            scheduler_output: 本轮 forward 对应的调度快照。
            sampler_output: 按当前请求槽位排列的设备端采样结果。

        Returns:
            按返回顺序包含以下各项：
                valid_sampled_token_ids: 有效 CPU token 行列表；异步模式尚未回收，整体为空列表。
                req_ids_output_copy: 本批请求 ID 顺序的快照。
                req_id_to_index_output_copy: 请求 ID 到本批输出行的映射快照。
                invalid_req_indices: 异步输出回收时需要过滤的无效行下标列表。

        Side Effects:
            更新 CPU token 行和 CachedRequestState.output_token_ids。
            异步模式额外保存本轮 GPU 采样结果及其有效请求映射。
        """
        num_reqs = self.input_batch.num_reqs
        req_ids = self.input_batch.req_ids  # 所有长度和候选行均使用稳定槽位顺序。
        # 当前 _prepare_inputs() 尚未保存 discard_request_mask；这里使用 CPU
        # 进度求同样的丢弃下标，避免为判断 chunk 完成情况读取 GPU tensor。
        seq_lens_cpu = self.input_batch.num_computed_tokens_cpu[:num_reqs] + np.array(
            [scheduler_output.num_scheduled_tokens[req_id] for req_id in req_ids],
            dtype=np.int32,
        )
        num_tokens_np = np.array(
            [self.requests[req_id].num_tokens for req_id in req_ids],
            dtype=np.int32,
        )
        # 结束位置尚未覆盖完整 prompt + 已有 output 时，prefill/恢复重算仍未完成。
        discard_sampled_tokens_req_indices = np.nonzero(
            seq_lens_cpu < num_tokens_np
        )[0]

        # 与 vLLM 一样复制两个容器，后续 batch 删除和压缩不会影响本轮输出映射。
        req_ids_output_copy = req_ids.copy()
        req_id_to_index_output_copy = self.input_batch.req_id_to_index.copy()

        num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
        sampled_token_ids = sampler_output.sampled_token_ids
        # 当前没有投机解码，每个请求必须恰好有一个设备端候选。
        assert sampled_token_ids.shape == (num_reqs, 1)
        invalid_req_indices: list[int] = []
        if not self.use_async_scheduling:
            # 同步模式在这里等待设备结果，并过滤未完成 prefill 的候选。
            valid_sampled_token_ids = sampled_token_ids.tolist()
            for i in discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[int(i)].clear()
        else:
            # CPU 结果由异步输出包装器延后提供，此处只读取 CPU 下标数组。
            valid_sampled_token_ids = []
            invalid_req_indices = discard_sampled_tokens_req_indices.tolist()
            invalid_req_indices_set = set(invalid_req_indices)

            # 当前没有投机解码，本轮候选直接作为下一轮 GPU 回填来源。
            self.input_batch.prev_sampled_token_ids = sampled_token_ids
            # 无效行不参与下一轮 GPU 回填，避免使用未完成 prefill 的候选。
            self.input_batch.prev_req_id_to_index = {
                req_id: i
                for i, req_id in enumerate(req_ids)
                if i not in invalid_req_indices_set
            }

        # 将有效 token 缓存在 Worker 中，Scheduler 下一轮只需同步输出数量。
        for req_idx in range(num_sampled_tokens):
            if self.use_async_scheduling:
                sampled_ids = (
                    [-1] if req_idx not in invalid_req_indices_set else []
                )
            else:
                sampled_ids = valid_sampled_token_ids[req_idx]

            # 无效采样不增加输出长度，也不在 CPU 输入行中追加占位。
            if not sampled_ids:
                continue

            req_id = req_ids[req_idx]
            req_state = self.requests[req_id]
            # 无投机解码时，镜像的 num_tokens 等价于 vLLM 的 num_tokens_no_spec。
            start_idx = req_state.num_tokens  # 在完整 prompt + output 序列末尾追加。
            end_idx = start_idx + len(sampled_ids)
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length: "
                f"req_id={req_id!r}, num_tokens={end_idx}, "
                f"max_model_len={self.max_model_len}."
            )
            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            req_state.output_token_ids.extend(sampled_ids)  # 槽位输出列表引用同一对象。

        return (
            valid_sampled_token_ids,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        )

    
    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        is_graph_capturing: bool = False,
        profile_seq_lens: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """用固定 buffer 执行模拟 batch，供 profiling、预热和 Graph 捕获复用。

        沿用 vLLM 的顺序：构造请求级    调度量，分派模式和补齐容量，准备
        Attention 输入，安装 ForwardContext，执行模型，最后提取每个请求的
        最后一个 query hidden state。当前支持单卡、文本 Qwen3、单 KV group
        和单 token decode，裁剪 LoRA、投机解码、DP/PP、microbatch 与分段编译。

        此方法用于启动阶段，要求 InputBatch 没有活跃请求。dummy 不申请真实
        KV blocks，页表全部指向保留的空块 0，所有 slot_mapping 都设为 -1，
        因而不会写入模型的 KV cache。返回值可以供后续 sampler profiling 使用。

        Args:
            num_tokens: 有效 packed token 数，范围为 [1, max_num_tokens]。
            cudagraph_runtime_mode: None 表示采用 Dispatcher 的结果；NONE 用于
                eager 预热或 profiling；FULL 使用完整图。显式模式必须与分派
                结果一致，不能传 FULL_DECODE_ONLY 等配置策略。
            force_attention: 即使使用 NONE 也构造 metadata、执行真实 FA kernel；
                要求 KV cache 配置和 metadata Builder 已初始化。
            uniform_decode: 每个 dummy 请求调度 uniform_decode_query_len 个
                token；当前长度固定为 1，因此 num_tokens 不能超过请求容量。
            is_profile: 强制使用 NONE；缺省不构造 metadata，FA 使用 profiling
                占位路径，此时无需已经分配正式 KV cache。
            create_mixed_batch: 构造若干单 token decode 和一个 prefill 请求，
                与 uniform_decode 互斥。
            is_graph_capturing: 使用 Builder 的 capture 接口及最大上下文上界。
                FULL 也始终使用该接口，保护 Wrapper 第一次调用的自动捕获。
            profile_seq_lens: 指定所有有效 dummy 请求的 KV 长度，用于预热或
                测量长上下文 workspace；只有构造 attention metadata 时生效。

        Returns:
            (hidden_states, last_hidden_states):
                hidden_states: dummy 批次包含 padding 在内的全部 hidden-state 行。
                last_hidden_states: 每个有效 dummy 请求最后一个 token 的 hidden states，
                    形状为 [num_reqs, hidden_size]；本方法不计算 logits 或采样。

        Side Effects:
            原地更新 Runner 的输入和 metadata 工作区，保持 GPU storage 地址；
            FULL 通过模型 Wrapper 捕获或回放图。CUDA 输入准备完成后同步当前
            stream，保证重复 dummy 调用不会覆盖仍被 H2D DMA 读取的 CPU buffer。
            不修改 requests、InputBatch 请求映射、输出列表或 execute_model_state。

        Raises:
            AssertionError: token 容量、batch 布局或显式模式与分派结果不一致。
            RuntimeError: InputBatch 仍有活跃请求，或真实 Attention 尚未初始化。
            NotImplementedError: 请求当前未支持的 PIECEWISE 路径。
            ValueError: 指定的 KV 长度无法覆盖有效 query，或超过页表容量。
        """
        # runtime mode 只描述本次执行，组合策略必须先交给 Dispatcher 解析。
        assert (
            cudagraph_runtime_mode is None
            or cudagraph_runtime_mode.is_valid_runtime_mode()
        )
        # 目前没有分段编译器，不能将 PIECEWISE 当成可执行的预热路径。
        if cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
            raise NotImplementedError("PIECEWISE dummy execution is not implemented")
        # 启动工作区与在线输入共用 storage；有活跃请求时不能用 dummy 覆盖它们。
        if self.input_batch.num_reqs:
            raise RuntimeError("Dummy execution requires an empty InputBatch")
        # 排除空输入和超出持久 token buffer 容量的调用。
        assert 0 < num_tokens <= self.max_num_tokens
        # profiling 必须 eager；真实捕获不能同时被声明为显存 profiling。
        assert not (is_profile and is_graph_capturing)

        # max_query_len 是 kernel 的单请求 query 长度上界；普通布局沿用上游
        # num_tokens 的保守上界，uniform decode 则使用固定的单请求 query 长度。
        max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens
        # max_num_reqs 是固定请求槽位容量，限制 dummy 的请求拆分数量。
        max_num_reqs = self.scheduler_config.max_num_seqs
        # 混合布局把前一部分 token 分给 decode，剩余 token 组成一个 prefill。
        if create_mixed_batch:
            assert not uniform_decode
            # num_decode_tokens 同时也是 decode 请求数，必须留一个槽位给 prefill。
            num_decode_tokens = min(max_num_reqs - 1, num_tokens // 2)
            # num_prefill_tokens 是最后一个请求的 query 长度，至少保留一个 token。
            num_prefill_tokens = num_tokens - num_decode_tokens
            # num_reqs 表示本次模拟的有效请求数，不包含 Graph 的补齐槽位。
            num_reqs = num_decode_tokens + 1
            # num_scheduled_tokens_list 按槽位排列，示例 6 token、4 槽位为 [1,1,1,3]。
            num_scheduled_tokens_list = [1] * num_decode_tokens + [num_prefill_tokens]
            # 混合布局的最大 query 来自最后的 prefill 请求。
            max_query_len = num_prefill_tokens
        # uniform 布局每个请求固定一个 query，供 FULL decode 捕获和 eager 预热。
        elif uniform_decode:
            # 不能通过截断请求数量来吞掉 token；当前单 token decode 也无残缺请求。
            assert num_tokens <= max_num_reqs * self.uniform_decode_query_len
            # cdiv 保留上游的请求数推导；当前 uniform_decode_query_len 固定为 1。
            num_reqs = min(max_num_reqs, cdiv(num_tokens, max_query_len))
            # 每个有效请求具有相同的 query 长度，不把 padding 混入此列表。
            num_scheduled_tokens_list = [max_query_len] * num_reqs
        # 普通布局尽量占满请求槽位，平均分配 token，余数交给最后一个请求。
        else:
            # 请求数不能超过 token 数，否则会产生没有 query 的有效请求。
            num_reqs = min(num_tokens, max_num_reqs)
            # min_tokens_per_req 是每个请求至少获得的 token 数。
            min_tokens_per_req = num_tokens // num_reqs
            # 示例 7 token、4 槽位先得到 [1,1,1,1]，再将余数补成 [1,1,1,4]。
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs

        # 这两项不变量保证请求级布局完整覆盖输入，且与有效槽位数一致。
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        # num_scheduled_tokens 是 Builder/Dispatcher 使用的 int32 CPU 调度数组。
        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
        # num_tokens_unpadded 保存有效 token 总数，后续与模型执行容量分开使用。
        num_tokens_unpadded = int(num_scheduled_tokens.sum())

        # _cudagraph_mode 是实际可用模式，batch_desc 则是对应的固定形状 key。
        # 当前丢弃 should_ubatch、num_tokens_across_dp 和 cudagraph_stats 三项返回值。
        _cudagraph_mode, batch_desc, _, _, _ = (
            self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens,
                max_num_scheduled_tokens=max_query_len,
                use_cascade_attn=False,
                force_eager=is_profile or cudagraph_runtime_mode == CUDAGraphMode.NONE,
                force_uniform_decode=uniform_decode,
            )
        )
        # 未显式指定时采用 Dispatcher 的结果；显式指定时必须验证，不能强行
        # 把 mixed batch 或没有合法 key 的尺寸交给 FULL Wrapper。
        if cudagraph_runtime_mode is None:
            cudagraph_runtime_mode = _cudagraph_mode
        else:
            assert cudagraph_runtime_mode == _cudagraph_mode, (
                "Cudagraph runtime mode mismatch in dummy_run. "
                f"Expected {_cudagraph_mode}, but got {cudagraph_runtime_mode}."
            )

        # num_tokens_padded 是 embedding/模型实际处理的 token 行容量。
        num_tokens_padded = batch_desc.num_tokens
        # num_reqs_padded 是 Attention 实际处理的请求槽位容量，NONE 描述符可无此字段。
        num_reqs_padded = (
            batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
        )
        # 补齐后的布局也必须落在启动时分配的固定 buffer 容量内。
        assert num_tokens_padded <= self.max_num_tokens
        assert num_reqs <= num_reqs_padded <= self.max_num_reqs

        # attn_metadata 缺省为空，允许模型 profiling 在正式 KV cache 分配前执行。
        attn_metadata: dict[str, AttentionMetadata] | None = None
        # slot_mappings_by_group 供 Builder 使用，slot_mappings 则供 Attention 层查找。
        # 无正式 KV 时 slot_mappings_by_gid、slot_mappings_by_layer 均为 None，使用空写入映射。
        slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
            num_tokens_padded=num_tokens_padded,
            num_reqs_padded=num_reqs_padded,
            num_tokens_unpadded=num_tokens_unpadded,
        )
        # dummy 的全部 token 都没有真实物理 slot，不能只禁止 padding 行的写入。
        if slot_mappings_by_group is not None:
            # sm 是一个 KV group 的持久 slot view；原地填 -1，不更换其地址。
            for sm in slot_mappings_by_group.values():
                sm.fill_(-1)

        # 模型输入使用合法 token 0 和 position 0；这些值只用于启动执行和内存测量。
        self.input_ids.cpu.zero_()
        self.input_ids.gpu.zero_()
        self.positions.zero_()
        # 清除上轮调度长度，仅将本次有效请求前缀上传到同一 GPU buffer。
        self.num_scheduled_tokens.cpu.zero_()
        self.num_scheduled_tokens.np[:num_reqs] = num_scheduled_tokens
        self.num_scheduled_tokens.copy_to_gpu(num_reqs_padded)
        # dummy 没有真实请求进度；GPU 已计算长度与无效序列长度均不能沿用旧值。
        self.num_computed_tokens.zero_()
        self.seq_lens.zero_()
        self.optimistic_seq_lens_cpu.zero_()
        with self.synchronize_input_prep():
            # force_attention 用于 eager FA 预热，FULL 则必须把真实 Attention 纳入图。
            if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
                # 真实 FA 必须已有 KV 规格与 Builder；group 是唯一 KV group 内共享
                # 后端和规格的一组 Attention 层，不能只有组信息而没有 Builder 实例。
                if (
                    slot_mappings_by_group is None
                    or not self.attn_groups
                    or any(not group.metadata_builders for group in self.attn_groups[0])
                ):
                    raise RuntimeError(
                        "Attention dummy execution requires initialized KV cache and builders"
                    )
                # max_scheduled_tokens 是本次实际最长 query，用于校验模拟 KV 能覆盖它。
                max_scheduled_tokens = int(num_scheduled_tokens.max())
                # 显式长度用于长上下文 workspace profiling；所有 dummy 请求共用此值。
                if profile_seq_lens is not None:
                    if not max_scheduled_tokens <= profile_seq_lens <= self.max_model_len:
                        raise ValueError(
                            f"profile_seq_lens must be between {max_scheduled_tokens} "
                            f"and {self.max_model_len}, got {profile_seq_lens}"
                        )
                    # seq_lens 可以是一个标量，也可以是每请求的 CPU 长度 tensor。
                    seq_lens: int | torch.Tensor = profile_seq_lens
                # 混合布局模拟短 decode 和 prefill 后一个 token 的上下文，沿用上游布局。
                elif create_mixed_batch:
                    seq_lens = torch.tensor(
                        [1] * num_decode_tokens
                        + [min(num_prefill_tokens + 1, self.max_model_len)],
                        dtype=torch.int32,
                    )
                # 普通布局沿用 max_query_len 上界，但读取长度不得超过固定页表容量。
                else:
                    seq_lens = min(max_query_len, self.max_model_len)
                # 太长的单请求 query 无法用当前上下文容量表示，不能让 FA 越界读取页表。
                if max_scheduled_tokens > self.max_model_len:
                    raise ValueError("Dummy query length exceeds max_model_len")
                # optimistic_seq_lens_cpu 为 metadata 提供无 GPU 同步的长度上界。
                self.optimistic_seq_lens_cpu[:num_reqs] = seq_lens
                self.seq_lens.copy_(self.optimistic_seq_lens_cpu, non_blocking=True)

                # cum_num_tokens 是每个 dummy 请求的 packed query 终点，示例 [1,1,3]
                # 得到 [1,2,5]；query_pos 是共用 helper 顺便填好的 query 内位置工作区。
                cum_num_tokens = self._get_cumsum_and_arange(
                    num_scheduled_tokens, self.query_pos.np,
                )
                # 起点显式置零；补齐请求是空 query，所以尾部重复最后一个有效终点。
                self.query_start_loc.np[0] = 0
                self.query_start_loc.np[1:num_reqs + 1] = cum_num_tokens
                self.query_start_loc.np[num_reqs + 1:].fill(num_tokens_unpadded)

                # 页表只读取保留空块；清除 CPU/GPU 旧页号，不改变页表 storage。
                self.input_batch.block_table.block_table.cpu.fill_(NULL_BLOCK_ID)
                self.input_batch.block_table.num_blocks_per_row.fill(0)
                self.input_batch.block_table.commit_block_table(num_reqs_padded)
                # pad_attn 表示是否使用 FULL 的固定 Attention 容量；NONE 使用有效尺寸。
                pad_attn = cudagraph_runtime_mode == CUDAGraphMode.FULL
                attn_metadata, _ = self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded,
                    num_reqs=num_reqs,
                    max_query_len=max_query_len,
                    num_tokens_padded=num_tokens_padded if pad_attn else None,
                    num_reqs_padded=num_reqs_padded if pad_attn else None,
                    # FULL 首次调用可能自动捕获，也必须固定 max_seq_len 为最大上下文。
                    for_cudagraph_capture=is_graph_capturing or pad_attn,
                    slot_mappings=slot_mappings_by_group,
                )

        # 当前尚无上游的异步 input-prep event 协议；启动 dummy 同步当前 stream，
        # 保证 pinned CPU metadata 的 H2D 已完成，也保证捕获 stream 能读取完整输入。
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

        # input_ids 和 positions 是持久 GPU buffer 的前缀 view，形状采用执行容量。
        input_ids = self.input_ids.gpu[:num_tokens_padded]
        positions = self.positions[:num_tokens_padded]
        # Context 同时传递按层 metadata、写入映射、运行模式和确切 Graph key。
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens_padded,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=batch_desc,
            slot_mapping=slot_mappings,
        ):
            # hidden_states 是 Qwen3 返回的全部 token hidden states，含模型 padding 行。
            hidden_states = self._model_forward(input_ids=input_ids, positions=positions)

        # logit_indices 只取有效请求的最后一个 query，不把 padding 请求算入采样负载。
        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        # logit_indices_device 在模型调用后上传，不是 Graph 的固定输入，也不参与捕获。
        logit_indices_device = torch.from_numpy(logit_indices).to(
            self.device, non_blocking=True,
        )
        return hidden_states, hidden_states[logit_indices_device]

    @torch.inference_mode()
    def _dummy_sampler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> SamplerOutput:
        """执行模拟词表投影与 temperature 采样，供 profile_run 测量显存。

        profile_run 在 dummy forward 后传入每个模拟请求最后一个 query 的
        hidden state。本方法覆盖 LMHead、logits 精度转换和随机采样的运行开销，
        让显存 profiling 包含生成 token 所需的后处理。当前只支持单卡和
        temperature 采样，使用独立的模拟参数，不读取真实 InputBatch 的采样状态。

        Args:
            hidden_states: 形状为 [num_reqs, hidden_size] 的模拟请求 hidden states。
                只沿用其形状、设备和精度；实际数值替换为新的随机 tensor，避免
                dummy forward 中可能出现的 NaN/Inf 影响采样。

        Returns:
            sampler_output: SamplerOutput，其中 sampled_token_ids 为 [num_reqs, 1] 的设备端
                int32 tensor。profile_run 持有该结果直到 GPU 同步完成，不将候选
                token 写入请求，也不转换成 CPU 列表。

        Side Effects:
            分配随机 hidden states、词表 logits、模拟 temperature 和采样输出，
            消耗默认设备 RNG 状态；Sampler 可原地修改临时 logits。调用方的
            hidden_states、真实请求状态及 InputBatch 参数保持不变。
            不创建 CUDA Graph，也不在本方法内同步 GPU；同步由 profile_run 负责。

        Raises:
            RuntimeError: 采样阶段显存不足时补充请求数与配置建议，并保留原异常
                作为 cause；其他运行时错误直接向上传播。
        """
        # 新 hidden_states 保留输入的形状、设备和精度，以 [0,1) 随机值替代
        # dummy 的原始结果；这里只重新绑定局部变量，不改写调用方的 tensor。
        hidden_states = torch.rand_like(hidden_states)
        # logits 是 [num_reqs, vocab_size] 的词表分数；模型 Wrapper 会将
        # compute_logits 属性透传给原模型，实际投影由 LMHead/LogitsProcessor 执行。
        logits = self.model.compute_logits(hidden_states)
        # num_reqs 是参与本次采样的有效请求数，决定模拟参数列和输出的行数。
        num_reqs = logits.size(0)
        # dummy_tensors 为每个请求创建相同取值的设备参数列；v 是该列的标量值。
        dummy_tensors = lambda v: torch.full((num_reqs,), v, device=self.device)
        # dummy_metadata 只包含目前支持的温度与 processor 容器，不依赖真实请求。
        dummy_metadata = SamplingMetadata(
            # 所有模拟请求使用正温度 0.5，确保执行完整的 temperature 随机采样路径。
            temperature=dummy_tensors(0.5),
            # 独立的空容器不会引入惩罚项或修改真实 logits processor 的请求状态。
            logitsprocs=LogitsProcessors(),
        )
        # 只包围采样调用，沿用 vLLM 在 sampler 预热显存不足时补充诊断信息的边界。
        try:
            # sampler_output 保存设备端候选；Sampler 内部完成 FP32 转换、温度
            # 缩放和随机选择，这些临时分配也会纳入外层显存 profiling 的峰值。
            sampler_output = self.sampler(
                logits=logits,sampling_metadata=dummy_metadata,
            )
        # e 是采样阶段的运行时异常；检查原因后决定是否补充显存配置建议。
        except RuntimeError as e:
            # OOM 分支说明本次请求数下的采样负载超出显存容量，保留原始异常链。
            if "out of memory" in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up sampler with "
                    f"{num_reqs} dummy requests. Please try lowering "
                    "`max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine."
                ) from e
            # 非 OOM 错误沿用原异常，避免把算子或参数问题误报成显存不足。
            else:
                raise e
        # 返回 sampler_output，候选仅用于模拟负载；本方法不回写真实 token。
        return sampler_output

        
    def _sync_device(self) -> None:
        """等待当前 CUDA 设备全部 stream 的已提交工作完成。

        Worker 已在初始化时绑定当前设备。profile_run 在释放模拟结果前调用
        此方法，保证 forward、词表投影和采样已经完成；此接口不统计显存，
        不清理缓存，也不改变当前 stream。
        """
        torch.cuda.synchronize()
    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype, numpy: bool = True
    ) -> CpuGpuBuffer:
        """按当前设备和锁页策略创建一对固定容量 CPU/GPU 工作区。

        Args:
            *size: 两侧 tensor 的形状，调用方自行维护有效前缀。
            dtype: 两侧一致的元素类型。
            numpy: 是否建立与 CPU tensor 共享存储的 NumPy view。

        Returns:
            buffer: 初始清零的 CpuGpuBuffer；两侧内容只在显式拷贝后保持一致。
        """
        return CpuGpuBuffer(
            *size,
            dtype=dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            with_numpy=numpy,
        )

    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,
        arange_out: np.ndarray,
        cumsum_dtype: np.dtype  = np.int32,
    )-> np.ndarray:
        """计算每个 request 的 token 前缀和及 request 内相对位置。

        给定每条request 本轮执行的 token 数，例如 ``[2, 5, 3]``，返回累计长度
        ``[2, 7, 10]``，同时将 ``arange_out[:10]`` 填为：

        ``[0, 1, 0, 1, 2, 3, 4, 0, 1, 2]``。

        Args:
            num_tokens: 形状为 ``[num_reqs]`` 的 int32 数组。第 i 项是第 i 条
                活跃 request 本轮执行的 token 数。
            arange_out: 可复用的 NumPy 输出 buffer，容量至少为所有 token 数之和。
            cumsum_dtype: 累计长度数组的 dtype，默认 int32，与 attention 边界一致。

        Returns:
            cu_num_tokens: 形状为 ``[num_reqs]``、类型为 cumsum_dtype 的累计数组。第 i 项表示前
                ``i + 1`` 条 request 本轮 query token 的总数。
        """
        # 先求累计终点：[2, 5, 3] -> [2, 7, 10]。
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
        total_num_tokens = int(cu_num_tokens[-1])
        # 将各 query 的起点按长度展开：[0, 0, 2, 2, 2, 2, 2, 7, 7, 7]。
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
        # 从全局 arange 减去各 query 起点，得到请求内位置并写入复用 buffer。
        np.subtract(
            self.arange_np[:total_num_tokens],
            cumsums_offsets,
            out=arange_out[:total_num_tokens],
        )
        return cu_num_tokens
    def _compute_prev_positions(self, num_reqs: int) -> None:
        """构建当前请求槽位到上一轮采样行号的映射。

        请求的删除、加入和 batch 压缩可能改变当前槽位，因此必须通过
        request ID 查找上一轮输出行，不能直接复用当前槽位下标。

        Args:
            num_reqs: 当前持久 batch 中的活跃请求数。

        Side Effects:
            更新 prev_positions 的 CPU 活跃前缀；没有上一轮有效结果的
            请求填为 -1，不修改上一轮请求映射。
        """
        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        prev_positions = self.prev_positions.np[:num_reqs]
        # 首轮执行或上一轮没有有效采样时，所有请求均使用 CPU 输入。
        if not prev_req_id_to_index:
            prev_positions.fill(-1)
            return
        # 用请求 ID 连接两轮 batch，允许当前槽位顺序与上一轮不同。
        for i, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
            prev_positions[i] = prev_req_id_to_index.get(req_id, -1)
    def _prepare_input_ids(
        self,
        scheduler_output: SchedulerOutput,
        num_reqs: int,
        total_num_scheduled_tokens: int,
        cu_num_tokens: np.ndarray,
    ) -> None:
        """准备本轮 GPU 输入，并回填上一轮尚未同步到 CPU 的采样 token。

        没有上一轮 GPU 采样结果时，直接上传 CPU 输入。否则，根据
        prev_positions 将上一轮结果复制到当前请求对应的 packed 输入位置。

        当前仅支持单 token 采样，因此每个连续请求最多回填一个 token，
        目标位置是该请求本轮 query 在 packed 输入中的最后一个位置。

        Args:
            scheduler_output: 当前调度快照。保留参数以对齐 vLLM 接口，
                当前无投机解码，不需要读取其中的 draft token 信息。
            num_reqs: 当前 batch 的活跃请求数。
            total_num_scheduled_tokens: 本轮 packed 输入的有效 token 数。
            cu_num_tokens: 各请求 query 长度的累计和，不包含起始零。
                例如 query 长度为 [2, 3] 时，该数组为 [2, 5]。

        Side Effects:
            更新 input_ids.gpu 的有效区域。GPU 回填与后续 forward
            使用同一计算流排序，不在此等待采样结果回传 CPU。
        """
        if self.input_batch.prev_sampled_token_ids is None:
            # 普通路径：CPU staging 已包含本轮所需的全部输入。
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            return
        # 异步路径：部分 CPU 输入仍是占位，需要用上一轮 GPU 结果覆盖。
        prev_positions = self.prev_positions.np[:num_reqs]
        sample_flattened_indices: list[int] = []
        prev_indices: list[int] = []
        common_indices_match = True
        max_flattened_index = -1

        # 收集回填目标位置，以及对应的上一轮采样来源行。
        for cur_index in range(num_reqs):
            prev_index = prev_positions[cur_index]
            if prev_index < 0:
                # 新请求或上一轮未产生有效输出的请求使用 CPU 输入。
                continue

            prev_indices.append(int(prev_index))

            # 无投机解码时，上一轮采样 token 位于本轮 query 的末尾。
            flattened_index = int(cu_num_tokens[cur_index]) - 1
            sample_flattened_indices.append(flattened_index)

            # 来源行与目标位置一致时，可能使用连续切片拷贝。
            common_indices_match &= prev_index == flattened_index
            max_flattened_index = max(max_flattened_index, flattened_index)
        num_common_tokens = len(sample_flattened_indices)
        if num_common_tokens < total_num_scheduled_tokens:
            # 存在 prefill 等 CPU 输入时先整体上传，随后覆盖其中的占位。
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
        if num_common_tokens == 0:
            # 本轮没有可复用的上一轮结果，CPU 上传已经完成输入准备。
            return
        if (
            common_indices_match
            and max_flattened_index == num_common_tokens - 1
        ):
            # 来源行和目标位置恰好覆盖相同的连续前缀，可以省去索引构造和 scatter。
            self.input_ids.gpu[:num_common_tokens].copy_(
                self.input_batch.prev_sampled_token_ids[:num_common_tokens, 0],
                non_blocking=True,
            )
            return
        # batch 重排或目标位置不连续时，上传 packed 输入的目标下标。
        sampled_tokens_index_tensor = torch.tensor(
            sample_flattened_indices,
            dtype=torch.int64,
            pin_memory=self.pin_memory,
        ).to(self.device, non_blocking=True)

        # 上传上一轮采样 tensor 的来源行号，保持与目标下标逐项对应。
        prev_common_req_indices_tensor = torch.tensor(
            prev_indices,
            dtype=torch.int64,
            pin_memory=self.pin_memory,
        ).to(self.device, non_blocking=True)

        # 直接从 GPU 采样结果取值，覆盖 CPU staging 上传的占位 token。
        self.input_ids.gpu.scatter_(
            dim=0,
            index=sampled_tokens_index_tensor,
            src=self.input_batch.prev_sampled_token_ids[
                prev_common_req_indices_tensor, 0
            ],
        )
    def _get_slot_mappings(
        self,
        num_tokens_padded:int,
        num_reqs_padded:int,
        num_tokens_unpadded:int,
        ubatch_slices:UBatchSlices|None = None,
    ) ->tuple[
        dict[int,torch.Tensor]|None,
        dict[str,torch.Tensor]|list[dict[str,torch.Tensor]]|None
    ]:
        """构造 Attention metadata 与 ForwardContext 所需的两种 slot mapping。

        ``BlockTable.compute_slot_mapping()`` 已经在 ``_prepare_inputs()`` 中计算出
        每个本轮 packed token 对应的物理 KV slot。本方法不重新计算 slot，只负责：

        1. 在 CUDA Graph padding 存在时，将补齐 token 标记为 ``-1``；
        2. 按 KV cache group 编号组织 mapping，供 Attention metadata 构建使用；
        3. 按 Attention layer 名称组织 mapping，供 ``ForwardContext`` 查找使用。

        当前 nano-vLLM 只支持一个 decoder-only KV cache group，因此整个模型的
        Attention layers 都复用 ``InputBatch.block_table`` 产生的同一个 mapping
        tensor。各层的 KV tensor 彼此独立；共享的是“本 token 使用哪个物理 slot”
        这一坐标，而不是不同层的 K/V 内容。

        Args:
            num_tokens_padded: 本轮模型实际执行的 packed token 数。全 CUDA Graph
                模式下它可能大于真实 token 数，以匹配已捕获图的固定输入形状。
            num_reqs_padded: 本轮模型实际执行的 request 槽位数。当前单 group
                mapping 仍按 token 轴切片；此值校验请求容量，并供 metadata
                路径构造 query_start_loc、seq_lens 与 block table 的固定视图。
            num_tokens_unpadded: 本轮真实调度的 packed token 数。前
                ``num_tokens_unpadded`` 个 slot 有效；其后的 padding token 必须
                使用 ``-1``，避免 Attention 将伪 token 的 K/V 写入真实 cache。
            ubatch_slices: DBO 将一次 forward 切分为多个 micro-batch 的切片信息。
                当前 nano-vLLM 尚未支持 DBO，因此只能为 ``None``。

        Returns:
            (slot_mappings_by_gid, slot_mappings_by_layer):
                slot_mappings_by_gid: KV group ID 到 slot mapping 的映射，供 metadata 构建使用。
                slot_mappings_by_layer: Attention 层全名到 slot mapping 的映射，供 ForwardContext 使用；KV 未初始化或无 group 时两项均为 None。

        Raises:
            AssertionError: 当前尚未支持 DBO micro-batch 切分，或当前运行时出现了
                多 KV cache group 这一未实现的布局时抛出。
        """
        # DBO 需要把 mapping 按每个 micro-batch 的 token_slice 切开；该路径留待
        # 后续实现，不能静默忽略，否则会让后续 micro-batch 写入错误的 KV slot。
        assert ubatch_slices is None, (
            "nano-vLLM 当前尚未支持 DBO，因此 ubatch_slices 必须为 None。"
        )
        # ``kv_cache_config`` 仅会在正式 KV cache 初始化后设置。没有 KV cache 的
        # profiling 或特殊模型路径无需构造 slot mapping。
        if not (
            hasattr(self, "kv_cache_config")
            and self.kv_cache_config is not None
            and self.kv_cache_config.kv_cache_groups
        ):
            return None, None
        # 当前 InputBatch 只持有一张 BlockTable，故只能支持一个逻辑 KV group。
        # 多 group 时每个 group 都需要自己独立的 block table 与 slot mapping。
        assert len(self.kv_cache_config.kv_cache_groups) == 1, (
            "当前 InputBatch 只支持一个 KV cache group。"
        )
        # 不允许 padding 边界颠倒。真实 token 是有效前缀，padding 只能位于其后。
        assert 0 <= num_tokens_unpadded <= num_tokens_padded, (
            "num_tokens_unpadded 必须满足 "
            f"0 <= {num_tokens_unpadded} <= {num_tokens_padded}。"
        )
        # 保留该断言以明确 request padding 是合法的 batch 形状信息。当前单 group
        # slot mapping 只沿 packed-token 轴组织，因此本方法不会直接按它切片。
        assert num_reqs_padded >= self.input_batch.num_reqs, (
            "num_reqs_padded 不能小于当前真实活跃 request 数。"
        )
        # 取出本轮 forward 实际要传给 Attention 的 slot mapping 前缀。它是持久 GPU
        # buffer 的 view，不会复制；后续 layer 字典中的各层也会共享此 tensor。
        slot_mapping = self.input_batch.block_table.slot_mapping.gpu[
            :num_tokens_padded
        ]
        # 若为了 CUDA Graph 将 token 数补齐，补齐位置没有真实 K/V 写入目标。Attention
        # backend 约定负 slot 表示 padding，因此必须覆写上轮可能残留的 slot 值。
        slot_mapping[num_tokens_unpadded:num_tokens_padded].fill_(-1)

        # 当前唯一 group 的编号固定为 0。该格式服务于按 group 构建 attention metadata
        # 的路径，而不是直接让模型层按数字查找。
        slot_mappings_by_gid = {0: slot_mapping}
        # ForwardContext 中的 Attention module 通过其稳定层名查询运行时 metadata。
        # 同组的所有层复用同一个 slot mapping tensor，不产生逐层副本。
        kv_cache_group = self.kv_cache_config.kv_cache_groups[0]
        slot_mappings_by_layer = {
            layer_name: slot_mapping
            for layer_name in kv_cache_group.layer_names
        }
        return slot_mappings_by_gid, slot_mappings_by_layer

    def may_reinitialize_input_batch(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int],
    )->None:
        """在最终 KV cache 布局与初始占位布局不一致时重建 InputBatch。

        ``InputBatch`` 的 block table 宽度、逻辑 block size 和 kernel page size
        都必须与最终 KV cache 规划一致。模型刚构造时尚未完成显存 profiling，
        因此 ``__init__`` 中只能创建占位 InputBatch；正式 KV cache 配置下发后，
        才能在这里验证或重建它。

        当前 InputBatch 只维护一张 ``BlockTable``，因此端到端执行仅支持一个
        KV cache group。该限制属于 Worker 侧 batch 表示能力，与
        ``KVCacheConfig`` 本身是否能表达多个 group 是两回事。

        Args:
            kv_cache_config: Engine 根据 profiling 生成的最终 KV cache 分配计划。
            kernel_block_sizes: 每个 KV cache group 对应的 Attention kernel page
                大小，顺序必须与 ``kv_cache_groups`` 一致。

        Raises:
            NotImplementedError: 当前模型包含多个 KV cache groups 时抛出。
            AssertionError: 尝试在已有活跃请求时替换 InputBatch 时抛出。

        Side Effects:
            若现有 InputBatch 的 block 相关布局不匹配，则创建新的 InputBatch。
            已初始化的 logits processor 容器会迁移到新 batch。
        """
        kv_cache_groups = kv_cache_config.kv_cache_groups

        # Attention-free 模型没有 block table；当前 Qwen 路径不会进入这里。
        if not kv_cache_groups:
            return
         # 每个 group 都必须有一个对应的 kernel block size。
        assert len(kernel_block_sizes) == len(kv_cache_groups)
        # 当前 InputBatch 是单 BlockTable 版本，不能假装支持多个独立 group。
        if len(kv_cache_groups) != 1:
            raise NotImplementedError(
                "当前 InputBatch 只支持一个 KV cache group；"
                "多 group 需要 MultiGroupBlockTable。"
            )
        # 唯一 group 决定 Scheduler 逻辑 block 与 Attention kernel page 的大小。
        kv_cache_group = kv_cache_groups[0]
        block_size = kv_cache_group.kv_cache_spec.block_size
        kernel_block_size = kernel_block_sizes[0]
        # 一条最大长度请求最多需要多少个 manager block；它不是全局 num_blocks。
        max_num_blocks_per_req = (
            self.max_model_len + block_size - 1
        ) // block_size
        # 目前就一个group
        if(
            block_size!=self._init_block_sizes[0]
            or kernel_block_sizes!=self._init_kernel_block_sizes
        ):
            # 在捕获前完成规格收尾。活跃请求或已捕获 Graph 都不能换掉其引用的
            # InputBatch/block table storage；后续容量调整必须先清理运行状态。
            assert self.input_batch.num_reqs == 0, "Cannot replace an active InputBatch"
            if isinstance(getattr(self, "model", None), CUDAGraphWrapper):
                assert not self.model.concrete_cudagraph_entries, (
                    "Cannot replace InputBatch after CUDA Graph capture"
                )
            self._init_block_sizes[0]=block_size
            self._init_kernel_block_sizes=kernel_block_sizes
            self.input_batch = InputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=self.max_model_len,
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                pin_memory=self.pin_memory,
                vocab_size=self.model_config.get_vocab_size(),
                block_size=block_size,
                kernel_block_size=kernel_block_size,
                max_num_blocks_per_req=max_num_blocks_per_req,
                logitsprocs=self.input_batch.logitsprocs,
            )

    def _allocate_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
    ) -> dict[str, torch.Tensor]:
        """按物理 allocation 计划创建 layer-name 到 raw storage 的映射。

        Args:
            kv_cache_config: Engine 下发的最终 KV cache 分配计划。

        Returns:
            kv_cache_raw_tensors: 从 Attention layer 名称到一维 int8 raw storage 的映射。同一个
                ``KVCacheTensor.shared_by`` 中的层会引用同一个 raw tensor。
        """
        kv_cache_raw_tensors:dict[str,torch.Tensor]={}  # 完整层名到 raw storage 的绑定表。
        # 分配计划按物理 tensor 组织，不一定与 layer 数量一一对应。
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            # int8 仅作为逐字节存储载体；后续会按 KVCacheSpec.dtype 重新 view。
            tensor = torch.zeros(
                kv_cache_tensor.size,
                dtype=torch.int8,
                device=self.device,
            )
            # shared_by 中的层共享这次分配，不能为每个层名再次分配一份。
            for layer_name in kv_cache_tensor.shared_by:
                kv_cache_raw_tensors[layer_name]=tensor
        # 用逻辑 group 的所有层反查分配结果，缺失或额外层都说明计划不一致。
        layer_names=set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                layer_names.add(layer_name)

        # 这里只验证绑定名称，具体 dtype/shape/stride 在 reshape 阶段解释。
        assert layer_names == set(kv_cache_raw_tensors.keys()),(
            "Some layers are not correctly initialized"
        )
        return kv_cache_raw_tensors
    def _reshape_kv_cache_tensors(
        self,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
        kernel_block_sizes: list[int],
    ) -> dict[str, torch.Tensor]:
        """按各 AttentionGroup 后端声明的布局重解释 raw KV storage。

        Args:
            kv_cache_raw_tensors: 层名到一维 int8 原始存储的映射。
            kernel_block_sizes: 按逻辑 KV group 编号排列的 kernel page 大小。

        Returns:
            kv_caches: 层名到 combined KV cache view 的映射。view 与原始存储共享内存；
                当前 FlashAttention 的逻辑布局是
                [num_kernel_blocks, K/V, kernel_block_size, kv_heads, head_size]。
        """
        kv_caches: dict[str, torch.Tensor] = {}
        # 一个逻辑 KV group 可以有多个 AttentionGroup；后者决定各层的布局。
        for groups in self.attn_groups:
            for group in groups:
                kv_cache_spec = group.kv_cache_spec
                if not isinstance(kv_cache_spec, FullAttentionSpec):
                    raise NotImplementedError("Only FullAttentionSpec is supported")

                kernel_block_size = kernel_block_sizes[group.kv_cache_group_id]
                assert kv_cache_spec.block_size % kernel_block_size == 0
                num_kernel_pages_per_block = (
                    kv_cache_spec.block_size // kernel_block_size
                )

                for layer_name in group.layer_names:
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
                    num_manager_blocks = (
                        raw_tensor.numel() // kv_cache_spec.page_size_bytes
                    )
                    num_kernel_blocks = (
                        num_manager_blocks * num_kernel_pages_per_block
                    )
                    # 后端定义逻辑形状；stride order 描述底层连续存储的维度顺序。
                    logical_shape = group.backend.get_kv_cache_shape(
                        num_kernel_blocks,
                        kernel_block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size,
                        cache_dtype_str="auto",
                    )
                    try:
                        stride_order = group.backend.get_kv_cache_stride_order()
                    except NotImplementedError:
                        stride_order = tuple(range(len(logical_shape)))
                    assert sorted(stride_order) == list(range(len(logical_shape)))
                    physical_shape = tuple(
                        logical_shape[dim] for dim in stride_order
                    )
                    inverse_order = tuple(
                        stride_order.index(dim) for dim in range(len(stride_order))
                    )
                    # view/permute 只改变坐标解释，不复制 GPU KV 内容。
                    kv_caches[layer_name] = (
                        raw_tensor.view(kv_cache_spec.dtype)
                        .view(physical_shape)
                        .permute(*inverse_order)
                    )

        return kv_caches
    
    @contextmanager
    def synchronize_input_prep(self):
        """在改写 CPU staging 前等待上一轮输入准备完成，并记录本轮边界。

        execute_model() 用它包住请求槽位更新、H2D 和输入 metadata 构建。
        退出作用域时的事件记录在当前计算流上、模型 forward 之前；下一轮
        只需等此前已提交的输入准备工作，而不在这里等待上一轮完整 forward。

        Yields:
            无值；作用域内可以复用上一轮已消费的 CPU staging tensor。

        Side Effects:
            异步模式进入时可能阻塞 CPU，并在正常或异常退出时记录新事件。
            同步模式未创建事件，直接执行作用域，不增加额外等待。
        """
        if self.prepare_inputs_event is None:
            yield
            return

        # non_blocking H2D 返回后设备可能仍在读取 CPU 源内存，改写前必须等待。
        self.prepare_inputs_event.synchronize()
        try:
            yield
        finally:
            # 即使准备失败也维持下一次进入时可等待的边界；不在这里捕获异常。
            self.prepare_inputs_event.record()

    def _model_forward(
        self,
        input_ids:torch.Tensor|None=None,
        positions:torch.Tensor|None=None,
        **model_kwargs:dict[str,Any],
    )->Any:
        """调用已安装的模型入口，沿用外层设置的 ForwardContext。

        Args:
            input_ids: 当前 packed token 输入，Graph 路径可包含容量补齐。
            positions: 与输入一一对应的完整序列位置。
            **model_kwargs: 保留扩展参数入口，当前文本模型路径不向模型转发。

        Returns:
            hidden_states: 模型产生的 hidden states；词表投影和采样由外层另行执行。
        """
        return self.model(
            input_ids = input_ids,
            positions=positions
        )
