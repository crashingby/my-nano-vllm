"""定义 GPU Worker 侧持久 InputBatch 的数据结构。

InputBatch 不是 Scheduler 的权威请求状态，而是 GPUModelRunner 为避免每轮重复
分配和复制大量 tensor 而维护的固定容量工作区。它保存当前活跃 request 的紧凑
槽位布局、完整 token 序列、KV block table 和采样参数。

当前实现只支持单 KV cache group、文本生成和 temperature sampling。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import torch

from nanovllm.sample.logits_processor import (
    BatchUpdateBuilder,
    LogitsProcessors,
    MoveDirectionality,
)
from nanovllm.sampling_params import SamplingParams
from nanovllm.sample.metadata import SamplingMetadata
from nanovllm.worker.block_table import BlockTable

@dataclass
class CachedRequestState:
    """Worker 侧缓存的单个文本生成请求状态。

    Scheduler 中的 ``Request`` 才是逻辑请求、调度状态和 KV block 所有权的权威。
    本类仅保存 GPUModelRunner 为后续连续 batch 执行所需的镜像状态。请求暂时未被
    调度或被抢占时，本对象可以继续保留；它不一定在当前 ``InputBatch`` 中。

    Attributes:
        req_id: 与 Scheduler Request 相同的全局唯一请求 ID。
        prompt_token_ids: 请求原始 prompt 的 token IDs。当前 MVP 只支持文本 token，
            不支持 prompt embeddings，因此始终是 list。
        sampling_params: 当前请求的采样参数。MVP 只会读取 ``temperature``。
        block_ids: 当前单 KV cache group 的完整逻辑 block table。
        num_computed_tokens: 当前已经写入 KV cache 的 token 数。它是 Worker mirror
            中的执行进度，不包含本轮尚未 forward 的 token。
        output_token_ids: 已经采样确认的 completion token IDs，不包含 prompt。
        num_prompt_tokens: prompt token 数，由 ``prompt_token_ids`` 在构造时计算。
    """
    # Worker 与 Scheduler 使用同一个 request ID 建立跨进程对应关系。
    req_id: str
    # 当前只支持 token prompt，不支持多模态 feature 或 prompt embeddings。
    prompt_token_ids: list[int]
    # temperature 等采样配置由 request 创建时确定，后续保持不变。
    sampling_params: SamplingParams
    # 单 group MVP 中 block table 直接是 list[int]，不使用 vLLM 的 tuple[list[int], ...]。
    block_ids: list[int]
    # 本轮 forward 前已经实际写入 KV cache 的逻辑 token 数。
    num_computed_tokens: int
    # Worker 在 sample_tokens() 后持续追加的生成 token 序列。
    output_token_ids: list[int]
    def __post_init__(self) -> None:
        """初始化与 prompt 长度相关的派生状态。"""
        # 当前 MVP 要求每个请求都携带真实 token IDs，因此长度可以直接获得。
        self.num_prompt_tokens = len(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        """返回当前 prompt 与已生成 token 拼接后的总长度。"""
        return self.num_prompt_tokens + len(self.output_token_ids)
    def get_token_id(self, index: int) -> int:
        """按逻辑位置获取一个 token ID。

        Args:
            index: 在 ``prompt + output`` 完整序列中的逻辑 token 下标。

        Returns:
            对应位置的 token ID。若该位置尚未存在，返回 ``-1``；后者通常意味着
            调度器要求 Worker 使用了尚未同步到本地的 token，属于内部协议错误。
        """
        if index < self.num_prompt_tokens:
            return self.prompt_token_ids[index]

        output_index = index - self.num_prompt_tokens
        if output_index < len(self.output_token_ids):
            return self.output_token_ids[output_index]

        return -1

class InputBatch:
    """维护当前 GPU 执行 batch 的固定容量槽位和输入 buffer。

    连续批处理的相邻两轮通常复用大部分 request。为了避免每轮重新构造
    ``[num_reqs, max_model_len]`` token tensor，本类按 ``max_num_reqs`` 一次性
    分配 CPU/GPU buffer，再通过 ``req_id_to_index`` 将逻辑 request 映射到物理槽位。

    任何时刻，``_req_ids``、``token_ids_cpu``、``num_computed_tokens_cpu``、
    ``temperature_cpu`` 和 ``block_table`` 的同一个 ``req_index`` 必须描述同一
    个 request。``remove_request()`` 后必须调用 ``condense()``，使有效 request
    再次紧凑地位于前 ``num_reqs`` 个槽位。

    Attributes:
        max_num_reqs: InputBatch 可容纳的 request 上限。
        max_model_len: 每个 request 可保存的 token 数上限。
        max_num_batched_tokens: 单轮 packed forward 的 token 上限。
        device: GPUModelRunner 使用的 CUDA device。
        pin_memory: CPU staging tensor 是否使用 pinned memory。
        vocab_size: 模型全局词表大小。当前 MVP 的 temperature sampling 尚未直接
            使用它，但它是后续 top-k、allowed-token mask 等采样功能的基础元数据。
        block_size: 当前唯一 KV cache group 的 block token 数。
        kernel_block_size: Attention kernel 实际读取 paged KV cache 时使用的 page
            大小。当前 MVP 要求它等于 ``block_size``；未来 hybrid block 支持会让
            一个逻辑 block 展开为多个 kernel blocks。
        max_num_blocks_per_req: 单个 request 的 block table 最多容纳的逻辑 block
            数。它决定 block-table buffer 的列数上限，而不是当前已分配 block 数。
        _req_ids: 按物理 batch 槽位保存的 request IDs；更新中暂时允许 ``None``。
        req_id_to_index: request ID 到当前物理槽位的反向索引。
        token_ids_cpu_tensor: 固定容量的 CPU token matrix。
        num_prompt_tokens_cpu_tensor: 每个槽位的 prompt 长度。
        num_computed_tokens_cpu_tensor: 每个槽位已经写入 KV cache 的 token 数。
        block_table: 单 KV cache group 的 GPU/CPU block-table 工作区。
        temperature_cpu_tensor: request temperature 的 CPU staging buffer。
        temperature: 与 CPU staging buffer 对应的 GPU temperature tensor。
        req_output_token_ids: 每个槽位引用的 output token 列表。
        batch_update_builder: 收集当前 engine step 的请求删除、新增和槽位移动事件；
            它同时负责提供按最小下标复用的空槽位。
        logitsprocs: 当前 Worker 上已初始化的 logits processor 容器。MVP 默认
            为空，但仍在 batch 更新边界调用其 ``update_state()``，以保持与 vLLM
            相同的生命周期。
    """
    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        device: torch.device,
        pin_memory: bool,
        vocab_size: int,
        # 当前只支持一个 KV cache group，因此不是 vLLM 的 block_sizes: list[int]。
        block_size: int,
        kernel_block_size: int,
        max_num_blocks_per_req: int | None = None,
        logitsprocs: LogitsProcessors | None = None,
    ) -> None:
        """创建固定容量的 CPU/GPU batch 工作区。

        Args:
            max_num_reqs: 同时驻留在 InputBatch 中的 request 数上限。
            max_model_len: 单个 request 的最大 token 长度。
            max_num_batched_tokens: 单轮模型 forward 的 packed token 上限。
            device: InputBatch 所属 GPU device。
            pin_memory: 是否为频繁传输到 GPU 的 CPU buffer 启用 pinned memory。
            vocab_size: 模型全局词表大小。当前不参与 temperature-only sampling，
                但保留它以维持 InputBatch 的语言模型元数据接口。
            block_size: 当前唯一 KV cache group 的一个 block 覆盖的 token 数。
            kernel_block_size: Attention backend 实际读取 block table 时使用的 page
                大小。当前 MVP 中必须等于 ``block_size``。
            max_num_blocks_per_req: 可选的单 request manager block 上限。未提供时按
                ``ceil(max_model_len / block_size)`` 推导；这是单 group MVP 对 vLLM
                ``MultiGroupBlockTable`` 默认推导规则的对应实现。
            logitsprocs: 已初始化的 logits processor 容器。未提供时使用空容器，
                使 InputBatch 的槽位更新流程不依赖具体采样 feature。
        """
        # 保存固定容量约束；后续 add_request() 和 prepare_inputs() 都依赖它们。
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.device = device
        self.vocab_size = vocab_size
        self.pin_memory = pin_memory
        # Scheduler 和 KVCacheManager 按这个大小分配逻辑 KV blocks。
        self.block_size = block_size
        # Attention backend 按这个大小解释 block table；当前先保存，实际 hybrid
        # block 转换由后续 BlockTable 实现负责。
        self.kernel_block_size = kernel_block_size

        # 未显式覆盖时，一条最大上下文 request 需要的 manager block 数只由逻辑
        # token 上限和 KVCacheManager 的 block_size 决定。不要在这里按
        # kernel_block_size 计算：hybrid page 展开由 BlockTable 统一负责。
        if max_num_blocks_per_req is None:
            max_num_blocks_per_req = (
                max_model_len + block_size - 1
            ) // block_size
        # 该值决定预分配 block-table 的行宽，零或负数无法表达一个合法 request。
        if max_num_blocks_per_req <= 0:
            raise ValueError("max_num_blocks_per_req must be positive")
        # 这里仍以 manager block 为单位；BlockTable 内部会在 hybrid 情况下换算为
        # kernel page 列数。
        self.max_num_blocks_per_req = max_num_blocks_per_req

        # _req_ids 可在 remove/add/condense 的中间阶段出现 None；稳定执行前必须紧凑。
        self._req_ids: list[str | None] = []
        # 这是从逻辑 request 身份查找 batch 槽位的唯一入口。
        self.req_id_to_index: dict[str, int] = {}
        # token matrix 不会整块拷贝到 GPU；ModelRunner 会按本轮实际执行位置 gather，
        # 因此与 vLLM 一样不需要为它使用 pinned memory。
        self.token_ids_cpu_tensor = torch.zeros(
            (max_num_reqs, max_model_len),
            device="cpu",
            dtype=torch.int32,
            pin_memory=False,
        )
        # NumPy view 便于在 Python 调度循环中批量写入 token IDs。
        self.token_ids_cpu = self.token_ids_cpu_tensor.numpy()
        # 当前仅支持 token-ID 文本输入；prompt embeddings 接入时恢复 is_token_ids_tensor。
        # 每个 request 的 prompt 长度。它用于从完整 token 行中区分 prompt/output。
        self.num_prompt_tokens_cpu_tensor = torch.zeros(
            (max_num_reqs,),
            device="cpu",
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        self.num_prompt_tokens = self.num_prompt_tokens_cpu_tensor.numpy()

        # 每个 request 已经写入 KV cache 的长度；ModelRunner 据此计算本轮 positions。
        self.num_computed_tokens_cpu_tensor = torch.zeros(
            (max_num_reqs,),
            device="cpu",
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        self.num_computed_tokens_cpu = (
            self.num_computed_tokens_cpu_tensor.numpy()
        )
        # 单 group 版本直接使用 BlockTable；未来多 group 才替换为 MultiGroupBlockTable。
        self.block_table = BlockTable(
            block_size=self.block_size,
            kernel_block_size=self.kernel_block_size,
            max_num_reqs=self.max_num_reqs,
            max_num_blocks_per_req=self.max_num_blocks_per_req,
            max_num_batched_tokens=self.max_num_batched_tokens,
            pin_memory=self.pin_memory,
            device=self.device,
        )
        # temperature 由 CPU 侧 request 状态更新，并在 refresh_metadata() 时批量传到 GPU。
        self.temperature = torch.empty(
            (max_num_reqs,),
            dtype=torch.float32,
            device=device,
        )
        self.temperature_cpu_tensor = torch.empty(
            (max_num_reqs,),
            dtype=torch.float32,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.temperature_cpu = self.temperature_cpu_tensor.numpy()

        # 这里保存 CachedRequestState.output_token_ids 的引用，采样器追加输出 token 后
        # InputBatch 无需复制即可看到新的逻辑序列长度。
        self.req_output_token_ids: list[list[int] | None] = []

        # 统一维护本轮的删除、新增和移动事件。不要另行维护删除槽位列表，否则
        # InputBatch 的槽位布局与 future logits processor 的状态可能出现分叉。
        self.batch_update_builder = BatchUpdateBuilder()
        # 即使当前没有具体 logits processor，也始终保留该容器和同步调用点。
        self.logitsprocs = logitsprocs or LogitsProcessors()
        # metadata 只引用活跃前缀；空 batch 初始化时 temperature 的形状为 [0]。
        self.sampling_metadata = self._make_sampling_metadata()

    @property
    def req_ids(self) -> list[str]:
        """返回当前稳定 batch 中按物理槽位排序的 request IDs.

        Returns:
            前 ``num_reqs`` 个连续槽位中的 request IDs。

        Raises:
            AssertionError: 在 remove/add/condense 的中间状态访问时抛出。
        """
        # 正常执行 forward 前，所有空洞必须已由 condense() 消除。
        assert len(self._req_ids) == self.num_reqs
        assert all(req_id is not None for req_id in self._req_ids)
        return cast(list[str], self._req_ids)

    @property
    def num_reqs(self) -> int:
        """返回当前持久 batch 中存活 request 的数量。

        `_req_ids` 在 remove/add/condense 的中间阶段可以含有 ``None``，因此不能
        用它的列表长度表示活跃请求数。`req_id_to_index` 只包含仍在 batch 中的
        request，恰好是 vLLM 同名属性使用的权威计数来源。
        """

        return len(self.req_id_to_index)

    def _register_add_request(
        self,
        request: CachedRequestState,
    ) -> int:
        """为新加入的 request 选择一个可用的物理 batch 槽位。

        优先复用本轮已删除 request 留下的最小槽位；没有空洞时才在列表尾部追加。
        除了选择槽位，本方法还要为 future logits processor 登记完整的新增请求
        元数据。这样 processor 才能在同一槽位丢弃旧状态并初始化新状态。

        Args:
            request: 即将加入 InputBatch 的 Worker 侧请求镜像。

        Returns:
            分配给该 request 的物理 ``req_index``。
        """
        # pop_removed() 会按需排序删除索引，并返回数值最小的空槽位。被取走的
        # 删除事件由下面的 added 事件覆盖，因此不应再出现在最终 BatchUpdate.removed。
        if (new_req_index := self.batch_update_builder.pop_removed()) is None:
            # 没有可复用空洞时，新 request 写在当前活跃前缀的末尾。
            new_req_index = self.num_reqs

        # Scheduler 的 max_num_seqs 约束应保证这里永远不会越界。
        assert new_req_index < self.max_num_reqs, (
            "InputBatch exceeds max_num_reqs: "
            f"req_index={new_req_index}, max_num_reqs={self.max_num_reqs}."
        )

        # 该标记供未来 pooling 等轻量路径快速判断 batch 布局是否改变。
        self.batch_update_builder.batch_changed = True
        # 当前所有 request 都是自回归文本生成请求，必定具有 SamplingParams；保存
        # output_token_ids 的同一列表引用，让 processor 在后续 decode 中看到新 token。
        self.batch_update_builder.added.append(
            (
                new_req_index,
                request.sampling_params,
                request.prompt_token_ids,
                request.output_token_ids,
            )
        )
        return new_req_index

    def add_request(
        self,
        request: CachedRequestState,
    ) -> int:
        """将一个 Worker 侧请求镜像写入 InputBatch 的一个物理槽位。

        本方法会同步所有与 ``req_index`` 对齐的状态：request ID、完整 token 行、
        prompt 长度、KV 进度、block table、temperature 和 output token 引用。

        Args:
            request: 要加入当前执行 batch 的 Worker 侧请求状态。

        Returns:
            分配给该 request 的物理 ``req_index``.

        Raises:
            AssertionError: request 已存在、token 长度超过模型上限，或复用槽位不是
                空槽位时抛出，表示 Worker/Scheduler 状态同步出现错误。
        """
        req_id = request.req_id
        assert req_id not in self.req_id_to_index, (
            f"Request {req_id!r} already exists in InputBatch."
        )

        req_index = self._register_add_request(request)
        num_tokens = request.num_tokens
        assert num_tokens <= self.max_model_len, (
            "Request token count exceeds InputBatch capacity: "
            f"req_id={req_id!r}, num_tokens={num_tokens}, "
            f"max_model_len={self.max_model_len}."
        )

        # 新槽位追加到列表末尾；复用槽位时该位置必须是 remove_request() 留下的空洞。
        if req_index == len(self._req_ids):
            self._req_ids.append(req_id)
            self.req_output_token_ids.append(request.output_token_ids)
        else:
            assert self._req_ids[req_index] is None
            self._req_ids[req_index] = req_id
            self.req_output_token_ids[req_index] = request.output_token_ids

        # 从逻辑 request ID 回查物理槽位，后续更新 block 或 token 时都依赖此映射。
        self.req_id_to_index[req_id] = req_index

        # prompt 与已生成 output 共同构成该 request 在 CPU token matrix 中的一整行。
        num_prompt_tokens = request.num_prompt_tokens
        self.token_ids_cpu[req_index, :num_prompt_tokens] = (
            request.prompt_token_ids
        )
        self.token_ids_cpu[
            req_index,
            num_prompt_tokens:num_tokens,
        ] = request.output_token_ids

        # 这些长度是 ModelRunner 计算本轮 input positions 的基础。
        self.num_prompt_tokens[req_index] = num_prompt_tokens
        self.num_computed_tokens_cpu[req_index] = (
            request.num_computed_tokens
        )

        # 新请求第一次进入 Worker 时发送的是完整 block table，因此直接覆盖该行。
        self.block_table.add_row(request.block_ids, req_index)

        # 当前 SamplingParams 只有 temperature；top-p/top-k/penalty 等 buffer 后续
        # 扩展采样功能时再按 vLLM 的原位置加入。
        self.temperature_cpu[req_index] = request.sampling_params.temperature

        return req_index
    def remove_request(self, req_id: str) -> int | None:
        """从当前执行 batch 移除 request，并保留一个待压缩的槽位空洞。

        被移除的 request 可能是真的 finished，也可能只是本轮未调度或被抢占。
        后两种情形的 ``CachedRequestState`` 仍由 GPUModelRunner.requests 保存；
        本方法只处理 InputBatch 中短生命周期的物理槽位。

        调用者必须在本轮所有 remove/add 操作完成后调用 ``condense()``。

        Args:
            req_id: 需要从当前 InputBatch 删除的 request ID。

        Returns:
            被移除的物理槽位；若 request 当前不在 InputBatch 中则返回 ``None``。
        """
        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is None:
            return None

        # 将槽位标记为空，但暂不移动其他 request，避免连续删除时重复搬运数据。
        self._req_ids[req_index] = None
        self.req_output_token_ids[req_index] = None

        # 不再让后续 attention metadata 使用旧 request 的 block table。
        self.block_table.clear_row(req_index)
        # 删除后还可能经历请求补齐，长度不可保留为上一轮的值。
        self.num_computed_tokens_cpu[req_index] = 0
        self.num_prompt_tokens[req_index] = 0

        # 删除槽位的排序、最小空槽位复用和最终 processor 更新均由同一个 builder
        # 维护，避免 InputBatch 私有状态与 processor 观察到的事件不一致。
        self.batch_update_builder.removed_append(req_index)

        return req_index

    def swap_states(
        self,
        index_1: int,
        index_2: int,
    ) -> None:
        """交换两个活跃 request 的全部按槽位对齐状态。

        该接口保留 vLLM 的 batch reordering 能力。当前 MVP 的 ``condense()`` 主要
        使用单向搬运，但未来 attention backend 需要按请求类型重排 batch 时，仍可
        安全复用本方法。

        Args:
            index_1: 第一个活跃 request 的物理槽位。
            index_2: 第二个活跃 request 的物理槽位。

        Raises:
            AssertionError: 任一槽位为空时抛出。
        """
        if index_1 == index_2:
            return

        req_id_1 = self._req_ids[index_1]
        req_id_2 = self._req_ids[index_2]
        assert req_id_1 is not None
        assert req_id_2 is not None

        # 只复制有效 token 前缀，而不是整行 max_model_len，避免重排代价随上下文上限增长。
        num_tokens_1 = self._get_active_token_count(index_1)
        num_tokens_2 = self._get_active_token_count(index_2)
        max_num_tokens = max(num_tokens_1, num_tokens_2)

        # request ID、output token 引用和反向索引必须始终一起交换。
        self._req_ids[index_1], self._req_ids[index_2] = (
            self._req_ids[index_2],
            self._req_ids[index_1],
        )
        self.req_output_token_ids[index_1], self.req_output_token_ids[index_2] = (
            self.req_output_token_ids[index_2],
            self.req_output_token_ids[index_1],
        )
        self.req_id_to_index[req_id_1], self.req_id_to_index[req_id_2] = (
            index_2,
            index_1,
        )

        # 按 request 行组织的长度与采样参数也必须同步迁移。
        self.num_prompt_tokens[index_1], self.num_prompt_tokens[index_2] = (
            self.num_prompt_tokens[index_2],
            self.num_prompt_tokens[index_1],
        )
        (
            self.num_computed_tokens_cpu[index_1],
            self.num_computed_tokens_cpu[index_2],
        ) = (
            self.num_computed_tokens_cpu[index_2],
            self.num_computed_tokens_cpu[index_1],
        )
        self.temperature_cpu[index_1], self.temperature_cpu[index_2] = (
            self.temperature_cpu[index_2],
            self.temperature_cpu[index_1],
        )

        # NumPy 不支持安全的行切片原地 tuple swap，因此显式保留一行临时副本。
        token_ids_tmp = self.token_ids_cpu[
            index_1,
            :max_num_tokens,
        ].copy()
        self.token_ids_cpu[index_1, :max_num_tokens] = self.token_ids_cpu[
            index_2,
            :max_num_tokens,
        ]
        self.token_ids_cpu[index_2, :max_num_tokens] = token_ids_tmp

        # block table 与 token row 的 index 必须一致，否则 slot mapping 会指向错误 page。
        self.block_table.swap_row(index_1, index_2)

        # logits processor 的逐请求状态也必须交换，才能继续与 InputBatch 的行对齐。
        self.batch_update_builder.moved.append(
            (index_1, index_2, MoveDirectionality.SWAP)
        )

    def _get_active_token_count(self, req_index: int) -> int:
        """返回一个活跃槽位当前真实需要保留的 token 前缀长度。"""
        output_token_ids = self.req_output_token_ids[req_index]
        assert output_token_ids is not None
        return int(self.num_prompt_tokens[req_index]) + len(output_token_ids)

    def condense(self) -> None:
        """将存活 request 向前压缩，消除 remove_request() 产生的槽位空洞。

        算法与 vLLM 一致：从最大活跃槽位向最小空槽位单向搬运，而不是每删除一个
        request 就整体移动后续所有行。完成后活跃 request 必须严格占据区间
        ``[0, num_reqs)``。

        Side Effects:
            更新 request ID 映射、CPU token 行、长度数组、temperature 和 block
            table；消费被新增请求或单向移动覆盖的空槽位，并将未覆盖的尾部删除
            槽位保留在 `BatchUpdateBuilder` 中，供 logits processor 在本轮同步。
        """
        num_reqs = self.num_reqs
        # 首次读取会将删除槽位降序排序；之后本轮不能再删除 request，因此调用方必须
        # 保证所有 remove_request() 已完成，再进入 condense()。
        if not (empty_req_indices := self.batch_update_builder.removed):
            # 所有删除槽位都可能刚好被本轮的新请求复用，无空洞时无需搬运。
            return
        # 当前 batch 已无活跃 request 时直接清空 Python 列表。
        if num_reqs == 0:
            self._req_ids.clear()
            self.req_output_token_ids.clear()
            return

        # 空槽位按降序排列；原 batch 逻辑长度等于活跃数加剩余空槽位数。
        last_req_index = num_reqs + len(empty_req_indices) - 1

        while empty_req_indices:
            # 跳过末尾本身就是空洞的位置；这些行会在最终截断列表时自然丢弃。
            while last_req_index in empty_req_indices:
                last_req_index -= 1

            # peek_removed() 返回最小空槽位，并保持与 builder 的排序状态一致。
            empty_index = self.batch_update_builder.peek_removed()
            assert empty_index is not None
            if empty_index >= last_req_index:
                break

            # 消费最小空洞，并将末尾活跃 request 单向移动到这里。
            popped_empty_index = self.batch_update_builder.pop_removed()
            assert popped_empty_index == empty_index
            req_id = self._req_ids[last_req_index]
            output_token_ids = self.req_output_token_ids[last_req_index]
            assert req_id is not None
            assert output_token_ids is not None

            # 当前 `_get_active_token_count()` 依赖源槽位的 output token 引用；必须
            # 在将该槽位清空为 None 前计算有效前缀长度。
            num_tokens = self._get_active_token_count(last_req_index)

            self._req_ids[empty_index] = req_id
            self._req_ids[last_req_index] = None
            self.req_output_token_ids[empty_index] = output_token_ids
            self.req_output_token_ids[last_req_index] = None
            self.req_id_to_index[req_id] = empty_index

            # 只移动该 request 的有效 token 前缀，而不是整行 max_model_len。
            self.token_ids_cpu[empty_index, :num_tokens] = self.token_ids_cpu[
                last_req_index,
                :num_tokens,
            ]

            # 所有 per-request 标量状态必须随 token 行一起移动。
            self.num_prompt_tokens[empty_index] = self.num_prompt_tokens[
                last_req_index
            ]
            self.num_computed_tokens_cpu[empty_index] = (
                self.num_computed_tokens_cpu[last_req_index]
            )
            self.temperature_cpu[empty_index] = self.temperature_cpu[
                last_req_index
            ]

            # block table 也是按 req_index 组织，必须同步移动。
            self.block_table.move_row(last_req_index, empty_index)
            # source 已不属于活跃前缀，清除其长度和页表，防止扩展 Graph 槽位时
            # 把搬走的请求当成仍然存在；只清 CPU staging，GPU 由 Runner 提交。
            self.block_table.clear_row(last_req_index)
            self.num_computed_tokens_cpu[last_req_index] = 0
            self.num_prompt_tokens[last_req_index] = 0

            # 目标槽位的旧请求已被新移动的 request 覆盖，因此 builder 不再将该
            # 槽位作为 removed 交给 processor；processor 只需同步这一条移动事件。
            self.batch_update_builder.moved.append(
                (
                    last_req_index,
                    empty_index,
                    MoveDirectionality.UNIDIRECTIONAL,
                )
            )

            # 最后一个槽位已经变成空洞，继续向前寻找下一个活跃 request。
            last_req_index -= 1

        # 槽位压缩后只保留活跃 request 的前缀，尾部 None 不再属于当前 batch。
        del self._req_ids[num_reqs:]
        del self.req_output_token_ids[num_reqs:]

    def refresh_metadata(self) -> None:
        """将当前活跃 request 的采样 metadata 从 CPU staging buffer 同步到 GPU。

        vLLM 原版在此构造 ``SamplingMetadata``、更新 logits processors、惩罚项、
        grammar mask 和 speculative decoding 状态。当前 MVP 先完整保留
        `BatchUpdateBuilder -> LogitsProcessor.update_state()` 的状态同步边界，再只
        同步 temperature，并构造只保留已支持字段的 SamplingMetadata。

        调用时机必须在 ``condense()`` 后，此时活跃 request 才连续位于前
        ``num_reqs`` 个物理槽位。

        Raises:
            AssertionError: batch 仍有空洞时抛出。
        """
        num_reqs = self.num_reqs
        assert len(self._req_ids) == num_reqs

        # 先交付并重置本轮布局变化。即使 num_reqs 为零，也必须执行这一步，才能
        # 让 processor 清理最后一批已完成请求的状态，并允许 builder 开始下一轮。
        batch_update = self.batch_update_builder.get_and_reset(num_reqs)
        for logitsproc in self.logitsprocs.all:
            logitsproc.update_state(batch_update)

        if num_reqs == 0:
            # 最后一条请求移除后，不能继续保留上一轮非空的 temperature view。
            self.sampling_metadata = self._make_sampling_metadata()
            return

        # pinned CPU memory 可与默认 CUDA stream 异步复制；未 pin 时 PyTorch 自动退化。
        self.temperature[:num_reqs].copy_(
            self.temperature_cpu_tensor[:num_reqs],
            non_blocking=self.pin_memory,
        )
        self.sampling_metadata = self._make_sampling_metadata()

    def _make_sampling_metadata(self) -> SamplingMetadata:
        """以当前紧凑槽位顺序构造采样参数，复用设备端 temperature buffer。

        不复制 temperature 或 processor 容器；下一轮复用 buffer 前，本轮采样
        必须已经完成。零请求时返回合法的空 temperature view。
        """
        return SamplingMetadata(
            temperature=self.temperature[:self.num_reqs],
            logitsprocs=self.logitsprocs,
        )
