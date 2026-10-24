"""供 Attention kernel 使用的运行时 block table 缓冲区。

Scheduler 与 KVCacheManager 操作的是“KV 管理器 block ID”；Attention kernel
操作的则是物理 kernel page ID。BlockTable 是 Worker 内部这两种表示之间的
适配器。它不分配、不释放 KV cache，只维护每次模型执行前提交到 GPU 的稠密表。
"""

import numpy as np
import torch
import triton
import triton.language as tl

from nanovllm.utils import CpuGpuBuffer
from nanovllm.attention.backends.utils import NULL_BLOCK_ID


# Attention 的 store kernel 约定负 slot 表示该 token 没有实际 KV 写入位置。
PAD_SLOT_ID = -1


class BlockTable:
    """维护 CPU/GPU block table 与每个 packed token 的 slot mapping。

    ``InputBatch`` 中的一个 request slot 对应 table 的一行；该行有效前缀存储该
    request 的物理 kernel page ID。普通情况下，一个 KVCacheManager block 正好
    对应一个 kernel page。若 attention kernel 要求更小的 page，则一个 manager
    block 会被展开为多个连续的 kernel page。

    Args:
        block_size: KVCacheManager 分配的一个 manager block 可容纳的 token 数。
        max_num_reqs: InputBatch 同时容纳的最大 request 数，即 table 最大行数。
        max_num_blocks_per_req: 单个 request 最多使用的 manager block 数。
        max_num_batched_tokens: 一次模型执行最多的 packed query token 数，同时也是
            ``slot_mapping`` 的容量。
        pin_memory: CPU staging tensor 是否使用 pinned memory。
        device: Attention kernel 读取 table 所在的设备。
        kernel_block_size: Attention kernel 所要求的物理 page token 容量。当它与
            ``block_size`` 不同时，必须整除 ``block_size``。

    Attributes:
        block_size: 内部使用的 kernel page 大小，单位为 token。slot mapping 和
            attention kernel 均以它为单位。
        blocks_per_kv_block: 一个 manager block 展开成的 kernel page 数；普通布局
            中它等于 1。
        max_num_blocks_per_req: 一行最多可放的 kernel page 数，不是 manager block
            数。
        block_table: 形状为 ``[max_num_reqs, max_num_blocks_per_req]`` 的 CPU/GPU
            双缓冲，元素为 kernel page ID。
        num_blocks_per_row: CPU 数组；元素为对应行当前有效 kernel page 的数量。
        slot_mapping: GPU buffer；将本轮每个 packed query token 映射到 attention
            kernel 读写 KV cache 的线性 slot。
    """

    def __init__(
        self,
        block_size: int,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        kernel_block_size: int,
    ) -> None:
        """创建固定容量的 CPU/GPU staging buffer。"""
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.pin_memory = pin_memory
        self.device = device

        if kernel_block_size == block_size:
            # 普通布局：Scheduler 给出的 manager block ID 可直接交给 kernel。
            self.block_size = block_size
            self.blocks_per_kv_block = 1
            self.use_hybrid_blocks = False
        else:
            # Hybrid 布局：一个 manager block 要展开为多个 kernel-size page。
            if block_size % kernel_block_size != 0:
                raise ValueError(
                    f"kernel_block_size {kernel_block_size} 必须整除 "
                    f"KVCacheManager 的 block_size {block_size}"
                )

            self.block_size = kernel_block_size
            self.blocks_per_kv_block = block_size // kernel_block_size
            self.use_hybrid_blocks = True

        # Hybrid 情况下，manager block 容量会换算为更大的 kernel page 容量，因此
        # 第二维始终以 kernel page 为单位。
        self.max_num_blocks_per_req = (
            max_num_blocks_per_req * self.blocks_per_kv_block
        )

        self.block_table = self._make_buffer(
            self.max_num_reqs,
            self.max_num_blocks_per_req,
            dtype=torch.int32,
        )
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)
        self.slot_mapping = self._make_buffer(
            self.max_num_batched_tokens,
            dtype=torch.int64,
        )

        # 广播后可将 manager ID ``b`` 展开为
        # ``b * blocks_per_kv_block + [0, 1, ...]``，无需 Python 循环。
        self._kernel_block_arange = (
            np.arange(self.blocks_per_kv_block).reshape(1, -1)
            if self.use_hybrid_blocks
            else None
        )

    def append_row(self, block_ids: list[int], row_idx: int) -> None:
        """向一个 request 行追加新分配的 manager block。

        Args:
            block_ids: 按 token 位置排序的新 manager block ID。Hybrid 布局中会在
                写入前转换成 kernel page ID。
            row_idx: 接收这些 ID 的 request slot 行号。

        Returns:
            无返回值。该方法只更新 CPU staging table；GPU 要观察到新内容前，调用方
            必须调用 :meth:`commit_block_table`。
        """
        if not block_ids:
            return

        if self.use_hybrid_blocks:
            # 一个 manager block 展开为多个连续 kernel page，之后的行长度按 page 计。
            assert self._kernel_block_arange is not None
            block_ids = self.map_to_kernel_blocks(
                np.asarray(block_ids),
                self.blocks_per_kv_block,
                self._kernel_block_arange,
            ).tolist()

        num_blocks = len(block_ids)
        start = int(self.num_blocks_per_row[row_idx])  # 从旧有效前缀末尾追加，不覆盖已有页。
        end = start + num_blocks
        if end > self.max_num_blocks_per_req:
            raise ValueError(
                f"BlockTable 第 {row_idx} 行需要 {end} 个 kernel page，但容量只有 "
                f"{self.max_num_blocks_per_req}"
            )

        # 同时更新编号和长度，后续 move/commit 才能正确解释这一行。
        self.block_table.np[row_idx, start:end] = block_ids
        self.num_blocks_per_row[row_idx] = end

    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        """用新 request 的完整 manager block table 覆盖一行。

        Args:
            block_ids: 该 request 当前完整的 manager block ID 列表。
            row_idx: 空闲或即将复用的 request slot 行号。

        Returns:
            无返回值。旧行的有效长度会先清零，再追加新 ID。
        """
        # 复用较短请求行时，完整清除旧尾部，不把旧请求的 page ID 留在 padding 区。
        self.block_table.np[row_idx].fill(NULL_BLOCK_ID)
        self.num_blocks_per_row[row_idx] = 0
        self.append_row(block_ids, row_idx)

    def clear_row(self, row_idx: int) -> None:
        """在 request 离开 InputBatch 后清空其行。

        Args:
            row_idx: 即将释放并复用的 request slot 行号。

        Returns:
            无返回值。清理发生在 CPU staging table；下一次提交后 GPU 才会看到零值。
        """
        self.block_table.np[row_idx].fill(NULL_BLOCK_ID)
        self.num_blocks_per_row[row_idx] = 0

    def move_row(self, src: int, tgt: int) -> None:
        """复制一个有效行前缀到另一个行。

        该操作支持 ``InputBatch.condense``。source 行不会立刻清空，调用方可在适当
        时机覆盖或清理它，这与 vLLM 的 move 语义一致。

        Args:
            src: 源 request slot 行号。
            tgt: 目标 request slot 行号。

        Returns:
            无返回值。
        """
        num_blocks = int(self.num_blocks_per_row[src])
        self.block_table.np[tgt, :num_blocks] = self.block_table.np[
            src, :num_blocks
        ]
        # 先复制再清目标尾部，既清除旧页号，也保持 src == tgt 时的有效前缀。
        self.block_table.np[tgt, num_blocks:].fill(NULL_BLOCK_ID)
        self.num_blocks_per_row[tgt] = num_blocks

    def swap_row(self, src: int, tgt: int) -> None:
        """交换两行完整的 table 内容和有效 page 数量。

        Args:
            src: 第一行 request slot。
            tgt: 第二行 request slot。

        Returns:
            无返回值。
        """
        src_tgt, tgt_src = [src, tgt], [tgt, src]
        self.num_blocks_per_row[src_tgt] = self.num_blocks_per_row[tgt_src]
        self.block_table.np[src_tgt] = self.block_table.np[tgt_src]

    def compute_slot_mapping(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """计算本轮每个 packed query token 的物理 KV cache slot。

        ``query_start_loc`` 将 packed ``positions`` 按 request 划分。逻辑位置为
        ``p`` 的 token 会从对应 request 行中取得 kernel page ID，并得到：
        ``kernel_page_id * kernel_block_size + p % kernel_block_size``。

        实现沿用 vLLM 的 Triton 组织方式：前 ``num_reqs`` 个 program 各处理一个
        request，最后一个 program 将固定容量 buffer 的无效后缀填为 ``-1``。本地
        MVP 没有 CP/DCP，因此不需要 vLLM kernel 中的并行 rank 与交错 page 分支。

        Args:
            num_reqs: 本轮位于 ``block_table`` 前部的活跃 request 行数。
            query_start_loc: 形状为 ``[num_reqs + 1]`` 的设备 tensor；相邻元素之差
                是对应 request 在本轮的 packed query token 数。
            positions: 形状为 ``[num_query_tokens]`` 的设备 tensor；元素是 token 在
                自身 request 内的逻辑位置。

        Returns:
            无返回值。有效 slot 写入 ``slot_mapping.gpu`` 前缀，固定容量后缀填为
            ``PAD_SLOT_ID``。
        """
        num_tokens = positions.numel()  # 真实 packed query 数，不包含固定缓冲区尾部。
        if num_tokens > self.max_num_batched_tokens:
            raise ValueError(
                f"本轮有 {num_tokens} 个 token，超过 slot_mapping 容量 "
                f"{self.max_num_batched_tokens}"
            )
        # 每个请求需要起点，最后还需一个终点，才能在 kernel 中切出各自区间。
        if query_start_loc.numel() != num_reqs + 1:
            raise ValueError(
                "query_start_loc 必须包含每个 request 的起点和最后一个结束位置"
            )
        if positions.device != self.device or query_start_loc.device != self.device:
            raise ValueError(
                "positions 和 query_start_loc 必须位于 BlockTable.device"
            )

        # 多出的一个 program 负责把无效后缀写为 -1，避免上轮 slot 被 padding 误用。
        _compute_slot_mapping_kernel[(num_reqs + 1,)](
            num_tokens,
            self.max_num_batched_tokens,
            query_start_loc,
            positions,
            self.block_table.gpu,
            self.block_table.gpu.stride(0),
            self.block_size,
            self.slot_mapping.gpu,
            PAD_SLOT_ID=PAD_SLOT_ID,
            BLOCK_SIZE=1024,
        )

    def commit_block_table(self, num_reqs: int) -> None:
        """将活跃 CPU block-table 行提交到 GPU。

        Args:
            num_reqs: 从第零行起连续活跃的 request 行数。

        Returns:
            无返回值。
        """
        self.block_table.copy_to_gpu(num_reqs)

    def clear(self) -> None:
        """将整个 CPU/GPU block table 重置为零。

        这是 Worker 生命周期级别的重置辅助接口。正常 request 完成时应使用
        :meth:`clear_row`，以免影响其他活跃行。
        """
        self.block_table.gpu.fill_(NULL_BLOCK_ID)
        self.block_table.cpu.fill_(NULL_BLOCK_ID)
        self.num_blocks_per_row.fill(0)

    @staticmethod
    def map_to_kernel_blocks(
        kv_manager_block_ids: np.ndarray,
        blocks_per_kv_block: int,
        kernel_block_arange: np.ndarray,
    ) -> np.ndarray:
        """将 manager block ID 展开为连续的 kernel page ID。

        Args:
            kv_manager_block_ids: 一维 manager block ID 数组，例如 ``[10, 3]``。
            blocks_per_kv_block: 每个 manager block 包含的 kernel page 数。例如
                manager block 为 8 token、kernel page 为 4 token 时等于 ``2``。
            kernel_block_arange: 用于组成 page offset 的可广播行
                ``[[0, ..., k - 1]]``，由 BlockTable 缓存。

        Returns:
            kernel_block_ids: 展平后的 kernel page ID 数组。若输入 ``[10, 3]`` 且每个 manager block
                包含两个 kernel page，返回 ``[20, 21, 6, 7]``。
        """
        if blocks_per_kv_block == 1:
            return kv_manager_block_ids

        kernel_block_ids = (
            kv_manager_block_ids.reshape(-1, 1) * blocks_per_kv_block
            + kernel_block_arange
        )
        return kernel_block_ids.reshape(-1)

    def get_device_tensor(self, num_reqs: int) -> torch.Tensor:
        """返回供 attention kernel 使用的活跃 GPU 行 view。

        Args:
            num_reqs: 要暴露的活跃行数。

        Returns:
            block_table_tensor: 形状为 ``[num_reqs, max_num_blocks_per_req]`` 的 view，不复制数据。
        """
        return self.block_table.gpu[:num_reqs]

    def get_cpu_tensor(self) -> torch.Tensor:
        """返回完整的 CPU kernel-page staging tensor。

        Returns:
            cpu_tensor: 完整 CPU kernel-page staging tensor，与内部 buffer 共享存储。
        """
        return self.block_table.cpu

    def get_numpy_array(self) -> np.ndarray:
        """返回用于低开销 host 行更新的 NumPy view。

        Returns:
            numpy_array: CPU 页表对应的 NumPy view，可用于低开销 host 行更新。
        """
        return self.block_table.np

    def _make_buffer(
        self,
        *size: int | torch.SymInt,
        dtype: torch.dtype,
    ) -> CpuGpuBuffer:
        """分配一个 CPU/GPU 双缓冲。

        Args:
            *size: tensor 的各维长度。
            dtype: CPU 与 GPU tensor 共用的元素类型。

        Returns:
            buffer: 带 CPU、GPU 与 NumPy view 的可变 runtime buffer。
        """
        return CpuGpuBuffer(
            *size,
            dtype=dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        )


@triton.jit
def _compute_slot_mapping_kernel(
    num_tokens,
    max_num_tokens,
    query_start_loc_ptr,
    positions_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    slot_mapping_ptr,
    PAD_SLOT_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """为 packed token 计算物理 KV cache slot。

    调用 grid 为 ``(num_reqs + 1,)``。前 ``num_reqs`` 个 program 的 ID 与
    request 在 GPU block table 中的行号一一对应：program ``r`` 处理
    ``[query_start_loc[r], query_start_loc[r + 1])`` 这个 packed token 区间。
    最后一个 program 不处理 request，只将固定容量 ``slot_mapping`` 中的无效后缀
    写为 ``PAD_SLOT_ID``。

    对一个逻辑位置 ``position``，最终地址为：

    ``page_id * block_size + position % block_size``。

    其中 ``page_id`` 来自 request 对应的 block-table 行：
    ``block_table[req_idx, position // block_size]``。

    Args:
        num_tokens: 本轮实际 packed token 总数，即 ``positions_ptr`` 的有效长度。
        max_num_tokens: ``slot_mapping_ptr`` 的固定容量。它可大于 ``num_tokens``，
            以支持 CUDA Graph 使用稳定的 tensor 形状。
        query_start_loc_ptr: int32 指针，形状为 ``[num_reqs + 1]``。相邻两个值给出
            一个 request 在 packed 输入中的半开区间。
        positions_ptr: int64 指针，形状为 ``[num_tokens]``。每项是 token 在自身
            request 内的逻辑位置，而不是 packed 下标。
        block_table_ptr: int32 指针，指向逻辑形状为
            ``[max_num_reqs, max_num_blocks_per_req]`` 的 GPU kernel-page-ID 表。
            Triton 按一维指针寻址，因此使用 ``block_table_stride`` 定位行。
        block_table_stride: 相邻两行开头之间的元素数，通常等于
            ``max_num_blocks_per_req``；单位是 int32 元素，不是字节。
        block_size: 一个 kernel page 容纳的 token 数。
        slot_mapping_ptr: int64 指针，指向形状为 ``[max_num_tokens]`` 的输出 buffer。
        PAD_SLOT_ID: 无效 slot 的固定值，当前为 ``-1``。store KV cache 时会跳过它。
        BLOCK_SIZE: 一个 Triton program 每轮向量化处理的 token 数，当前为 ``1024``。

    Returns:
        无返回值。结果直接写入 ``slot_mapping_ptr`` 指向的 GPU buffer。

    Preconditions:
        ``block_table_ptr`` 中前 ``num_reqs`` 行已由 ``commit_block_table`` 提交到
        GPU；每个 position 访问的 page 下标必须落在对应行的有效前缀内。
    """
    req_idx = tl.program_id(0)

    if req_idx == tl.num_programs(0) - 1:
        # 最后一个 program 只做 padding：维持固定大小的输出 buffer，方便后续
        # CUDA Graph 重放时复用相同形状。mask 防止最后一轮越过总容量。
        for i in range(num_tokens, max_num_tokens, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(
                slot_mapping_ptr + offsets,
                PAD_SLOT_ID,
                mask=offsets < max_num_tokens,
        )
        return

    # 当前 program 的 ID 就是 request 的 block-table 行号；两个相邻 prefix sum
    # 元素定义该 request 的 packed token 半开区间 [start_idx, end_idx)。
    start_idx = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int64)
    # block_table_ptr 在 Triton 中是一维地址。乘以“每行元素数”后得到本行起点。
    row_offset = req_idx * block_table_stride

    for i in range(start_idx, end_idx, BLOCK_SIZE):
        # 一个向量 lane 对应本 request 的一个 packed token。最后不足 BLOCK_SIZE 个
        # token 的 lane 由 mask 遮住；masked load 的 other=0 只给无效 lane 占位。
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < end_idx
        positions = tl.load(positions_ptr + offsets, mask=mask, other=0)

        # position // block_size 是本 request 的逻辑 kernel-page 下标。加上行起点
        # 后可从扁平地址中取出该 page 的物理 page ID。
        page_indices = positions // block_size
        page_ids = tl.load(block_table_ptr + row_offset + page_indices).to(tl.int64)

        # page 内偏移与物理 page ID 合并成该层扁平 KV cache 的精确 token slot。
        page_offsets = positions - page_indices * block_size
        slot_ids = page_ids * block_size + page_offsets
        tl.store(slot_mapping_ptr + offsets, slot_ids, mask=mask)
