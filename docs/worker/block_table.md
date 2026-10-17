# BlockTable：从 Manager Block Table 到 Kernel Page Table

## 最重要的结论

BlockTable **不分配、不释放、也不拥有** KV cache block。

KVCacheManager 管理 request 的 manager block table；Worker 侧的 BlockTable 接收它，
并将其中的 manager block ID 转换为 Attention kernel 真正能够访问的 kernel page ID。

因此两张表的关系是：

~~~text
KVCacheManager 的 manager block table
    request 逻辑 manager block 下标 -> manager block ID

                 经过 BlockTable 的页粒度转换

Worker BlockTable 的 kernel page table
    request 逻辑 kernel page 下标 -> kernel page ID
~~~

当前本地实现位于 [block_table.py](/home/huangxy/Projects/my-nano-vllm/nanovllm/worker/block_table.py)。它已具备单进程所需的行更新、manager block 到 kernel page 的转换、CPU/GPU 提交和 slot mapping 计算；本文档解释这些接口必须维持的语义。

## 为什么需要转换

KVCacheManager 的分配粒度，和 Attention backend 的访问粒度不一定相同。

| 层次 | 谁使用 | 单位大小 | 主要职责 |
| --- | --- | --- | --- |
| manager block | KVCacheManager、Scheduler | block_size 个 token | 分配、共享、引用计数、抢占、prefix cache。 |
| kernel page | Attention backend | kernel_block_size 个 token | paged attention 读取历史 K/V、store_kvcache 写入新 K/V。 |

例如：

~~~text
KVCacheManager block_size = 8
Attention kernel_block_size = 4
~~~

KVCacheManager 认为一个 manager block 能覆盖 8 个 token；Attention kernel 只认识
4-token page。于是一个 manager block 必须拆成两个 kernel page：

~~~text
一个 manager block
token 范围:       [0, 1, 2, 3, 4, 5, 6, 7]

拆为 kernel page:
第 0 个 page:     [0, 1, 2, 3]
第 1 个 page:     [4, 5, 6, 7]
~~~

这是同一段 KV memory 的两种粒度视角，不是两套相互独立的 cache。

## 两类 block ID

### Manager block ID

Manager block ID 是 KVCacheManager 分配、引用计数和释放的资源编号。

假设 request A 的 manager block table 是：

~~~text
逻辑 manager block 下标:  0   1
manager block ID:          10  3
~~~

这意味着：

~~~text
A 的 token 0..7   使用 manager block 10
A 的 token 8..15  使用 manager block 3
~~~

KVCacheManager 只需要知道 10、3：它据此维护 block 的 ref_cnt、prefix hash 和空闲队列。
它不需要知道 Attention kernel 如何把每个 block 拆成 page。

### Kernel page ID

Kernel page ID 是 Attention backend 实际用于寻址 K/V cache 的编号。vLLM 的转换约定是：

~~~text
blocks_per_kv_block = block_size // kernel_block_size

kernel_page_id = manager_block_id * blocks_per_kv_block + page_offset_in_manager_block
~~~

上例中：

~~~text
block_size = 8
kernel_block_size = 4
blocks_per_kv_block = 2

manager block 10 -> kernel pages [20, 21]
manager block 3  -> kernel pages [6, 7]
~~~

所以 Worker BlockTable 中真正写入 GPU 的 row 是：

~~~text
输入：KVCacheManager block_ids = [10, 3]

输出：kernel page table row = [20, 21, 6, 7]
~~~

Attention kernel 只会看到 20、21、6、7。它不直接读取 KVCacheManager 的 10、3。

### `_kernel_block_arange`：展开时使用的 page 内偏移行

`_kernel_block_arange` 只在 `use_hybrid_blocks == True` 时存在。它不是 block table，
也不是 GPU tensor；它只是一个常驻 CPU 的 NumPy 小数组，用来让一次广播计算完成所有
manager block 到 kernel page 的展开。

仍使用前面的例子：manager block 为 8 token，kernel page 为 4 token。因此：

~~~python
blocks_per_kv_block = 8 // 4  # 2
_kernel_block_arange = np.arange(2).reshape(1, -1)
# 形状是 (1, 2)，内容为 [[0, 1]]
~~~

`map_to_kernel_blocks()` 接收的一维 manager block ID 为 `[10, 3]`。它先把这一行变成
形状 `(2, 1)` 的列，再与 `(1, 2)` 的偏移行广播相加：

~~~python
kv_manager_block_ids.reshape(-1, 1) * blocks_per_kv_block
# [[20],
#  [ 6]]

kernel_block_ids = (
    kv_manager_block_ids.reshape(-1, 1) * blocks_per_kv_block
    + _kernel_block_arange
)
# [[20, 21],
#  [ 6,  7]]

kernel_block_ids.reshape(-1)
# [20, 21, 6, 7]
~~~

这里 `[0, 1]` 的每个值表示“同一个 manager block 内的第几个 kernel page”。因此，
它的长度永远是 `blocks_per_kv_block`，形状固定为 `(1, blocks_per_kv_block)`：

- manager block ID 为 `b` 时，结果是 `b * k + [0, 1, ..., k - 1]`。
- `reshape(1, -1)` 很重要：它让 NumPy 把偏移理解为“对每一行都重复一次”的横向向量。
- 普通布局中 `k == 1`，转换就是恒等，代码不会创建这个数组，也不会进入广播路径。

## 三种下标与一个最终地址

对 request A 的 manager table [10, 3]、kernel page table [20, 21, 6, 7]：

| 名称 | 位置 9 的值 | 计算方式 | 含义 |
| --- | --- | --- | --- |
| manager block 下标 | 1 | 9 // 8 | 位置 9 落在第几个 manager block。 |
| kernel page 下标 | 2 | 9 // 4 | 位置 9 落在 Worker BlockTable 的第几列。 |
| kernel page ID | 6 | table[row, 2] | Attention 要访问的实际 page 编号。 |
| page 内 offset | 1 | 9 % 4 | 位置 9 在 page 6 内的偏移。 |
| slot | 25 | 6 * 4 + 1 | 扁平 K/V cache 中精确 token 地址。 |

关键点是：manager block 下标和 kernel page 下标在 hybrid 模式中**不相同**。

~~~text
position 9:

manager block 下标 = 9 // 8 = 1
kernel page 下标  = 9 // 4 = 2
~~~

因此，GPU BlockTable 的列索引永远应以 kernel_block_size 为准，而不是 manager block_size。

## BlockTable 的成员到底维护什么

### 容量成员

| 成员 | 数的是谁 | 生命周期 | 不代表什么 |
| --- | --- | --- | --- |
| max_num_reqs | 持久 InputBatch 可容纳的 request 行数。 | BlockTable 生命周期内固定。 | 不是当前活跃 request 数。 |
| 传入的 max_num_blocks_per_req | 一个 request 最多跨多少 manager block。通常是 ceil(max_model_len / block_size)。 | 构造输入。 | 不是 GPU table 最终列数，hybrid 时还要展开。 |
| blocks_per_kv_block | 一个 manager block 拆成多少 kernel page。等于 block_size // kernel_block_size。 | 构造后固定。 | 不是一个 request 的 page 数。 |
| 内部 max_num_blocks_per_req | GPU BlockTable 一行最多容纳多少 kernel page ID。等于输入 manager 容量乘 blocks_per_kv_block。 | 构造后固定。 | 不是当前行的有效 page 数。 |
| max_num_batched_tokens | 一次 packed model forward 最多执行多少 token。决定 slot_mapping buffer 长度。 | BlockTable 生命周期内固定。 | 不是 max_model_len，也不是 block 数。 |
| 构造参数 block_size | KVCacheManager 的 manager block 粒度。 | 构造输入。 | 不是内部 self.block_size 的最终语义。 |
| 构造参数 kernel_block_size | Attention backend 的 page 粒度。 | 构造输入。 | 不是 KV cache 总大小。 |
| 内部 self.block_size | Attention kernel page 的粒度，即 kernel_block_size。 | 构造后固定；slot 计算按它进行。 | 不是 manager block 大小。 |
| pin_memory | CPU staging table 是否锁页。 | 构造后固定。 | 不改变任何 block/page ID。 |
| device | GPU table 与 slot_mapping 所在设备。 | 构造后固定。 | 不表示 KVCacheManager 的所有权。 |

下面是完整的数量推导：

~~~text
max_model_len = 20
manager block_size = 8
kernel_block_size = 4

输入 max_num_blocks_per_req
    = ceil(20 / 8)
    = 3 个 manager block

blocks_per_kv_block
    = 8 / 4
    = 2 个 kernel page / manager block

GPU table 内部行容量
    = 3 * 2
    = 6 个 kernel page ID

max_num_batched_tokens = 8
    = 单轮 packed forward 最多 8 个 token
    = slot_mapping 最多 8 个地址
~~~

这四个“数量”分别在数：最长上下文的 manager block、manager block 内的 page、GPU 页表列、
以及本轮 token；不能互相替代。

### 动态状态成员

| 成员 | 推荐类型 / shape | 保存什么 | 谁修改 |
| --- | --- | --- | --- |
| block_table.cpu | torch.int32[max_num_reqs, effective_max_num_kernel_pages_per_req] | 可由 Python/NumPy 编辑的 kernel page table。 | add_row、append_row、clear_row、move_row、swap_row。 |
| block_table.gpu | 同 shape 的 CUDA tensor | 已提交给 Attention kernel 的 kernel page table。 | commit_block_table。 |
| block_table.np | block_table.cpu 的 NumPy view | CPU 行表的便捷写入 view。 | 与 CPU tensor 共享内存。 |
| num_blocks_per_row | numpy.int32[max_num_reqs] | 每一行当前有效的 kernel page 数。 | 所有行修改接口。 |
| slot_mapping.gpu | torch.int64[max_num_batched_tokens] | 当前 packed forward 中每个 token 的最终 KV 写入地址。 | compute_slot_mapping，每轮重算。 |

num_blocks_per_row 始终数 **kernel page**，不是 manager block。因为 GPU table 本身存的是转换后的 page ID。

固定宽度表不能通过值为 0 判断行尾：kernel page ID 0 也是合法 ID。例如：

~~~text
CPU row 0 = [20, 21, 6, 7, 0, 0]
num_blocks_per_row[0] = 4
~~~

这里有效内容只是 [20, 21, 6, 7]；尾部的 0 是未使用容量。

## slot 与 slot_mapping

### slot 是什么

一个 kernel page 包含 kernel_block_size 个 token。将 KV cache 的前两维概念性地展平：

~~~text
KV cache 的概念 shape:
[num_kernel_pages, kernel_block_size, num_kv_heads, head_dim]

slot = kernel_page_id * kernel_block_size + page_offset
~~~

因此 slot 代表一个 token 的精确 K/V 位置，概念上等价于：

~~~python
k_cache[kernel_page_id, page_offset]
v_cache[kernel_page_id, page_offset]
~~~

当前 store_kvcache Triton kernel 使用 slot 计算连续内存地址。slot 不是 request 内 position，
也不是 manager block ID；它是“某层 K/V cache 中的 token 格子编号”。

每一层 Attention 有各自的 K/V tensor，但同一个 request 的同一个 token 在所有层使用相同
slot_mapping。原因是每一层的页组织一致，只是各层写入不同的 K/V 向量。

### slot_mapping 是什么

模型前向会把多个 request 本轮要执行的 token 拼成 packed tensor。slot_mapping 是一个
按 packed token 下标索引的数组：

~~~text
slot_mapping[packed_token_index] = 该 token 的最终 KV slot
~~~

假设本轮执行：

~~~text
A 的 position 5、8
C 的 position 4

packed token 下标:  0  1  2
所属 request:       A  A  C
逻辑 position:      5  8  4
~~~

继续使用 A 的 kernel page table [20, 21, 6, 7]。若 C 的 kernel page table 为 [14, 15]：

| packed 下标 | request | position | kernel page 下标 | kernel page ID | offset | slot |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | A | 5 | 5 // 4 = 1 | A_table[1] = 21 | 1 | 21 * 4 + 1 = 85 |
| 1 | A | 8 | 8 // 4 = 2 | A_table[2] = 6 | 0 | 6 * 4 + 0 = 24 |
| 2 | C | 4 | 4 // 4 = 1 | C_table[1] = 15 | 0 | 15 * 4 + 0 = 60 |

所以：

~~~text
slot_mapping = [85, 24, 60]

K[0], V[0] 写入 slot 85
K[1], V[1] 写入 slot 24
K[2], V[2] 写入 slot 60
~~~

BlockTable 与 slot_mapping 的粒度不同：

| 对象 | 一个元素对应什么 | 更新时机 |
| --- | --- | --- |
| BlockTable 的一行 | 一个 request 的所有 kernel page ID。 | request 加入、扩容、删除或重排时。 |
| slot_mapping 的一个元素 | 本轮 packed 输入中的一个 token。 | 每一个 model forward。 |

读历史 KV 时，Attention backend 使用 BlockTable，因为它需要访问一个 request 的多页历史。
写本轮 K/V 时，store_kvcache 使用 slot_mapping，因为它只关心 packed K/V 的每一行写到哪一个精确 token 格子。

未来 CUDA Graph 的 padding token 可以使用 slot = -1；store_kvcache 遇到 -1 应跳过写入。

## 接口：输入、输出与状态变化

以下样例使用：

~~~text
block_size = 8
kernel_block_size = 4
blocks_per_kv_block = 2
max_num_reqs = 3
输入 max_num_blocks_per_req = 3
内部 GPU table 列容量 = 3 * 2 = 6
max_num_batched_tokens = 8
~~~

### add_row：完整 manager block table 转换并覆盖一行

~~~python
def add_row(self, block_ids: list[int], row_idx: int) -> None: ...
~~~

输入 block_ids 必须是 KVCacheManager 给出的**完整 manager block ID 列表**。新 request 进入
InputBatch，或被抢占 request 恢复时调用。

调用前：

~~~text
num_blocks_per_row = [0, 0, 0]
CPU row 0 = [0, 0, 0, 0, 0, 0]
~~~

调用：

~~~python
block_table.add_row(block_ids=[10, 3], row_idx=0)
~~~

内部变化：

~~~text
输入 manager IDs: [10, 3]
转换 kernel IDs:  [20, 21, 6, 7]
~~~

调用后：

~~~text
num_blocks_per_row = [4, 0, 0]
CPU row 0 = [20, 21, 6, 7, 0, 0]
~~~

add_row 必须先将目标 row 的有效长度设为 0，再调用 append_row。尾部旧值不必全部清零，
因为只有前 num_blocks_per_row[row_idx] 个 kernel page ID 有效。

### append_row：追加新的 manager block，并展开为 kernel page

~~~python
def append_row(self, block_ids: list[int], row_idx: int) -> None: ...
~~~

running request 跨过 manager block 边界时，KVCacheManager 分配新的 manager block。假设
A 原来是 [10, 3]，本轮新分配 manager block 7：

~~~python
block_table.append_row(block_ids=[7], row_idx=0)
~~~

调用前：

~~~text
num_blocks_per_row[0] = 4
CPU row 0 = [20, 21, 6, 7, 0, 0]
~~~

调用后的转换：

~~~text
manager ID 7 -> kernel pages [14, 15]
~~~

最终：

~~~text
num_blocks_per_row[0] = 6
CPU row 0 = [20, 21, 6, 7, 14, 15]
~~~

append_row 的输入仍是 manager ID；CPU/GPU row 存储的是 kernel page ID。空列表是无操作，
追加后不能超过内部有效行容量。

### clear_row：取消 Worker 对旧 request 页表的引用

~~~python
def clear_row(self, row_idx: int) -> None: ...
~~~

request 完成、被抢占或移出 InputBatch 时调用。例如清空上面的 row 0：

~~~text
调用前:
num_blocks_per_row[0] = 6
CPU row 0 = [20, 21, 6, 7, 14, 15]

调用后:
num_blocks_per_row[0] = 0
CPU row 0 = [0, 0, 0, 0, 0, 0]  # 推荐清掉原有效前缀
~~~

它不释放 manager block 10、3、7，也不释放对应 kernel page。真正资源生命周期只属于
KVCacheManager；BlockTable 只是不再让 Worker row 0 指向旧 request。

### move_row：condense 时移动转换后的 kernel page row

~~~python
def move_row(self, src: int, tgt: int) -> None: ...
~~~

假设 row 1 空出、row 2 的 C 仍存活：

~~~text
num_blocks_per_row = [6, 0, 2]
row 0: A -> [20, 21, 6, 7, 14, 15]
row 1: 空
row 2: C -> [30, 31]
~~~

调用：

~~~python
block_table.move_row(src=2, tgt=1)
~~~

调用后：

~~~text
num_blocks_per_row = [6, 2, 2]
row 1: C -> [30, 31]
row 2: 旧 C 数据允许残留
~~~

move_row 复制的是已经转换后的 kernel page ID，不再次转换。InputBatch 还必须同步移动
request ID、token 行、长度、采样参数和 BatchUpdateBuilder 的移动事件。

### swap_row：交换两个活跃 request 的 kernel page row

~~~python
def swap_row(self, src: int, tgt: int) -> None: ...
~~~

它交换两行 kernel page ID 和对应的 num_blocks_per_row。调用者必须同步交换 InputBatch
中所有其他按 req_index 排列的状态，并记录 SWAP 事件。

### commit_block_table：CPU kernel page table 提交到 GPU

~~~python
def commit_block_table(self, num_reqs: int) -> None: ...
~~~

CPU 行表的 add、append、clear、move、swap 都完成且 InputBatch.condense() 结束后调用。
它只复制活跃前缀行：

~~~python
self.block_table.gpu[:num_reqs].copy_(
    self.block_table.cpu[:num_reqs],
    non_blocking=True,
)
~~~

GPU table 未提交前仍可能保存上一轮的 kernel page ID。pin_memory 只影响这次 CPU 到 GPU
传输是否能异步，不改变表的语义。

### get_device_tensor：Attention 使用的 kernel page table view

~~~python
def get_device_tensor(self, num_reqs: int) -> torch.Tensor: ...
~~~

返回内部 GPU table 的前 num_reqs 行 view，不复制数据。Attention backend 读取它来定位一个
request 的历史 K/V page。调用前必须先 commit_block_table。

### compute_slot_mapping：本轮 packed token 的最终地址

~~~python
def compute_slot_mapping(
    self,
    num_reqs: int,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
) -> None: ...
~~~

| 参数 | 它表示什么 |
| --- | --- |
| num_reqs | 本轮 packed batch 中活跃 request 数。 |
| query_start_loc | CUDA int32[num_reqs + 1]。第 i 个 request 的 packed token 范围是 [start[i], start[i + 1])。 |
| positions | CUDA int64[num_batched_tokens]。每个 packed token 在自身 request 中的逻辑位置。 |

它读取已提交的 GPU kernel page table，按 position // kernel_block_size 和
position % kernel_block_size 生成 slot_mapping.gpu。这个 GPU buffer 的总长度固定为
max_num_batched_tokens；前 num_batched_tokens 个元素是有效 slot，剩余后缀统一填为 -1。

当前本地实现和 vLLM 一样使用 Triton kernel：前 num_reqs 个 program 各处理一个 request，
最后一个 program 将无效后缀填为 -1。vLLM 原 kernel 额外处理 context parallelism 和
decode-context parallelism；nano-vLLM 的版本删去了这些并行 rank、交错 page 与本地性判断。

### 单卡 `_compute_slot_mapping_kernel` 的执行流程

调用方先执行 `commit_block_table(num_reqs)`，确保 CPU 侧刚更新的 page table 已复制到 GPU。
之后按下面的 grid 启动 Triton：

~~~python
_compute_slot_mapping_kernel[(num_reqs + 1,)](...)
~~~

因此 program ID 的分工非常直接：

| program ID | 工作 |
| --- | --- |
| `0` 到 `num_reqs - 1` | 每个 program 负责一个 request 的所有本轮 query token。 |
| `num_reqs` | 不处理 request；只把 `slot_mapping[num_tokens:]` 填为 `-1`。 |

后一个额外 program 的意义是保持 `slot_mapping` 的物理长度恒为
`max_num_batched_tokens`。CUDA Graph 捕获时 buffer 的形状不能随本轮 token 数变化，因此
无效后缀必须有确定值；`store_kvcache_kernel` 遇到 `-1` 会跳过写入。

对负责 request `r` 的 program，kernel 中的每轮 `BLOCK_SIZE=1024` 向量循环依次完成：

~~~python
start_idx = query_start_loc[r]
end_idx = query_start_loc[r + 1]

# offsets 是当前 1024 个 packed token 下标；mask 遮住最后不足 1024 个的 lane。
positions = positions_ptr[offsets]

page_indices = positions // block_size
page_ids = block_table[r, page_indices]
page_offsets = positions % block_size
slot_ids = page_ids * block_size + page_offsets

slot_mapping[offsets] = slot_ids
~~~

这里的 `block_size` 已是 `BlockTable` 内部的 **kernel page 大小**。在当前单卡版本中，
一个逻辑位置只会落到本地唯一的 page，因此不需要 vLLM CP/DCP 版本里的 virtual block、
`is_local` 判断和跨 rank 的 offset 重排。

### 两个 request 的完整例子

下面给出一次实际 kernel launch 的完整输入。设 Worker 的 block table 总容量为
`max_num_reqs=4`、每个 request 最多使用 `max_num_blocks_per_req=4` 个 kernel page；
本轮仅前两个 request 活跃。

~~~python
num_reqs = 2
num_tokens = 5
max_num_tokens = 8
block_size = 4
PAD_SLOT_ID = -1
BLOCK_SIZE = 1024

# query_start_loc 的长度必须是 num_reqs + 1。
# request 0 使用 packed 下标 [0, 2)，request 1 使用 [2, 5)。
query_start_loc = torch.tensor(
    [0, 2, 5],
    dtype=torch.int32,
    device="cuda",
)

# 五个 packed token 在各自 request 内的逻辑位置。
# request 0 在本轮前已经计算 4 个 token，本轮连续计算位置 [4, 6)。
# request 1 在本轮前已经计算 3 个 token，本轮连续计算位置 [3, 6)。
positions = torch.tensor(
    [4, 5, 3, 4, 5],
    dtype=torch.int64,
    device="cuda",
)

# 这就是 block_table_ptr 指向的完整 GPU tensor，而不是仅有效前缀。
# 每一行的有效长度由 CPU 侧 num_blocks_per_row 维护：本例前两行都是 2。
block_table = torch.tensor(
    [
        [8, 2, 0, 0],  # request 0：有效 page ID 是 [8, 2]
        [9, 4, 0, 0],  # request 1：有效 page ID 是 [9, 4]
        [0, 0, 0, 0],  # 当前没有 request 使用这两行
        [0, 0, 0, 0],
    ],
    dtype=torch.int32,
    device="cuda",
)

block_table_stride = block_table.stride(0)  # 4，单位为 int32 元素
slot_mapping = torch.empty(
    max_num_tokens,
    dtype=torch.int64,
    device="cuda",
)

# 2 个 request program，加上 1 个 padding program。
grid = (num_reqs + 1,)  # (3,)

_compute_slot_mapping_kernel[grid](
    num_tokens,
    max_num_tokens,
    query_start_loc,
    positions,
    block_table,
    block_table_stride,
    block_size,
    slot_mapping,
    PAD_SLOT_ID=PAD_SLOT_ID,
    BLOCK_SIZE=BLOCK_SIZE,
)
~~~

这里所有实参与 kernel 签名的对应关系是：

| kernel 参数 | 本例实参 | 值或形状 | 用途 |
| --- | --- | --- | --- |
| `num_tokens` | `num_tokens` | `5` | 实际需要计算 slot 的 packed token 数。 |
| `max_num_tokens` | `max_num_tokens` | `8` | 固定输出 buffer 容量。 |
| `query_start_loc_ptr` | `query_start_loc` | `int32[3]`，值 `[0, 2, 5]` | 划分两个 request 的 packed 区间。 |
| `positions_ptr` | `positions` | `int64[5]`，值 `[4, 5, 3, 4, 5]` | 每个 packed token 的 request 内逻辑位置。 |
| `block_table_ptr` | `block_table` | `int32[4, 4]` | 所有 request slot 的 kernel page ID 行表。 |
| `block_table_stride` | `block_table.stride(0)` | `4` | 从一行起点到下一行起点的元素数。 |
| `block_size` | `block_size` | `4` | 一个 kernel page 的 token 容量。 |
| `slot_mapping_ptr` | `slot_mapping` | `int64[8]` | kernel 写入的固定容量输出 buffer。 |
| `PAD_SLOT_ID` | `-1` | 编译期常量 | 标记无效输出后缀。 |
| `BLOCK_SIZE` | `1024` | 编译期常量 | 每个 program 每轮处理的向量 lane 数。 |

`grid=(3,)` 让三个 program 分别处理：

| program ID | 读取的区间 | 工作 |
| --- | --- | --- |
| `0` | `query_start_loc[0:2]`，即 packed `[0, 2)` | 计算 request 0 的两个 token。 |
| `1` | `query_start_loc[1:3]`，即 packed `[2, 5)` | 计算 request 1 的三个 token。 |
| `2` | 不读取 request 数据 | 将 `slot_mapping[5:8]` 填为 `-1`。 |

前两个 request program 得到：

| packed 下标 | request | position | page 下标 | page ID | page 内 offset | slot |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 4 | 1 | 2 | 0 | 2 * 4 + 0 = 8 |
| 1 | 0 | 5 | 1 | 2 | 1 | 2 * 4 + 1 = 9 |
| 2 | 1 | 3 | 0 | 9 | 3 | 9 * 4 + 3 = 39 |
| 3 | 1 | 4 | 1 | 4 | 0 | 4 * 4 + 0 = 16 |
| 4 | 1 | 5 | 1 | 4 | 1 | 4 * 4 + 1 = 17 |

因此，kernel 完成后 `slot_mapping` 的**完整 GPU 输出**是：

~~~text
slot_mapping = [8, 9, 39, 16, 17, -1, -1, -1]
~~~

前五个元素由 request 0 与 request 1 的 program 写入；后三个元素由最后一个 padding
program 写入。注意 `block_table` 中每行尾部的零只是未使用容量，kernel 不会访问它们：
本轮的所有 position 分别只索引前两行的第 0 或第 1 列。slot 的数值没有按 request 连续，
也不会等于逻辑 position，因为它是物理 page ID 与 page 内偏移共同决定的 K/V cache 地址。
但同一个 request 本轮的 position 必须连续：它总是从该 request 的
`num_computed_tokens` 开始，取长度为 `num_scheduled_tokens` 的连续区间。当前的
纯 decoder MVP 不会在同一 request 内跳过位置 1、2、3 后直接计算位置 4、7。

## 当前 MVP 是上面转换的退化情况

当前 nano-vLLM 的端到端 KV cache 布局仍只接通：

~~~text
block_size == kernel_block_size
blocks_per_kv_block = 1
~~~

此时转换仍然存在，只是恒等：

~~~text
manager block ID 8 -> kernel page ID [8]
manager block ID 2 -> kernel page ID [2]
manager block IDs [8, 2, 11]
                    ->
kernel page row  [8, 2, 11]
~~~

因此，当前端到端 MVP：

- KVCacheManager 的 block_ids 可以原样镜像到 BlockTable。
- manager block 下标等于 kernel page 下标。
- 输入 max_num_blocks_per_req 等于内部 GPU table 列容量。
- slot 使用 block_size 和 kernel_block_size 得到相同结果。

这只是转换倍率为 1 的特例，不能反过来把 BlockTable 定义为“KVCacheManager 的普通镜像”。

不过，BlockTable 类本身已保留 vLLM 的 hybrid 转换逻辑。等 KV cache tensor 的真实页布局和
attention backend 都支持不同 page 大小时，ModelRunner 只需传入不同的两个 size，就能使用该
转换；在那之前，配置层仍应拒绝或避免这种组合。

## 实现不变量

- KVCacheManager 是 manager block 的唯一资源所有者；BlockTable 不拥有和释放 block。
- add_row 与 append_row 的输入是 manager block ID；GPU row 的内容是 kernel page ID。
- num_blocks_per_row 数的是有效 kernel page，不是 manager block。
- GPU page table 的列索引和 slot 的页内 offset 永远按 kernel_block_size 解释。
- add_row 接收完整 manager table；append_row 接收新增 manager block。
- CPU table 更新、InputBatch condense 完成后，才提交 GPU table。
- slot_mapping buffer 的元素数固定为 max_num_batched_tokens；只有前 num_tokens 个
  元素对应本轮 packed Q/K/V token，后缀必须是 -1。
- kernel_block_size 不能整除 block_size 时，必须拒绝配置。
