# InputBatch：Worker 的持久请求槽位工作区

源码：[gpu_input_batch.py](../../nanovllm/worker/gpu_input_batch.py)。

## 它解决什么问题

连续批处理不会每一轮都使用同一组请求：有的请求完成，有的新请求加入，有的长 prompt
只计算一部分，有的 request 暂时没有被调度。若每轮都根据活跃 request 重新创建 token
矩阵、block table 和采样状态，会产生大量 Python 对象、CPU tensor 分配和 H2D 拷贝。

`InputBatch` 是 GPU Worker 长期持有的一块**固定容量工作区**。它将每个当前参与执行的
request 放进一个物理槽位 `req_index`，并让同一槽位的所有行式数据保持同步。它不是
Scheduler 的请求真相来源，也不执行模型 forward；它是 `GPUModelRunner` 在两次 forward
之间复用状态、整理输入和维护紧凑 batch 的工具类。

可以把它理解为 Worker 内部的“请求行表”：

~~~text
物理槽位 req_index:       0             1             2
_req_ids:                 A             B             C
token_ids_cpu:       A 的完整 token  B 的完整 token  C 的完整 token
num_computed_tokens:      4             9             2
temperature:             0.7           1.0           0.9
block_table 行:        A 的 page 表   B 的 page 表   C 的 page 表
processor 状态行:     A 的状态        B 的状态        C 的状态
~~~

这里最重要的不是某一个 tensor，而是**行对齐不变量**：任意时刻，同一个 `req_index`
上的 request ID、token、KV 进度、block table、温度以及未来 logits processor 状态，
必须全部属于同一个 request。

## 与其他对象的边界

| 对象 | 它保存什么 | 是否是权威状态 | 与 InputBatch 的关系 |
| --- | --- | --- | --- |
| `Scheduler` / `Request` | 排队、运行/抢占/完成状态、逻辑 token 序列、物理 KV block 所有权。 | 是。 | 通过 `SchedulerOutput` 告诉 Worker 本轮要运行谁。 |
| `CachedRequestState` | Worker 对单个 `Request` 的长期镜像：prompt、输出 token、block IDs、KV 进度。 | 对 Worker 来说是镜像真相。 | 可以存在于 `GPUModelRunner.requests` 中，但暂时不位于 InputBatch。 |
| `InputBatch` | **本轮实际执行** request 的紧凑物理槽位布局和固定容量 buffer。 | 否。 | 从 `CachedRequestState` 拷贝或引用数据；本轮未调度的 request 可以离开它。 |
| `BlockTable` | 每个槽位的 manager block ID 到 kernel page ID 转换结果，以及 GPU page table。 | 否。 | `InputBatch` 持有一个 `BlockTable`，并在槽位变化时同步其行。 |
| `BatchUpdateBuilder` | 本 engine step 槽位发生的删除、新增、移动事件。 | 否。 | 由 `InputBatch` 写入，在 `refresh_metadata()` 时交付给 logits processor。 |
| `LogitsProcessor` | repetition penalty、grammar 等未来采样 feature 的逐 request 状态。 | 各 processor 自己拥有。 | 必须通过 `BatchUpdate` 跟随 InputBatch 的槽位变化。 |

`CachedRequestState` 与 `InputBatch` 的区别尤其重要：一个 request 被抢占或本轮未调度时，
它可以从 `InputBatch` 移除，但 Worker 仍保留其 `CachedRequestState`，以便以后恢复；
因此“离开 InputBatch”不等于“请求结束”。

## 固定容量布局

构造 `InputBatch` 时会一次性确定以下上限：

| 参数或派生值 | 约束的对象 | 典型 buffer |
| --- | --- | --- |
| `max_num_reqs` | 持久 batch 最多可同时驻留多少 request。 | 所有按 request 行组织的 buffer 的第一维。 |
| `max_model_len` | 单个 request 在 Worker 可保存的最长完整 token 序列。 | `token_ids_cpu_tensor` 的第二维。 |
| `max_num_batched_tokens` | 单次 forward 最多 packed 多少 query token。 | `BlockTable.slot_mapping` 的容量。 |
| `max_num_blocks_per_req` | 一个 request 最多需要多少 manager block。 | `BlockTable` 输入行宽；hybrid 时内部会换算为 kernel page 行宽。 |
| `block_size` | KVCacheManager 分配 manager block 的 token 粒度。 | 影响最大 block 数和逻辑 block table。 |
| `kernel_block_size` | Attention kernel 使用的 page token 粒度。 | 影响 Worker 的 kernel page table 与 slot mapping。 |

当前 MVP 只有一个 KV cache group、文本 token 输入和 temperature sampling；因此每个槽位
只保存一条 token 行、一个 `BlockTable` 行和一个 temperature。多模态、多个 cache group、
PP 输入传输等 feature 加入后，仍应保持“同一 `req_index` 的所有行一起移动”的原则。

## 每个槽位维护哪些成员

### 身份与 Python 容器

| 成员 | 类型 | 含义 | 何时变化 |
| --- | --- | --- | --- |
| `_req_ids` | `list[str \| None]` | 每个物理槽位当前是谁。更新中允许 `None` 空洞。 | 新增、删除、交换、压缩。 |
| `req_id_to_index` | `dict[str, int]` | request ID 到物理槽位的反向映射；`num_reqs` 以它的长度为准。 | 新增、删除、交换、压缩。 |
| `req_output_token_ids` | `list[list[int] \| None]` | 指向 `CachedRequestState.output_token_ids` 的**引用**。 | 新增、删除、交换、压缩。 |

`_req_ids` 的列表长度在删除之后可能大于 `num_reqs`。例如 `A, None, C, D` 有四个列表
元素，却只有三个活跃 request；此时不能调用 `req_ids` 属性，也不能 forward，必须先
`condense()`。

保存 `output_token_ids` 引用而非副本，是为了让采样器将新 token 追加到
`CachedRequestState.output_token_ids` 后，`InputBatch` 立刻能看到新的长度；不需要每次
decode 都重新复制整个 token 序列。

### CPU token 与进度行

| 成员 | shape / 类型 | 第 `i` 行或元素的含义 | 写入者 |
| --- | --- | --- | --- |
| `token_ids_cpu_tensor` | CPU `int32[max_num_reqs, max_model_len]` | request `i` 的 `prompt + 已确认输出` token 前缀。 | `add_request()`；ModelRunner 采样后补写新 token。 |
| `token_ids_cpu` | 上述 tensor 的 NumPy view。 | 供 Python 侧行拷贝和切片读取。 | 与 tensor 共享内存。 |
| `num_prompt_tokens_cpu_tensor` | CPU `int32[max_num_reqs]` | request `i` 的 prompt 长度。 | `add_request()`、交换、压缩。 |
| `num_computed_tokens_cpu_tensor` | CPU `int32[max_num_reqs]` | request `i` 已写入 KV cache 的 token 数。 | `add_request()`、SchedulerOutput 增量更新、交换、压缩。 |

`token_ids_cpu[i]` 保存的是完整逻辑序列；`num_computed_tokens_cpu[i]` 则划分其中哪部分
已经有 K/V。若该值为 `4`、本轮调度该 request 的 token 数为 `3`，ModelRunner 应读取：

~~~python
start = input_batch.num_computed_tokens_cpu[i]  # 4
end = start + num_scheduled_tokens[i]           # 7

input_ids = input_batch.token_ids_cpu[i, start:end]
positions = range(start, end)                    # [4, 5, 6]
~~~

因此 `InputBatch` 保存生成位置所需的历史进度，但不自己决定 `num_scheduled_tokens`；后者
只能来自 SchedulerOutput。

### KV page table 与采样元数据

| 成员 | 含义 | 关键更新点 |
| --- | --- | --- |
| `block_table` | 单 group 的 `BlockTable`。其第 `i` 行是 request `i` 的 GPU kernel page table。 | 新增 `add_row()`、删除 `clear_row()`、交换 `swap_row()`、压缩 `move_row()`。 |
| `temperature_cpu_tensor` / `temperature_cpu` | 每个 request 的 CPU temperature staging buffer 与 NumPy view。 | 新增、交换、压缩。 |
| `temperature` | GPU `float32[max_num_reqs]`。 | `refresh_metadata()` 从 CPU 前缀复制。 |
| `batch_update_builder` | 本轮槽位变化的事件收集器。 | 新增、删除、交换、压缩。 |
| `logitsprocs` | 已创建的 logits processor 分组。当前默认为空。 | `refresh_metadata()` 调用每个 processor 的 `update_state()`。 |

`InputBatch` 负责令 `BlockTable` 的行和 token 行使用同一个 `req_index`，但**不负责**在
每轮 forward 前提交 GPU page table 或生成 slot mapping。Runner 的 `_prepare_inputs()` 在
所有行整理完成后调用：

~~~python
input_batch.block_table.commit_block_table(input_batch.num_reqs)
input_batch.block_table.compute_slot_mapping(
    num_reqs=input_batch.num_reqs,
    query_start_loc=query_start_loc,
    positions=positions,
)
~~~

其中 `query_start_loc` 与 `positions` 由 ModelRunner 根据每个槽位的
`num_computed_tokens_cpu` 和 SchedulerOutput 生成。完整 page 转换与 Triton slot kernel
说明见 [block_table.md](block_table.md)。

## 稳定状态与更新状态

`InputBatch` 有两种合法形态，不能混淆：

| 形态 | `_req_ids` 示例 | 可以做什么 | 不可以做什么 |
| --- | --- | --- | --- |
| 稳定、紧凑 | `[A, B, C]` | 读取 `req_ids`、构造 packed inputs、提交 metadata、forward。 | 在没有记录 Builder 事件时直接改槽位。 |
| 更新中、有空洞 | `[A, None, C, D]` | 连续执行删除、新增和压缩。 | 调用 `req_ids`、直接进入 forward。 |

完成一个 engine step 内的槽位更新后，必须恢复前一种形态：

~~~text
remove_request / add_request / swap_states
                ↓
            condense
                ↓
        refresh_metadata
                ↓
       _prepare_inputs + forward
~~~

`refresh_metadata()` 必须位于 `condense()` 后：它需要稳定的逻辑 batch 大小作为
`BatchUpdate.batch_size`，并且 processor 只能看到最终紧凑布局。

## 接口与状态变化

### `num_reqs` 与 `req_ids`

~~~python
@property
def num_reqs(self) -> int: ...

@property
def req_ids(self) -> list[str]: ...
~~~

- `num_reqs` 返回 `len(req_id_to_index)`，即真实存活 request 数；即便 `_req_ids` 中有
  空洞也正确。
- `req_ids` 返回稳定前缀的 request ID 列表。它会断言 `_req_ids` 没有空洞，因此是
  forward 前的检查点，不是更新中的随意查询接口。

### `_register_add_request(request)`

这是 `add_request()` 的内部第一步，职责仅是选择物理槽位并记录 `added` 事件：

1. 调用 `batch_update_builder.pop_removed()`；若本轮已有删除，优先取数值最小的空槽位。
2. 若没有可复用空洞，则将新 request 放到当前 `num_reqs` 的位置。
3. 记录 `(req_index, sampling_params, prompt_token_ids, output_token_ids)` 到
   `builder.added`。

被 `pop_removed()` 消费的删除事件不会出现在最终 `BatchUpdate.removed` 中，因为该槽位
已经被新 request 直接覆盖；processor 只需根据 `added` 在同一行初始化新状态。

### `add_request(request)`

`add_request()` 将一个 `CachedRequestState` 写入完整的一行。设新 request `E` 的：

~~~text
prompt_token_ids    = [101, 102, 103]
output_token_ids    = [201]
num_computed_tokens = 3
block_ids           = [8]
temperature         = 0.7
~~~

若它被放进 `req_index=1`，调用后至少发生这些变化：

| 成员 | 调用后的第 1 行 / 条目 |
| --- | --- |
| `_req_ids` | `"E"` |
| `req_id_to_index` | `{"E": 1}`，并保留其他 request 映射 |
| `token_ids_cpu` | `[101, 102, 103, 201, ...]` |
| `num_prompt_tokens` | `3` |
| `num_computed_tokens_cpu` | `3` |
| `req_output_token_ids` | 指向 `E.output_token_ids` 的同一列表对象 |
| `block_table.cpu` | 由 `BlockTable.add_row([8], 1)` 写入 page ID 行 |
| `temperature_cpu` | `0.7` |
| Builder | 新增一条 `added` 事件 |

它不会直接拷贝整行 metadata 到 GPU；temperature 在 `refresh_metadata()` 复制，page table
在 `_prepare_inputs()` 阶段提交。

### `remove_request(req_id)`

`remove_request()` 只将 request 从**当前执行 batch** 拿走，不决定其 Scheduler 生命周期。
调用 `remove_request("B")` 后：

~~~text
调用前: _req_ids = [A, B, C, D]
调用后: _req_ids = [A, None, C, D]

req_id_to_index 中删除 B
req_output_token_ids[1] = None
BlockTable 第 1 行清零，num_blocks_per_row[1] = 0
Builder.removed 记录 [1]
~~~

可能出现的原因包括：`B` 已完成、被 Scheduler 抢占，或只是这一轮没有被调度。后两种情况
不能删除 `GPUModelRunner.requests["B"]` 中的 `CachedRequestState`。

### `swap_states(index_1, index_2)`

交换两个仍活跃槽位。这是对齐数据的完整交换，而不是只交换 `_req_ids`：

~~~text
_req_ids
req_id_to_index
req_output_token_ids
token_ids_cpu 的有效 token 前缀
num_prompt_tokens
num_computed_tokens_cpu
temperature_cpu
BlockTable 行
~~~

最后追加：

~~~python
(index_1, index_2, MoveDirectionality.SWAP)
~~~

到 `builder.moved`，使 processor 同样交换自己的逐 request 状态。当前 MVP 的常规
`condense()` 不需要它，但未来按请求类型重排 batch 时可以安全复用。

### `condense()`

`condense()` 消除删除留下的空洞。它不会依次左移每一个后续行，而是将**最末尾的活跃
request**搬进**最靠前的空槽位**，从而减少需要移动的行数。

它同步搬动的内容与 `swap_states()` 基本相同，但方向是单向的：

~~~text
source 行: token 有效前缀、prompt 长度、computed 长度、temperature、BlockTable 行、输出列表引用
                    ↓
target 空洞行: 完整覆盖
~~~

每一次单向搬动都记录：

~~~python
(source_index, target_index, MoveDirectionality.UNIDIRECTIONAL)
~~~

被搬入的目标空洞通过 `pop_removed()` 消费，因此不会作为独立 `removed` 事件交付。源行在
最终逻辑 `batch_size` 之外，processor 只需按 `UNIDIRECTIONAL` 将源状态移到目标状态。

`condense()` 结束后会截断 `_req_ids` 与 `req_output_token_ids` 的尾部，使：

~~~python
len(self._req_ids) == self.num_reqs
~~~

重新成立。

### `refresh_metadata()`

这是 InputBatch 的**提交边界**。必须在所有增删和压缩后、构造模型输入前调用：

1. 调用 `builder.get_and_reset(num_reqs)`，取得 `BatchUpdate` 或 `None`。
2. 对每个 logits processor 调用 `update_state(batch_update)`。
3. 若 batch 非空，将 `temperature_cpu_tensor[:num_reqs]` 异步复制到 GPU 的
   `temperature[:num_reqs]`。

即使 `num_reqs == 0`，第 1、2 步也不能跳过：最后一个 request 完成时，processor 仍需
收到删除事件并清理自己的状态。只有 temperature 的 GPU 拷贝可以跳过。

`BatchUpdateBuilder` 的排序、复用和交付细节见
[BatchUpdate.md](../sample/BatchUpdate.md)。

## 完整时间线：删除、立即复用与 BlockTable 联动

设固定容量为 `max_num_reqs=4`、`block_size=4`。本轮开始前：

~~~text
槽位:                   0       1       2       3
_req_ids:                A       B       C       D
num_computed_tokens:     4       6       3       7
block_table CPU 有效行: [8]   [2, 5]   [4]   [6, 1]
temperature CPU:        0.7     1.0     0.9     0.8
Builder: removed=[], added=[], moved=[]
~~~

现在 `B` 完成，而 SchedulerOutput 又让新 request `E` 本轮加入。`E` 的 block IDs 为
`[11]`，temperature 为 `0.6`。

### 1. `remove_request("B")`

~~~text
槽位:                   0       1       2       3
_req_ids:                A      None     C       D
num_computed_tokens:     4       6       3       7
block_table CPU 有效行: [8]     []      [4]   [6, 1]
Builder: removed=[1], added=[], moved=[]
~~~

第 1 行的 token 和标量 buffer 中旧值可以暂时保留，因为 `_req_ids[1]` 已经是 `None`，
后续逻辑不会读取它；但 `BlockTable.clear_row(1)` 会清掉有效 page 前缀，防止错误的
attention metadata 继续把槽位 1 解释成 `B`。

### 2. `add_request(E)`：立即复用槽位 1

`_register_add_request()` 调用 `pop_removed()` 得到 `1`，所以删除事件被消费。随后
`add_request()` 覆盖所有按行对齐的数据：

~~~text
槽位:                   0       1       2       3
_req_ids:                A       E       C       D
num_computed_tokens:     4       0       3       7
block_table CPU 有效行: [8]    [11]     [4]   [6, 1]
temperature CPU:        0.7     0.6     0.9     0.8
Builder: removed=[], added=[(1, E.params, E.prompt, E.outputs)], moved=[]
~~~

此时没有空洞，`condense()` 不会移动任何 request。对 processor 而言，正确语义不是“清空
B 后保留空行”，而是“第 1 行直接初始化为 E”；因此最终 `BatchUpdate.removed=[]` 而
`BatchUpdate.added` 包含 E。

### 3. `refresh_metadata()`：向 processor 提交最终布局

~~~python
batch_update = BatchUpdate(
    batch_size=4,
    removed=[],
    added=[(1, E.sampling_params, E.prompt_token_ids, E.output_token_ids)],
    moved=[],
)
~~~

之后 builder 切换到新的空事件列表，temperature 的前四项被复制至 GPU。后续
`_prepare_inputs()` 将提交更新后的 BlockTable CPU 行，并针对本轮实际调度的 token 计算
新的 `slot_mapping`。

## 第二个时间线：只有删除时如何压缩

仍从 `A, B, C, D` 开始，但本轮没有新 request 进入：

1. `remove_request("B")` 后，槽位布局是 `[A, None, C, D]`，Builder 有
   `removed=[1]`。
2. `condense()` 取最小空槽位 `1`，并将最末尾活跃 request `D` 从 `3` 单向搬到 `1`。
3. `BlockTable.move_row(3, 1)` 同时移动 D 的 kernel page 行，`req_id_to_index[D]`
   更新为 `1`。
4. `_req_ids` 尾部截断，最终稳定布局为 `[A, D, C]`。

最终事件是：

~~~python
BatchUpdate(
    batch_size=3,
    removed=[],
    added=[],
    moved=[(3, 1, MoveDirectionality.UNIDIRECTIONAL)],
)
~~~

`removed=[]` 仍然正确。B 原来的行已经被 D 覆盖；processor 只要将 D 的状态从第 3 行搬到
第 1 行即可。第 3 行不再属于新的逻辑 batch 前缀，之后若被新 request 使用，会以 `added`
事件重新初始化。

## 与连续批处理输入的衔接

`InputBatch` 不决定调度，但它保存 ModelRunner 将调度翻译为 packed tensor 的全部原料。
假设稳定 batch 为 `A, D, C`，并且本轮 SchedulerOutput 是：

~~~text
num_scheduled_tokens = {
    "A": 2,
    "D": 1,
    "C": 3,
}
~~~

ModelRunner 应按 `InputBatch.req_ids` 顺序读取每行的已计算长度：

~~~text
A: num_computed_tokens = 4 -> positions [4, 5]
D: num_computed_tokens = 7 -> positions [7]
C: num_computed_tokens = 3 -> positions [3, 4, 5]

packed positions       = [4, 5, 7, 3, 4, 5]
query_start_loc        = [0, 2, 3, 6]
~~~

随后它应：

1. 用每段连续位置从对应 `token_ids_cpu` 行取出 `input_ids`。
2. 调用 `block_table.commit_block_table(num_reqs=3)`。
3. 调用 `block_table.compute_slot_mapping(3, query_start_loc, positions)`。
4. 将 packed input、position、page table、slot mapping 安装到 Attention context。
5. 执行 forward；每个 request 本轮最后一个 token 对应的 logits 用于采样。

这就是为什么 `InputBatch` 要持久维护 `num_computed_tokens`、token 行和 BlockTable 行；三者
缺任意一个都无法从混合的 SchedulerOutput 恢复本轮的物理执行输入。

## 与 Runner 和采样器的分工

`InputBatch` 自身已经具备固定槽位、行对齐、增删复用、压缩、BlockTable 联动和
`BatchUpdateBuilder` 交付边界。以下执行路径由 `GPUModelRunner` 编排，已与本类接通：

- 在 `_update_states()` 中消费 `SchedulerOutput`，维护 `CachedRequestState`，并按正确
  顺序调用 `remove_request()`、`add_request()`、`condense()`、`refresh_metadata()`。
- 在 `_prepare_inputs()` 中从 `token_ids_cpu` 和 `num_computed_tokens_cpu` 构造 packed
  `input_ids`、`positions`、`query_start_loc` 与 logits 采样索引。
- 在 forward 后将采样 token 追加到 `CachedRequestState.output_token_ids`，并把新 token
  写进对应 `token_ids_cpu` 行，使下一轮 decode 能读取它。

`Sampler` 已调用 processor 容器，默认容器为空。引入 repetition penalty、top-k/top-p、
grammar 等规则时，还需要各自的实现与配置；现有 `BatchUpdate` 负责保持逐请求状态的槽位一致。

## 必须保持的不变量

- `req_id_to_index` 中每个 request ID 恰好对应 `_req_ids` 的一个非 `None` 槽位。
- 稳定状态下，活跃 request 严格占据 `[0, num_reqs)`，此时才允许 forward。
- 对同一个 `req_index`，token 行、prompt 长度、computed 长度、temperature、BlockTable
  行、输出 token 引用和 processor 状态必须属于同一 request。
- `num_computed_tokens_cpu[i] <=` 该 request 当前完整 token 数；本轮 positions 从这个值
  开始连续递增。
- 每次调用 `remove_request()` 后，在 forward 前必须先完成所有必要的 `add_request()` 和
  `condense()`。
- 每个 engine step 只能在所有删除登记完成后首次读取 Builder 的 `removed`、
  `peek_removed()` 或 `pop_removed()`；之后不可再登记新删除。
- `refresh_metadata()` 必须在布局稳定后调用，且即使 batch 为空也要交付 Builder 事件。
- `BlockTable` 的 CPU 行变化必须在生成 slot mapping 前提交到 GPU；slot mapping 中的
  position 必须对应 token 行里本轮实际送入模型的连续 token 区间。
