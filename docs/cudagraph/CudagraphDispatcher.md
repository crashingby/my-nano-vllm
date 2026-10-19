# CudagraphDispatcher：选择本轮执行模式与 batch 容量

[`CudagraphDispatcher`](../../nanovllm/cudagraph_dispatcher.py) 为一个 Runner 维护可用的 CUDA Graph 执行计划。启动时，它根据已经确定的策略和捕获尺寸登记 batch 布局；推理时，它把本轮真实 token 数和 batch 类型转换成一个可用计划，交给 Runner 准备输入。

例如，当前使用 FULL_DECODE_ONLY，候选尺寸是 `[1,2,4,8,16]`，请求上限是 8。每个 decode 请求本轮计算一个 Query，因此可以准备一槽、两槽、四槽和八槽的 FULL 图。Dispatcher 用 `self.cudagraph_keys` 保存这组允许使用的布局：

```python
cudagraph_keys = {
    CUDAGraphMode.FULL: {
        BatchDescriptor(1, 1, True),
        BatchDescriptor(2, 2, True),
        BatchDescriptor(4, 4, True),
        BatchDescriptor(8, 8, True),
    },
    CUDAGraphMode.PIECEWISE: set(),
}
```

这个字典按具体模式分组，每组是一个描述符集合。`BatchDescriptor(4,4,True)` 表示 FULL 分支允许使用四行、四槽位的专用 decode 布局。启动阶段，Runner 从这些集合取得按 8、4、2、1 排列的捕获计划，逐档准备 dummy，再由 Wrapper 保存实际捕获的图。

在线请求持续加入和结束，真实 batch 大小会逐轮变化。为了快速选择容量，Dispatcher 在初始化时还准备 `self._bs_to_padded_graph_size` 查表。例如真实 token 数 3 对应容量 4，5 对应 8，9 对应 16。查表给出候选容量，`cudagraph_keys` 再决定这个容量在当前模式下是否可用。前例中，16 属于候选尺寸，而专用 decode 计划取请求上限内的 1、2、4、8。

当 A/B/C 三个请求各执行一个 Query 时，`dispatch(3, True)` 先选中容量 4，再匹配四槽 FULL 描述符，返回 `(FULL, BatchDescriptor(4,4,True))`。Runner 据此填充四行输入和四槽 metadata，Wrapper 查找并回放四槽图。下一轮换成 B/C/D，只要仍满足这套布局，就可以继续选择同一个计划。

因此，这个对象在整个运行过程中持续保留的是**布局集合和容量选择规则**。`keys_initialized` 在初始化完成时设为 True，之后在线分派反复读取这些规则；请求数据保存在 Runner 的输入 buffer 中，实际图保存在 Wrapper 中。两者通过同一个描述符衔接。

下面逐个介绍接口的声明和调用约定，再用完整例子展示启动与在线调用，最后简述处理逻辑。模式含义见 [CUDAGraphMode](CUDAGraphMode.md)，具体布局见 [BatchDescriptor](BatchDescriptor.md)，GPU 捕获原理见 [总览](overview.md)。

## 对外接口

以下声明保留当前源码中的参数、默认值和返回类型。每个声明下方说明调用约定，并给出具体例子。

### __init__()：保存配置并建立初始状态

```python
def __init__(self, vllm_config: NanovllmConfig) -> None:
    ...
```

**输入：** `vllm_config` 为 Runner 使用的总配置。Dispatcher 读取其中的 `compilation_config` 和 `scheduler_config.max_num_seqs`。启用 Graph 时，捕获尺寸列表应由总配置完成排序与过滤，最大捕获尺寸等于列表末项。

**返回：** `CudagraphDispatcher(config)` 创建并返回 Dispatcher 对象；其中的 `__init__()` 方法返回 None。

**状态变化：** 保存总配置和编译配置的引用；将 `uniform_decode_query_len` 设为 1，FULL 与 PIECEWISE 的 key 集合设为空，`keys_initialized` 设为 False，当前策略设为 NONE。

**调用时机：** `GPUModelRunner` 构造时创建。之后由 Attention 能力解析流程确定策略并初始化 keys。

例如，创建对象后立即分派三个 token：

```python
dispatcher = CudagraphDispatcher(config)
assert dispatcher.dispatch(3, True) == (
    CUDAGraphMode.NONE,
    BatchDescriptor(3),
)
```

此时对象处于初始化前的状态，分派结果使用普通调用和真实 token 数 3。

### initialize_cudagraph_keys()：建立合法执行计划

```python
def initialize_cudagraph_keys(
    self, cudagraph_mode: CUDAGraphMode, uniform_decode_query_len: int = 1
) -> None:
    ...
```

**输入：** `cudagraph_mode` 为 Attention 能力解析后的最终策略。当前 FA2 通常传入 FULL_DECODE_ONLY。`uniform_decode_query_len` 表示每请求 decode Query 数，当前接受 1，其他值触发 ValueError。

**返回：** None。建立的计划保存在实例成员中，后续通过 `get_capture_descs()` 取得。

**状态变化：** 将最终策略写入 `self.cudagraph_mode`。启用 Graph 时，建立容量查表并向各模式集合登记 key；结束时设置 `keys_initialized=True`。NONE 策略直接设置初始化完成标记。

**调用时机：** Runner 的 `_check_and_update_cudagraph_mode()` 在后端能力解析后调用，正式预热与捕获随后使用这些计划。这个对象随 Runner 在启动阶段初始化一次。

假设捕获候选为 `[1,2,4,8,16]`、请求容量为 8：

```python
dispatcher.initialize_cudagraph_keys(
    CUDAGraphMode.FULL_DECODE_ONLY,
    uniform_decode_query_len=1,
)
assert dispatcher.keys_initialized
assert dispatcher.cudagraph_keys[CUDAGraphMode.FULL] == {
    BatchDescriptor(n, n, True) for n in (1, 2, 4, 8)
}
```

每个 decode 请求执行一个 Query，请求容量为 8，因此登记 1、2、4、8 四个档位。之后 Runner 逐档准备 dummy，并由 Wrapper 创建对应图。

### get_capture_descs()：返回启动捕获计划

```python
def get_capture_descs(self) -> list[tuple[CUDAGraphMode, list[BatchDescriptor]]]:
    ...
```

**输入：** 调用时读取实例中的初始化状态、最终策略和各模式 key 集合。

**返回：** `[(具体模式, 描述符列表), ...]`。分组顺序为 PIECEWISE、FULL，各组内部按 `num_tokens` 降序排列；初始化前、NONE 策略或空计划返回空列表。

**状态变化：** 将已有集合复制成排序后的列表，实例中的 key 集合保持原值。

**调用时机：** 启动捕获时，`GPUModelRunner.capture_model()` 取得计划，再逐项执行预热和正式捕获。

沿用前例，返回值为：

```python
assert dispatcher.get_capture_descs() == [
    (
        CUDAGraphMode.FULL,
        [BatchDescriptor(n, n, True) for n in (8, 4, 2, 1)],
    ),
]
```

Runner 先处理八槽位，再处理四槽、两槽和一槽。大尺寸先捕获，后续小图可以复用共享 Graph pool 的分配空间。

### dispatch()：选择本轮模式与执行容量

```python
def dispatch(
    self,
    num_tokens: int,
    uniform_decode: bool = False,
    valid_modes: AbstractSet[CUDAGraphMode] | None = None,
    invalid_modes: AbstractSet[CUDAGraphMode] | None = None,
) -> tuple[CUDAGraphMode, BatchDescriptor]:
    ...
```

**输入：** 调用方按本轮实际调度情况提供四个参数。

- `num_tokens`：非负整数，表示各请求本轮 Query 数的总和。例如 `[3,1]` 对应 4，`[1,1,1]` 对应 3。
- `uniform_decode`：Runner 已确认每个有效请求都按固定 decode Query 长度执行，当前每请求为 1。例如 `[1,1,1]` 传 True，`[3,1]` 传 False。历史 KV 长度 `[128,42,7]` 由后续 metadata 单独保存。
- `valid_modes`：本轮允许尝试的具体模式集合。None 或空集合使用 `{NONE,PIECEWISE,FULL}`，`{NONE}` 指定普通执行。
- `invalid_modes`：从本轮允许集合中排除的具体模式。例如 `{FULL}` 排除完整图，继续尝试已有的 PIECEWISE 计划或 NONE。

模式集合填写 NONE、PIECEWISE、FULL；整体组合策略在初始化阶段确定。

**返回：** 二元组 `(runtime_mode, batch_descriptor)`。

- FULL 的描述符保存模型行容量、Attention 请求槽位容量和归一化 uniform 标记。Runner 据此填充固定布局，再回放完整图。
- PIECEWISE 的描述符保存 token 档位，`num_reqs=None、uniform=False`。具备分段执行链时，各片段按这个档位执行。
- NONE 的描述符保存真实 token 数，`num_reqs=None、uniform=False`，Runner 按真实 batch 普通调用模型。

**状态变化：** 读取容量查表和合法 key 集合，创建本次返回的描述符。Dispatcher 成员及调用方传入的模式集合保持原值。

**调用时机：** 在线路径由 `_determine_batch_execution_and_padding()` 调用，随后 Runner 按结果准备 padding；启动阶段 `_dummy_run()` 也经过这条分派路径，以保持 dummy 与在线布局使用相同规则。

例如，初始化完成后分派三个单 token 请求：

```python
mode, desc = dispatcher.dispatch(num_tokens=3, uniform_decode=True)
assert mode is CUDAGraphMode.FULL
assert desc == BatchDescriptor(4, 4, True)
```

查表把真实数量 3 放入容量 4，单 token decode 对应四个请求槽位。返回后，Runner 保留真实数量 3，并将输入和 metadata 准备成四槽布局。图缓存的查找与回放由 Wrapper 完成。

当前项目已跑通 FULL_DECODE_ONLY。PIECEWISE 的分派规则保留在本类中，分段编译执行链待实现，配置解析器现阶段对分段策略报错。

### add_cudagraph_key()：登记一个布局

```python
def add_cudagraph_key(
    self, runtime_mode: CUDAGraphMode, batch_descriptor: BatchDescriptor
) -> None:
    ...
```

**输入：** `runtime_mode` 取 FULL 或 PIECEWISE；`batch_descriptor` 表示该模式下要登记的执行布局。传入其他模式触发断言。

**返回：** None。

**状态变化：** 向 `cudagraph_keys[runtime_mode]` 集合插入描述符。相同字段值的描述符按集合规则去重，`keys_initialized` 保持原值。

**调用时机：** `initialize_cudagraph_keys()` 逐项登记计划时使用。手工增加计划时，需要为该布局同时安排 Runner 输入准备与实际捕获。

前例已包含四槽位计划，再登记相同描述符后，FULL 集合仍有四项：

```python
dispatcher.add_cudagraph_key(
    CUDAGraphMode.FULL,
    BatchDescriptor(4, 4, True),
)
assert len(dispatcher.cudagraph_keys[CUDAGraphMode.FULL]) == 4
```

## 完整例子：启动取得计划，在线三请求选择四槽图

### 用 CPU 示例观察接口结果

下面使用真实 `CompilationConfig`，并用 `SimpleNamespace` 提供 Dispatcher 读取的配置字段。这个示例专门演示本类接口；实际 Runner 传入完整 `NanovllmConfig`，并通过后端能力解析确定策略。

```python
from types import SimpleNamespace
from typing import cast

from nanovllm.config import CompilationConfig, CUDAGraphMode, NanovllmConfig
from nanovllm.cudagraph_dispatcher import CudagraphDispatcher
from nanovllm.forward_context import BatchDescriptor

compilation_config = CompilationConfig(
    cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
    cudagraph_capture_sizes=[1, 2, 4, 8, 16],
    max_cudagraph_capture_size=16,
)
config = cast(
    NanovllmConfig,
    SimpleNamespace(
        compilation_config=compilation_config,
        scheduler_config=SimpleNamespace(max_num_seqs=8),
    ),
)

dispatcher = CudagraphDispatcher(config)
dispatcher.initialize_cudagraph_keys(CUDAGraphMode.FULL_DECODE_ONLY)

# 启动阶段：取得 Runner 将逐项预热和捕获的计划。
plans = dispatcher.get_capture_descs()
assert plans == [
    (
        CUDAGraphMode.FULL,
        [BatchDescriptor(n, n, True) for n in (8, 4, 2, 1)],
    ),
]

# 在线阶段：A/B/C 各调度一个 token，选择四槽位布局。
mode, desc = dispatcher.dispatch(num_tokens=3, uniform_decode=True)
assert mode is CUDAGraphMode.FULL
assert desc == BatchDescriptor(4, 4, True)
```

示例只运行 Dispatcher 的 CPU 逻辑。真实启动时，Runner 取得 `plans` 后逐档准备 dummy，Wrapper 创建图并缓存；在线请求随后使用这些已准备的执行计划。

### 启动接口的输入怎样变成捕获计划

捕获候选尺寸是 `[1,2,4,8,16]`，请求容量是 8。当前每请求 decode 一个 Query，因此专用 decode 图最多需要八个 token 行和八个请求槽位。

据此，本类返回四个 FULL 档位，顺序为 8、4、2、1。候选尺寸 16 留在配置范围内，decode 专用的捕获计划取请求容量内的档位。

Runner 按返回顺序预热和捕获，大尺寸先执行，后续小图可以复用共享 Graph pool 中的分配空间。关于实际捕获和图缓存，见 [Graph Wrapper](CUDAGraphWrapper.md)。

### 在线接口的输入怎样变成返回值

假设 A/B/C 本轮的输入如下：

| 请求 | 本轮 Query 数 | token ID | position | 写入后总 KV 长度 |
| --- | --- | --- | --- | --- |
| A | 1。 | 15。 | 127。 | 128。 |
| B | 1。 | 25。 | 41。 | 42。 |
| C | 1。 | 32。 | 6。 | 7。 |

Runner 得到总 token 数 3，并确认每个请求本轮都只有一个 Query，于是调用 `dispatch(3, True)`。

用通俗语言描述，中间转换就是：

1. **找能容纳这三个 token 的档位。** 1 和 2 都偏小，4 是最小兼容容量。
2. **确定 Attention 要准备几个槽位。** 当前每请求一个 Query，四行 decode 布局对应四个请求槽位。
3. **选择这类 batch 对应的执行路线。** FULL_DECODE_ONLY 的单 token 分支使用 FULL 和专用 uniform 标记。
4. **确认这套布局已在启动计划中登记。** 找到四槽位计划后，返回 `(FULL, BatchDescriptor(4,4,True))`。

此时调用结果表达了两个数量：真实请求与 token 各为 3，选中的执行容量各为 4。

### Runner 怎样使用这个返回值

Runner 按四槽图更新固定地址中的内容：

| 数据 | 三个请求的有效内容 | 交给四槽执行方案的视图 |
| --- | --- | --- |
| `input_ids` | `[15,25,32]`。 | `[15,25,32,0]`。 |
| `positions` | `[127,41,6]`。 | `[127,41,6,0]`。 |
| `query_start_loc` | `[0,1,2,3]`。 | `[0,1,2,3,3]`，末槽位 Query 长度为 0。 |
| `seq_lens` | `[128,42,7]`。 | `[128,42,7,0]`。 |
| `slot_mapping` | 三行真实 KV 写入 slot。 | 三行真实 slot，加一个 `-1`。 |
| `block_table` | 三行真实请求的页映射。 | 四行，末行填保留空页 0。 |

随后 Runner 在 `ForwardContext` 中传入 FULL 和四槽描述符，Wrapper 回放对应的图。模型执行四行，Runner 最后选择 A/B/C 的三行 hidden states 计算 logits 并采样。

这些具体填充值由 Runner 与 BlockTable 准备。详细的 slot 数值、固定地址和 kernel 参数对应关系见 [BatchDescriptor 的完整例子](BatchDescriptor.md#完整例子三个请求怎样补到四槽图)。

### 下一轮和混合 batch 会得到什么

A 结束、D 进入 decode 后，B/C/D 仍各执行一个 Query，调用仍是 `dispatch(3, True)`，返回同一个四槽位描述符。Runner 把新请求的数据写入固定 buffer 后，可以继续使用对应图。

同一配置下的其他接口结果如下：

| 本轮 Query 长度 / 调用限制 | 输入 | 返回 | 通俗解释 |
| --- | --- | --- | --- |
| `[1,1]` | `dispatch(2, True)`。 | FULL，`BatchDescriptor(2,2,True)`。 | 使用两槽图。 |
| `[1,1,1,1,1]` | `dispatch(5, True)`。 | FULL，`BatchDescriptor(8,8,True)`。 | 五个请求放入八槽布局。 |
| `[3,1]` | `dispatch(4, False)`。 | NONE，`BatchDescriptor(4)`。 | 当前混合分支普通执行四个 token。 |
| `[17]` | `dispatch(17, False)`。 | NONE，`BatchDescriptor(17)`。 | 超过捕获上限 16，普通执行。 |
| `[1,1,1]`，本轮指定 eager | `dispatch(3, True, valid_modes={CUDAGraphMode.NONE})`。 | NONE，`BatchDescriptor(3)`。 | 按三个真实 token 普通执行。 |

## 处理逻辑概述

### 初始化：把捕获尺寸整理成可用布局

初始化先建立“真实 token 数 → 执行容量”的查表。例如尺寸列表 `[1,2,4,8,16]` 会把 3 映射到 4、5 映射到 8、9 映射到 16。

接着按最终策略登记布局。当前 FULL_DECODE_ONLY 每请求一个 Query，请求容量为 8，因此 FULL 集合包含 `(1,1,True)`、`(2,2,True)`、`(4,4,True)`、`(8,8,True)`。每个三元组依次表示 token 容量、请求槽位容量和 uniform 标记。登记结束后，设置 `keys_initialized=True`。

### 在线分派：选择容量，匹配模式，再返回计划

```mermaid
flowchart LR
    A["真实 token 数与 batch 类型"] --> B["确定本轮允许模式"]
    B --> C["查找最小兼容容量"]
    C --> D["按 FULL、PIECEWISE 顺序匹配布局"]
    D --> E["返回模式与描述符<br>需要时回退 NONE"]
```

例如三个单 token 请求先查到容量 4，再按 FULL_DECODE_ONLY 构造 `(4,4,True)`，命中后返回 FULL。混合 batch 则使用 mixed 分支，当前策略返回 NONE 和真实 token 数。

FULL 的匹配使用描述符的三个字段；PIECEWISE 将请求维度统一填写为 None、uniform 统一填写为 False，按 token 容量选择片段布局。组合策略保留专用 decode 标记，独立 FULL 使用一般布局标记 False。

初始化前、NONE 策略或超出捕获范围时，接口提前返回 NONE。模式过滤后为空，或进入布局匹配后全部落空且调用方排除了 NONE 时，会触发断言。早返回分支沿用上游规则，调用方若严格要求特定运行模式，应检查返回结果。

### 捕获计划：按模式分组，大尺寸优先

`get_capture_descs()` 将合法布局整理为 PIECEWISE、FULL 两个分组，各组内部按 token 容量降序排列。当前策略只返回 FULL 的 8、4、2、1 四个档位，Runner 按这个顺序预热并捕获。

Dispatcher 保存布局规则，Runner 准备固定输入，Wrapper 保存实际图和捕获完成状态。

## 与 Runner 和 vLLM 的对应关系

| 场景 | 当前项目中的调用位置 | 分工 |
| --- | --- | --- |
| 最终策略初始化 | `GPUModelRunner._check_and_update_cudagraph_mode()`。 | 查询 Attention 能力，解析模式，然后调用 `initialize_cudagraph_keys()`。 |
| 启动捕获 | `capture_model()` 与 `_capture_cudagraphs()`。 | 取得 `get_capture_descs()`，逐项预热和捕获。 |
| 本轮执行选择 | `_determine_batch_execution_and_padding()`。 | 判断 uniform、设置模式限制并调用 `dispatch()`。 |
| 固定布局准备 | `execute_model()`、`_get_slot_mappings()`、`_build_attention_metadata()`。 | 按返回容量更新输入、padding 与 metadata。 |
| 图缓存与执行 | `CUDAGraphWrapper.__call__()`。 | 根据 ForwardContext 的具体模式和 key 捕获或回放图。 |

本类对应上游 [`vllm/v1/cudagraph_dispatcher.py`](https://github.com/vllm-project/vllm/blob/b5adb027ad03c29b46181752ba3b1cb84eff1dd4/vllm/v1/cudagraph_dispatcher.py)，保留尺寸映射、分支 key、FULL 优先匹配和捕获顺序。当前范围为普通单 token decode，LoRA、spec decode 和专用 `compile_sizes` 校验留待相应功能接入。
