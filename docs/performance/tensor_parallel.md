# 单机张量并行（TP）性能对比

本次在两张 NVIDIA A30 上，使用 Qwen3-0.6B 对比 TP=1 与 TP=2 的吞吐和延迟。
测试日期为 **2026-10-30**，两组均开启 CUDA Graph、异步调度和前缀缓存，
并统一使用 `mp` 执行器与 `spawn` 启动方式。

在 1024 条 Alpaca 请求、最多 16 条活跃请求、每条固定生成 128 token 的负载下，
TP=2 的平均输出吞吐比 TP=1 **下降 20.92%**，平均 TTFT **增加 25.96%**。
TP=2 实际使用 NCCL SHM 通信；独立 Nsight 采集中，NCCL 占 GPU kernel
累计时间的 **43.95%**。这组结果表明，本轮配置下增加 TP 没有取得性能收益。

## 环境与代码版本

| 项目 | 本次环境 |
| --- | --- |
| GPU | 2 × NVIDIA A30，每卡 24576 MiB 显存，无 NVLink。 |
| PCIe 与拓扑 | 两卡均为 Gen4 ×16；GPU 间为 `SYS`，跨 CPU / NUMA 节点。 |
| CPU | 2 × Intel Xeon Gold 6330，112 个逻辑 CPU，2 个 NUMA 节点。 |
| 驱动 | 550.163.01。 |
| Python / PyTorch | 3.10.20 / 2.5.1+cu124。 |
| Triton / FlashAttention | 3.1.0 / 2.8.3.post2，使用 FA2 后端。 |
| Transformers / NCCL | 4.57.6 / 2.21.5。 |
| 模型 | 本地 `~/Models/Qwen3-0.6B`，BF16。 |
| 执行器与线程 | TP=1/2 均为 `mp` + `spawn`，`OMP_NUM_THREADS=1`，未额外绑核。 |



本机的硬件和软件环境与 [CUDA Graph](cuda_graph.md)、[异步调度](async_schedule.md)
等文档中的 RTX 4060 Laptop 不同，不能将这些文档之间的绝对 TPS 差解释为代码优化收益。

## 固定负载与配置

| 项目 | 配置 |
| --- | --- |
| 数据集 | 本地 Alpaca Parquet，按源文件顺序取前 1024 条，不重排、不重复、不跳过。 |
| 输入构造 | 使用 `instruction`，有 `input` 时以两个换行拼接；不使用参考 `output`。 |
| 对话模板 | Qwen3 user 模板，`add_generation_prompt=True`、`enable_thinking=False`。 |
| 输入长度 | 共 30123 token；最小 17、平均 29.417、中位数 26、最大 135。 |
| 输出与采样 | 每条固定 128 token，忽略 EOS，`temperature=0.6`、`seed=0`；每轮共 131072 个输出 token。 |
| 最大活跃请求数 | `max_active=16`，完成后持续补充后续请求。 |
| 单轮 token 预算 | `max_num_batched_tokens=1024`。 |
| 上下文与输入上限 | `max_model_len=1024`、`max_prompt_tokens=512`；策略为 `truncate`，本批没有实际截断。 |
| CUDA Graph | `FULL_DECODE_ONLY`；uniform decode 使用完整 Graph，prefill 和混合批次走 eager。 |
| 异步调度 | 开启，通过 `step_fn()` 提交与回收在途批次。 |
| 前缀缓存 | 开启，`block_size=256`。 |
| 显存预算 | 每张使用中的 GPU 按总显存的 0.2 规划。 |
| 额外请求预热 | 每次 Engine 执行 16 条合成请求，输入为 128 个 token ID 100，输出 16 token。 |
| 随机数重置 | 预热后通过 Worker RPC 恢复 seed 0，不保证跨 TP 配置逐 token 一致。 |
| NCCL 路径 | TP=2 channel 均使用 SHM。 |

TP=1 也显式使用多进程执行器，避免把 `uni` 与 `mp` 的进程模型差异混入 TP 对比。
两组具有相同的请求并发与输出总量；TP=2 是两张卡协同运行**一个模型副本**。

## 测试方法与计时边界

使用原有 [吞吐入口](../../benchmarks/benchmark_engine_throughput.py) 和
[延迟入口](../../benchmarks/benchmark_engine_latency.py)，每种 TP 配置、每个入口
各重复 3 次，共 **12 次独立 Engine 测量**。每轮分别完成吞吐与延迟测试，
两个入口内的 TP 顺序一致：

| 轮次 | TP 顺序 |
| --- | --- |
| 第 1、3 轮 | TP=1 → TP=2。 |
| 第 2 轮 | TP=2 → TP=1。 |

数据读取、模板构造与编码、模型加载、显存 profiling、Graph 捕获、合成请求预热、
JSON 保存和最终清理均在正式计时之外。回放计时覆盖提交、调度、模型执行、
采样、结果回收和循环内日志；延迟入口额外记录每个 token 在本地 Engine 可见的时间。
正式性能运行不启用 Nsight。

吞吐使用实际输出 token 数除以回放耗时，报告三次 TPS 的算术平均与**总体标准差**。
延迟合并三轮原始样本后计算均值与分位数，分位数采用线性插值，不取三次 P99 的平均。

TTFT 从 `add_request()` 前开始计时，包含提交后的等待；尚在 benchmark 侧等待
活跃名额的样本不计入请求延迟，也不包含 HTTP 网络与文本解码。每请求固定输出
128 token，因此 `TPOT = (末 token 时间 - 首 token 时间) / 127`。
完整指标定义见 [Benchmark 文档](../../benchmarks/README.md)。



## 吞吐结果

单位为**整个模型副本合计的输出 token/s**，TP=2 的结果已经包含两张卡的协作，
不再乘以卡数。延迟入口的 TPS 来自另外的独立运行。

| 入口 | TP / 通信路径 | 第 1 轮 | 第 2 轮 | 第 3 轮 | 平均 ± 标准差 |
| --- | --- | ---: | ---: | ---: | ---: |
| 吞吐 | 1 / 无 TP 通信 | 3304.31 | 3321.13 | 3316.55 | **3313.99 ± 7.10** |
| 吞吐 | 2 / SHM | 2616.46 | 2651.84 | 2594.15 | **2620.82 ± 23.75** |
| 延迟 | 1 / 无 TP 通信 | 3316.67 | 3320.16 | 3317.27 | **3318.03 ± 1.52** |
| 延迟 | 2 / SHM | 2606.85 | 2600.20 | 2661.41 | **2622.82 ± 27.42** |

按吞吐入口的三次均值计算，TP=2 / TP=1 为 **0.7908×**，输出吞吐下降 **20.92%**。
两个入口、每一轮都观察到 TP=2 更慢，性能变化大于本轮三次测量的运行间波动。

## 延迟结果

每个 TP 配置合并三轮，共 **3072 个请求**、**390144 个 ITL 间隔**。
下表单位均为 ms。

| 指标 | TP / 通信路径 | 平均 | P50 | P95 | P99 |
| --- | --- | ---: | ---: | ---: | ---: |
| TTFT | 1 | 30.424 | 30.308 | 31.812 | 34.276 |
| TTFT | 2 / SHM | 38.322 | 37.901 | 41.124 | 45.951 |
| TPOT | 1 | 4.619 | 4.613 | 4.665 | 4.678 |
| TPOT | 2 / SHM | 5.846 | 5.905 | 6.047 | 6.085 |
| ITL | 1 | 4.619 | 4.578 | 4.903 | 5.068 |
| ITL | 2 / SHM | 5.846 | 5.863 | 6.500 | 6.740 |
| E2EL | 1 | 617.038 | 615.995 | 623.051 | 624.679 |
| E2EL | 2 / SHM | 780.733 | 787.075 | 806.170 | 814.667 |

TP=2 的平均 TTFT 增加 **25.96%**，平均 TPOT 增加 **26.56%**，平均端到端延迟
增加 **26.53%**；首 token 与持续生成阶段都没有获得延迟收益。
固定输出长度使 TPOT 与 ITL 的均值相等，但两者分别以请求和 token 间隔为样本，
分位数不必相同。

## Nsight 通信分析

### 采集方式与统计口径

使用 [profile_tensor_parallel.py](../../benchmarks/profile_tensor_parallel.py)
分别创建 TP=1/2 Engine，回放相同的 1024 条请求、每条输出 128 token。
合成预热后，通过 RPC 在每个 Worker 调用 `cudaProfilerStart()`；回放并排空在途队列后
调用 `cudaProfilerStop()`。主进程的 `tp_profile` NVTX 范围作为跨进程的统一统计窗口，
`--cuda-graph-trace=node` 用于记录 Graph 内部的 NCCL kernel。

采集有额外开销，下面的回放时间不替代正式 benchmark 的 TPS 或延迟。
[分析脚本](../../benchmarks/analyze_tensor_parallel.py) 使用两种通信占比：

- **kernel 累计占比**：所有 GPU 的 NCCL kernel 时长之和，除以所有 GPU kernel
  时长之和。跨 GPU、跨 stream 的重叠时间可以重复累计。
- **墙钟占比**：所有 GPU 的 NCCL 活动区间取并集，再除以回放墙钟。
  两卡同时通信只计一次，表示至少一张卡执行 NCCL 的时间比例。

NCCL kernel 时间包含传输、协议处理、同步与等待，**不能直接当作纯 PCIe 传输时间，
也不能当作可全部消除的端到端损失**。普通 H2D/D2H memcpy、CPU/Gloo、ZMQ/RPC
不在 NCCL 分子内。跨边界的 kernel 裁剪到 NVTX 窗口；窗口包含最后的队列排空，
可能略长于 benchmark 的 `elapsed`。

### 通信时间与调用次数

| 配置 | 回放墙钟 | GPU kernel 数 | NCCL kernel 数 | GraphLaunch 数 | NCCL 累计占比 | NCCL 墙钟占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TP=1 | 41.364704 s | 3625344 | 0 | 8128 | 0.00% | 0.00% |
| TP=2，SHM | 52.641858 s | 7860480 | 950272 | 16256 | **43.95%** | **49.61%** |

TP=2 的 GPU kernel 累计 **90.698996 s**，其中 NCCL 累计 **39.866235 s**；
两者相除得到 43.95%。两卡 NCCL 时间区间的并集为 **26.114793 s**，
除以 52.641858 s 得到 49.61%。

| GPU | kernel 累计时间 | NCCL 累计时间 | NCCL 累计占比 | NCCL 墙钟占比 | Graph 内 NCCL 数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 45.203556 s | 19.859978 s | 43.93% | 37.73% | 463296 |
| 1 | 45.495440 s | 20.006256 s | 43.97% | 38.00% | 463296 |

逐 GPU 的墙钟占比也使用整个回放窗口作分母。两卡 NCCL 活动有重叠，
不能将 37.73% 与 38.00% 相加作为全局墙钟占比。

| NCCL kernel | 两卡合计次数 | 累计时间 | 平均每次 |
| --- | ---: | ---: | ---: |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 933888 | 34.306888 s | 36.736 μs |
| `ncclDevKernel_AllGather_RING_LL` | 16384 | 5.559346 s | 339.316 μs |

原测试以 `nsys stats --filter-time=开始纳秒/结束纳秒` 交叉核对同一全局窗口，
kernel 数、NCCL 数及累计时间与分析脚本一致。没有直接使用父进程的
`--filter-nvtx` 过滤 Worker 的 GPU 活动，以免漏掉子进程 kernel。

### 为什么这组负载的 TP=2 更慢

Qwen3-0.6B 有 28 个 decoder layer。每层 Attention 输出投影与 MLP down projection
各执行一次 AllReduce，embedding 另有一次，因此**每个 rank、每次 forward 共
57 次 AllReduce**；logits 还需一次 AllGather。相关实现分别位于
[RowParallelLinear](../../nanovllm/model_executor/layers/linear.py)、
[VocabParallelEmbedding](../../nanovllm/model_executor/layers/vocab_parallel_embedding.py)
与 [LogitsProcessor](../../nanovllm/model_executor/layers/logits_processor.py)。

TP 将部分矩阵运算分摊到两张卡，同时在执行路径上加入集合通信。本组是小模型、
短输入、16 条活跃请求的 decode 为主负载，分片后单卡计算量减小，通信调用仍然频繁。
结合较高的 NCCL 时间占比，**本轮计算分摊的收益没有抵消新增的通信与协调开销**，
是与观测一致的解释。

但本次没有隔离后台任务，也没有获得成功的 P2P 对照，因此无法将吞吐差全部归因于
SHM、跨 NUMA 或某一个通信 kernel，更不能由 NCCL 时间占比推断 PCIe 带宽已经饱和。



## 结果适用范围

- **当前负载优先使用 TP=1。** Qwen3-0.6B 可以单卡容纳，本轮双卡 TP 的吞吐与延迟均更差。
- **TP 的性能收益需要按模型、并发和拓扑重新测量。** 更大的模型、更高并发或 NVLink
  环境会改变计算与通信的比例，本轮结果不能推广为 TP=2 总是更慢。
- **本轮没有测试双副本吞吐。** TP=2 只有一个模型副本，不能据此推断两张卡分别运行
  TP=1 副本时的服务总吞吐。
- **本轮只比较固定开启 Graph、异步调度和前缀缓存时的 TP 大小。** 没有测量 TP
  与这些开关的交互收益，不能把之前单卡文档中的提升比例直接相乘。

