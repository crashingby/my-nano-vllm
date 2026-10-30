# 单机 TP 的性能与通信实测

## 本轮结论

在两张 NVIDIA A30 上， Alpaca 负载
**1024 条真实请求**，每条固定生成 128 token，每轮输出 **131072 token**。
CUDA Graph、异步调度和前缀缓存全部开启；TP=1 与 TP=2 均使用 `mp` 执行器。

- **强制 P2P 未跑通**：设置 `NCCL_P2P_LEVEL=SYS` 后，NCCL 日志确实切换为
  `P2P/CUMEM`，但最小 BF16 集合通信在两端未能完整完成，触发 watchdog 超时。
  传统 `P2P/IPC` 和 `Simple` 协议探针也超时。因此本轮没有有效的 P2P 推理
  TPS 或延迟，也无法判断它比 SHM 快多少。
- **默认 SHM 路径可以正常运行**。相同探针通过 AllReduce、AllGather 和 CUDA
  Graph 回放的数据校验；随后完成三轮 TP=1/2 吞吐、三轮 TP=1/2 延迟，以及
  两份 1024 请求的完整 Nsight 采集。
- 默认路径下，TP=1 输出吞吐为 **3313.99 ± 7.10 token/s**，
  TP=2 为 **2620.82 ± 23.75 token/s**，TP=2/TP=1 为
  **0.7908×**，下降 **20.92%**。
- 平均 TTFT：TP=1 **30.424 ms**，TP=2
  **38.322 ms**；平均 TPOT 分别为
  **4.619 ms**、**5.846 ms**。
- 完整 TP=2 trace 中，NCCL 占 GPU kernel 累计时间的
  **43.95%**；两卡通信时间区间取并集后，占回放
  墙钟 **49.61%**。这些值包含 NCCL 协议和等待，
  不表示纯 PCIe 传输耗时。

上述结果来自共享 GPU 环境。模型启动前，两卡各已使用约 13929 MiB 显存，
后台进程持续运行，整卡利用率约 25%。实验未停止这些进程；不能把性能变化全部归因于
PCIe 或 SHM，也不能据此推断链路带宽已经饱和。

原始数据：[summary.json](../../tmp/tp_alpaca_p2p_20261030/summary.json)、[逐轮 runs.csv](../../tmp/tp_alpaca_p2p_20261030/runs.csv)、
[完整命令与状态](../../tmp/tp_alpaca_p2p_20261030/commands.json)。全部报告和数据位于项目 `tmp` 目录。

## P2P 实际验证

### 能力报告与实际传输

`nvidia-smi topo -p2p r/w` 对两卡两个方向均报告 `OK`；两个 Worker 的
`torch.cuda.can_device_access_peer()` 也都返回 `True`。这些能力检查没有保证本次
NCCL 传输一定能完成。

本机为双路 Intel Xeon，GPU 间拓扑为跨 NUMA 的 `SYS`。对应版本 NCCL 2.21.5
的 `ncclTopoCheckP2p()` 在 Intel x86 上默认以 `PXB` 为 P2P 距离上限，因而默认
排除这条 `SYS` 路径；设置 `NCCL_P2P_LEVEL=SYS` 放宽距离筛选后才会选择 P2P。
这与驱动报告支持 P2P 并不矛盾。源码见
[NCCL 2.21.5 paths.cc](https://github.com/NVIDIA/nccl/blob/v2.21.5-1/src/graph/paths.cc#L256-L278)。

SHM 是主机内存中转的传输方式，仍然涉及 PCIe，并可能经过跨 CPU 互连；它不是
CUDA kernel 的 shared memory。`SYS` 描述物理拓扑，`SHM`/`P2P` 描述 NCCL
选择的传输机制。见 [NCCL 参数说明](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2215/user-guide/docs/env.html#nccl-shm-disable)。

### 最小探针结果

探针通过两个独立进程执行 BF16 AllReduce 与 AllGather。计划测试 128、16384、
1048576 个元素，并对每个规模执行 10 次 CUDA Graph 回放；只有数据逐元素校验
通过才写成功结果。失败探针在最小规模即受阻，未进入后续完整 Graph 校验。

| 配置 | NCCL 实际路径 | 结果 |
| --- | --- | --- |
| 默认配置 | `SHM/direct/direct` | 三种规模的通信和 Graph 回放均正确，修正探针清理后正常退出。 |
| `NCCL_P2P_LEVEL=SYS` | `P2P/CUMEM` | 最小集合通信未完整完成，60 秒 watchdog 超时。 |
| `SYS` + `NCCL_CUMEM_ENABLE=0` | `P2P/IPC` | 最小集合通信未完整完成，20 秒 watchdog 超时。 |
| `SYS` + `NCCL_PROTO=Simple` | `P2P/CUMEM` | 最小集合通信未完整完成，20 秒 watchdog 超时。 |

证据：[探针清单及超时行](../../tmp/tp_alpaca_p2p_20261030/probe_manifest.json)、
[默认 SHM 正常退出日志](../../tmp/tp_alpaca_p2p_20261030/probe_shm_clean.log)、
[默认 P2P 超时日志](../../tmp/tp_alpaca_p2p_20261030/probe.log)、[IPC 超时日志](../../tmp/tp_alpaca_p2p_20261030/probe_p2p_ipc.log)、
[Simple 超时日志](../../tmp/tp_alpaca_p2p_20261030/probe_p2p_simple.log)。

另对 P2P 探针采集了 15 秒的 Nsight 诊断，结束时主动终止该探针进程组：
[诊断 .nsys-rep](../../tmp/tp_alpaca_p2p_20261030/profile_p2p_failed.nsys-rep)、
[SQLite](../../tmp/tp_alpaca_p2p_20261030/profile_p2p_failed.sqlite)、[诊断说明](../../tmp/tp_alpaca_p2p_20261030/p2p_failed_trace_summary.json)。
未完成的 GPU 活动可能不出现在导出表中，因此不能从该文件计算有效吞吐或通信
占比，也不能把导出表中少量已经完成的 kernel 当成整段通信已经完成。

准确结论是：**当前机器、驱动和 NCCL 组合下，这几种强制 P2P 配置没有通过实际
通信验证；默认 SHM 是本轮验证可用的路径。** 本轮没有证明 GPU 硬件永远无法
使用 P2P，也没有定位超时的最终根因。NVIDIA 文档列出的潜在影响包括 PCIe
ACS、IOMMU、虚拟化和 BIOS 配置，后续需以实际带宽/正确性测试继续定位；这些
只是排查方向，本轮没有修改系统配置。见
[NVIDIA GPU Direct 排查说明](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2215/user-guide/docs/troubleshooting.html#gpu-direct)。

## 硬件与软件

| 项目 | 本次配置 |
| --- | --- |
| GPU | 2 × NVIDIA A30，每卡 24576 MiB，无 NVLink。 |
| PCIe / 拓扑 | 两卡均为 Gen4 ×16；GPU 间 `SYS`，跨 CPU / NUMA。 |
| CPU | 2 × Intel Xeon Gold 6330，112 个逻辑 CPU，2 个 NUMA node。 |
| 驱动 | 550.163.01。 |
| Python / PyTorch | 3.10.20 / 2.5.1+cu124。 |
| Triton / FlashAttention | 3.1.0 / 2.8.3.post2，使用 FA2。 |
| Transformers / NCCL | 4.57.6 / 2.21.5。 |
| Nsight Systems | TP=1、TP=2 和失败诊断均使用同一套 2024.2.3.38。 |
| 执行器 | TP=1/2 都使用 `mp` + `spawn`。 |
| 线程 | `OMP_NUM_THREADS=1`，未额外绑核。 |
| 源码基线 | `0ed68b26f2f10255fd278d09e48deaaa8f864f8e`，加已有 benchmark 改动。 |

见 [硬件记录](../../tmp/tp_alpaca_p2p_20261030/hardware_measurements.txt)、[环境与数据集 SHA-256](../../tmp/tp_alpaca_p2p_20261030/environment_measurements.json)、
[测量期每秒 GPU 采样](../../tmp/tp_alpaca_p2p_20261030/gpu_telemetry_measurements.csv) 和
[采集期每秒 GPU 采样](../../tmp/tp_alpaca_p2p_20261030/gpu_telemetry_profiles.csv)。采样记录的是整卡占用，
包含后台任务，不能视为本模型进程独占的利用率。

## 数据与测量协议

| 项目 | 取值与含义 |
| --- | --- |
| 模型 | `~/Models/Qwen3-0.6B`，BF16。 |
| 数据文件 | `~/datasets/alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet`。 |
| 样本 | 按源文件顺序取前 1024 条，不重排、不重复、不跳过，本批无截断。 |
| 对话模板 | Qwen3 chat template，关闭 thinking，不额外添加共享 system prompt。 |
| 输入 token | 合计 30123；最小 17、平均 29.417、中位数 26.0、最大 135。 |
| 输出 | `temperature=0.6`、`--ignore-eos`，每条 128 token，每轮 131072 token。 |
| 并发与预算 | 活跃请求最多 16，单轮 token 预算 1024。 |
| 上下文 / 输入上限 | 1024 / 512 token。 |
| CUDA Graph | `FULL_DECODE_ONLY`；uniform decode 使用完整 Graph，prefill/混合批次走 eager。 |
| 异步调度 | 开启，通过 `step_fn()` 提交与回收在途批次。 |
| 前缀缓存 | 开启，block size 256；下文单列实际查询与命中。 |
| 显存预算 | 每张使用中的 GPU 按总显存的 0.2 规划。 |
| 预热 | 每次独立 Engine 执行 16 条合成请求，输入为 128 个 token 100，输出 16 token。 |
| RNG | 预热后通过 Worker RPC 重置为 seed 0，不保证跨 TP 逐 token 一致。 |
| 重复次数 | 吞吐和延迟各三轮，TP=1/2 各自重建 Engine，总计 12 次正式测量。 |
| 对照顺序 | 每轮分别完成吞吐和延迟；TP 顺序为第 1 轮 1→2，第 2 轮 2→1，第 3 轮 1→2。 |
| NCCL 配置 | 正式性能测试清除 P2P 距离及 P2P/SHM 禁用开关，实际 TP=2 channel 均为 SHM。 |

所有正式运行的输入 token SHA-256 相同，输出计数都为 131072，实际 scheduled
token 都为 **160171**。数据编码、模型加载、Graph 捕获、
合成预热、JSON 保存和最终清理均在正式计时之外。计时覆盖提交、调度、执行、
结果回收、循环内日志；延迟入口还覆盖逐 token 时间戳观察。

TTFT 从请求实际提交到 Engine 前开始计时，不包含仍在 benchmark 侧等待名额的
样本；结果是本地 Engine 延迟，不包含 HTTP 网络。固定 128 个输出 token 时，
`TPOT = (最后 token 时间 - 首 token 时间) / 127`。

### 前缀缓存的实际效果

| 组别 | 每轮请求数 | 每轮查询 token | 每轮命中 token | 抢占请求数 |
| --- | ---: | ---: | --- | --- |
| `throughput_tp1` | 1024 | 30123 | 0, 0, 0 | 0, 0, 0 |
| `throughput_tp2_shm` | 1024 | 30123 | 0, 0, 0 | 0, 0, 0 |
| `latency_tp1` | 1024 | 30123 | 0, 0, 0 | 0, 0, 0 |
| `latency_tp2_shm` | 1024 | 30123 | 0, 0, 0 | 0, 0, 0 |

缓存开关开启不等于自然数据一定命中；本表按正式计时窗口重置统计，合成预热
不使用正式样本。本组负载不用于推断长共享前缀的缓存收益。

## 吞吐结果

单位为整个模型副本合计的输出 token/s。均值是三次 TPS 的算术平均，
“±”为三次测量的总体标准差。延迟入口是另外的独立运行。

| 入口 | TP / 路径 | 第 1 轮 | 第 2 轮 | 第 3 轮 | 平均 ± 标准差 |
| --- | --- | ---: | ---: | ---: | ---: |
| 吞吐 | 1 / 无 TP 通信 | 3304.31 | 3321.13 | 3316.55 | **3313.99 ± 7.10** |
| 吞吐 | 2 / SHM | 2616.46 | 2651.84 | 2594.15 | **2620.82 ± 23.75** |
| 延迟 | 1 / 无 TP 通信 | 3316.67 | 3320.16 | 3317.27 | **3318.03 ± 1.52** |
| 延迟 | 2 / SHM | 2606.85 | 2600.20 | 2661.41 | **2622.82 ± 27.42** |

P2P 因正确性探针超时，没有进入正式推理性能测量，结果为不可用，不能记成
0 token/s，也不能拿这轮 SHM 与旧 128 请求数据计算 P2P 收益。

## 延迟结果

每个 TP 配置合并三轮，共 **3072 个请求**和 **390144 个 ITL 样本**。
下表单位均为毫秒；分位数对原始样本排序后线性插值计算，没有平均各轮分位数。

| 指标 | TP / 路径 | 平均 | P50 | P95 | P99 |
| --- | --- | ---: | ---: | ---: | ---: |
| TTFT | 1 | 30.424 | 30.308 | 31.812 | 34.276 |
| TTFT | 2 / SHM | 38.322 | 37.901 | 41.124 | 45.951 |
| TPOT | 1 | 4.619 | 4.613 | 4.665 | 4.678 |
| TPOT | 2 / SHM | 5.846 | 5.905 | 6.047 | 6.085 |
| ITL | 1 | 4.619 | 4.578 | 4.903 | 5.068 |
| ITL | 2 / SHM | 5.846 | 5.863 | 6.500 | 6.740 |
| E2EL | 1 | 617.038 | 615.995 | 623.051 | 624.679 |
| E2EL | 2 / SHM | 780.733 | 787.075 | 806.170 | 814.667 |

TPOT 与 ITL 的均值在固定输出长度下相等，但分位数来自不同样本集合。

## Nsight 采集与通信占比

TP=1/2 使用同一套 Nsight Systems 2024.2.3，分别重新创建 Engine，回放
同样的 1024 请求。预热后在每个 Worker 调用 `cudaProfilerStart()`，回放
结束并排空队列后调用 `cudaProfilerStop()`；主进程 `tp_profile` NVTX 范围
限定全局统计窗口。使用 `--cuda-graph-trace=node` 记录 Graph 内 NCCL kernel。

采集引入额外开销，trace 的回放时间不用来替代前面的正式 TPS/延迟。

- **kernel 累计占比**：跨两卡 NCCL kernel 时长之和 / 跨两卡所有 GPU kernel
  时长之和。跨卡、跨 stream 的重叠时间可被重复累计。
- **墙钟占比**：所有卡 NCCL 活动区间取并集，再除以回放墙钟；两卡同时通信
  只计一次。这个比例同样不能直接解释为可消除的端到端损失。

NCCL 活动包含协议处理、同步与等待。普通 H2D/D2H memcpy、CPU/Gloo 和
ZMQ/RPC 不在 NCCL kernel 分子内。跨越边界的 kernel 裁剪到 NVTX 范围。
NVTX 范围包含最终在途队列排空，可能略长于 benchmark 的 `elapsed`。

TP=2 在 `cudaProfilerStop()` 刷新大量活动时出现过 60 秒的共享内存广播
等待提示。这是 Worker RPC 控制通道的等待，发生在回放结束之后；Worker
最终均正常退出。该等待不属于下表窗口内的 NCCL 通信，也未计入正式延迟。

| trace | 回放墙钟 | GPU kernel 数 | NCCL 数 | GraphLaunch 数 | NCCL 累计占比 | NCCL 墙钟占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `tp1` | 41.364704 s | 3625344 | 0 | 8128 | 0.00% | 0.00% |
| `tp2_shm` | 52.641858 s | 7860480 | 950272 | 16256 | 43.95% | 49.61% |

TP=2 的所有 GPU kernel 累计 **90.698996 s**，NCCL 累计
**39.866235 s**；NCCL 跨卡时间并集为
**26.114793 s**。前两项相除得到累计占比，
并集时间除以回放墙钟得到墙钟占比。

| GPU | kernel 累计时间 | NCCL 累计时间 | NCCL / kernel | NCCL / 墙钟 | Graph 内 NCCL 数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 45.203556 s | 19.859978 s | 43.93% | 37.73% | 463296 |
| 1 | 45.495440 s | 20.006256 s | 43.97% | 38.00% | 463296 |

| NCCL kernel | 两卡合计次数 | 累计时间 | 平均每次 |
| --- | ---: | ---: | ---: |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 933888 | 34.306888 s | 36.736 μs |
| `ncclDevKernel_AllGather_RING_LL` | 16384 | 5.559346 s | 339.316 μs |

模型有 28 个 decoder layer，每层 Attention 输出投影和 MLP down projection 各
一次 AllReduce，embedding 另一次，合计每个 rank 每次 forward 57 次 AllReduce；
logits 一次 AllGather。对应代码：[RowParallelLinear](../../nanovllm/model_executor/layers/linear.py)、
[VocabParallelEmbedding](../../nanovllm/model_executor/layers/vocab_parallel_embedding.py)、
[LogitsProcessor](../../nanovllm/model_executor/layers/logits_processor.py)。

对照 `nsys stats --filter-time=开始纳秒/结束纳秒` 导出的同一全局窗口，kernel 数、
NCCL 数及累计时间与自定义分析完全相同：
[交叉校验](../../tmp/tp_alpaca_p2p_20261030/analysis_validation.json)、[TP=1 复算](../../tmp/tp_alpaca_p2p_20261030/communication_tp1.json)、
[TP=2 复算](../../tmp/tp_alpaca_p2p_20261030/communication_tp2_shm.json)。没有用父进程的 `--filter-nvtx`
直接过滤子进程 GPU 活动，避免将 Worker 的 kernel 漏掉。

## 复现命令与原始文件

全部数据位于项目根目录的 [`tmp/tp_alpaca_p2p_20261030/`](../../tmp/tp_alpaca_p2p_20261030/)。
有效完整 trace：[TP=1 .nsys-rep](../../tmp/tp_alpaca_p2p_20261030/profile_tp1.nsys-rep)、
[TP=2 SHM .nsys-rep](../../tmp/tp_alpaca_p2p_20261030/profile_tp2_shm.nsys-rep)；对应 `.sqlite` 在相同目录。
Nsight 2024.2.3 沿用此前在项目 tmp 解包的官方工具，没有安装到系统或改动依赖。

以下命令复现默认 TP=2 的正式吞吐配置；延迟将模块换为
`benchmarks.benchmark_engine_latency`，TP=1 将并行数改为 1。`TMPDIR` 使用较短的
项目路径，避免 Unix socket 的 108 字节限制。

```shell
source .venv/bin/activate
env -u NCCL_P2P_LEVEL -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE \
  CUDA_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_HOST_IP=127.0.0.1 OMP_NUM_THREADS=1 TMPDIR="$PWD/tmp" \
  python -m benchmarks.benchmark_engine_throughput \
    --model-path "$HOME/Models/Qwen3-0.6B" \
    --dataset-name alpaca \
    --dataset-path "$HOME/datasets/alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet" \
    --tensor-parallel-size 2 --distributed-executor-backend mp \
    --cuda-graph --async-scheduling --enable-prefix-caching \
    --num-prompts 1024 --max-active 16 --max-tokens 128 --ignore-eos --seed 0 \
    --max-model-len 1024 --max-prompt-tokens 512 --long-prompt-policy truncate \
    --max-num-batched-tokens 1024 --gpu-memory-utilization 0.2 \
    --warmup-num-prompts 16 --warmup-output-tokens 16 \
    --result-path tmp/tp_repeat_1024.json
```

复测脚本及每条命令的环境变量见 [run_suite.py](../../tmp/tp_alpaca_p2p_20261030/run_suite.py) 和
[commands.json](../../tmp/tp_alpaca_p2p_20261030/commands.json)。采集复用
[profile_tensor_parallel.py](../../benchmarks/profile_tensor_parallel.py)，导出与交叉校验见
[analyze_suite.py](../../tmp/tp_alpaca_p2p_20261030/analyze_suite.py)。数据目录还保存原始逐请求/逐 token 延迟、
所有 rank 的 NCCL channel 日志、源码快照和 SHA-256。

启动阶段曾遇到临时路径过长及编译工具 PATH 缺失，两次都没有进入正式回放，
日志保留并排除在统计之外。初版 SHM 探针虽然数据校验通过，但未先释放 Graph
引用，清理进程组时超时；修正后重新验证正常退出，详见探针清单。

最初将 Nsight 内部临时流放到项目 NFS 时，两份报告都没有 CUDA/NVTX 活动，
已保存到 `invalid_nfs_profiles/` 并排除。文件时钟检查显示 NFS 的 mtime 落后
节点时间约 4793 秒，而本地文件的时间正常，见
[文件时钟诊断](../../tmp/tp_alpaca_p2p_20261030/filesystem_clock_diagnostic.json)。这一现象与 NVIDIA
工程师描述的“临时文件时间早于采集而被丢弃”问题相符，参见
[官方论坛排查记录](https://forums.developer.nvidia.com/t/nsys-cant-capture-anything-cuda-programs-only/338034/14)。
随后仅将采集器内部缓冲改到本地临时目录，结束后清理；最终保留的报告和数据
仍全部位于项目 tmp。重新采集后检查 NVTX、每个 GPU 的 kernel 和 Graph 活动，
并通过前述标准统计交叉校验，才纳入本报告。正式十二次性能结果未因此重测或混入采集开销。

旧 128 请求结果见 [上一版报告](../../tmp/tp_alpaca_20261030/tensor_parallel_128_requests.md)。
本轮性能表和通信占比只使用 1024 请求的新数据，没有混入旧测量或失败 P2P trace。
