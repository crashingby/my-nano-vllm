# 实现文档索引

文档按执行流程与实现层次组织。仓库内源码链接使用相对路径，上游参考源码链接固定到核对过的版本。

首次运行先阅读 [安装与运行环境](installation.md)，完成依赖安装、GPU 检查和算子验证。

## Engine 与请求驱动

| 文档 | 阅读内容 |
| --- | --- |
| [LLMEngine](engine/LLMEngine.md) | 请求入口、step_fn 的同步与异步路径、generate 的增量累计及资源收尾。 |
| [SchedulerOutput](core/sched/output-zh.md) | Scheduler 下发的执行计划及各字段的语义。 |
| [异步调度执行链](core/sched/AsyncScheduler-zh.md) | 输出占位、跨批次依赖、GPU token 回填与输入/输出同步。 |

## 执行器与 Worker

| 文档 | 阅读内容 |
| --- | --- |
| [Executor](executor/Executor.md) | 执行端公共协议、KV 初始化与输出交付。 |
| [UniProcExecutor](executor/UniProcExecutor.md) | 当前进程内的 Worker 生命周期与本地调用。 |
| [GPUModelRunner](worker/gpu_model_runner.md) | 请求同步、packed 输入、KV 初始化、采样、profiling 与图执行。 |
| [InputBatch](worker/InputBatch.md) | 持久请求槽位、行对齐与布局更新。 |
| [BlockTable](worker/block_table.md) | manager block、kernel page 和 slot mapping 的转换。 |
| [BatchUpdate](sample/BatchUpdate.md) | logits processor 随请求槽位变化的状态同步协议。 |

## Attention 与 CUDA Graph

| 文档 | 阅读内容 |
| --- | --- |
| [Attention 后端抽象](attention/backend.md) | Backend、Builder、Impl 与模型层的职责及对象关系。 |
| [FlashAttention 后端](attention/flash_attn.md) | FA2 的 metadata、缓存写入与分页计算。 |
| [CUDA 与 FA 基础接口](attention/fa_utils.md) | 平台查询、FA2 适配层与 Triton 缓存写入接口。 |
| [CUDA Graph 总览](cudagraph/overview.md) | 捕获与回放原理、FA2 约束及当前执行范围。 |
| [CUDAGraphMode](cudagraph/CUDAGraphMode.md) | 配置策略与每轮运行模式。 |
| [CudagraphDispatcher](cudagraph/CudagraphDispatcher.md) | 合法档位、捕获计划与在线分派。 |
| [BatchDescriptor](cudagraph/BatchDescriptor.md) | 图布局描述符、固定 buffer 与补齐数据的完整例子。 |
| [CUDAGraphWrapper](cudagraph/CUDAGraphWrapper.md) | 图缓存、实际捕获、回放及输出存储生命周期。 |

## 模型层、算子与 kernel

按“模型层语义 → 算子后端 → kernel 实现”阅读：

| 层次 | 文档 |
| --- | --- |
| 模型层 | [SiluAndMul 的原理与调用位置](model_executor/layers/activation.md)。 |
| 算子 | [silu_and_mul 的三种后端与性能记录](ir/ops/activation.md)。 |
| kernel | [Triton SiluAndMul 的持久化调度与自动调优](kernels/triton/activation/SiluAndMul.md)。 |
| kernel | [CuTe DSL SiluAndMul 的线程映射、访存与官方调优](kernels/cute/activation/SiluAndMul.md)。 |

## 性能记录

| 文档 | 对比内容 |
| --- | --- |
| [CUDA Graph 与 eager](performance/cuda_graph.md) | 关闭异步调度，记录固定负载下的吞吐、首 token 延迟与逐 token 延迟。 |
| [异步调度的性能与重叠](performance/async_schedule.md) | 分别比较 eager、CUDA Graph 下的同步与异步调度，用 Nsight 量化跨批次 CPU/GPU 重叠。 |
| [前缀缓存收益](performance/prefix_cache.md) | 核对 Qwen 模板，对比短输入与共享长 system 下缓存开关的计算量、吞吐和延迟。 |
| [单机 TP 的性能与通信](performance/tensor_parallel.md) | 在 Alpaca 上比较 TP=1/2，开启 Graph、异步调度与前缀缓存，用 Nsight 记录 NCCL 和实际传输路径。 |

端到端吞吐与延迟测试的用法见 [Benchmark 文档](../benchmarks/README.md)。算子测试位于
[`tests/test_ops`](../tests/test_ops)，在项目根目录运行：

```shell
python tests/test_ops/test_silu_and_mul.py --dtypes float16
```
