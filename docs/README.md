# 实现文档索引

文档按执行流程与实现层次组织。仓库内源码链接使用相对路径，上游参考源码链接固定到核对过的版本。

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

端到端吞吐与延迟测试见 [Benchmark 文档](../benchmarks/README.md)。算子测试位于
[`tests/test_ops`](../tests/test_ops)，在项目根目录运行：

```shell
python tests/test_ops/test_silu_and_mul.py --dtypes float16
```
