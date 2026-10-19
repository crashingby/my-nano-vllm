# silu_and_mul：后端与性能

[`silu_and_mul`](../../../nanovllm/ir/ops/activation.py) 将输入 `[..., 2H]` 等分为 gate 和 up，返回 `SiLU(gate) * up`，输出形状为 `[..., H]`。计算原理及模型中的调用位置见 [SiluAndMul 层](../../model_executor/layers/activation.md)。

## 三种后端

模型通过 `kernel_config.silu_and_mul_backend` 或 `SiluAndMul(backend=...)` 选择后端。当前提供三种实现，**配置中的 `torch` 对应 IR 中的 `native`**：

| 配置后端 | IR 实现名 | 实现方式 |
| --- | --- | --- |
| `torch` | `native` | 使用 PyTorch 的 `F.silu(gate) * up`，由[模型层](../../../nanovllm/model_executor/layers/activation.py)共享 `torch.compile` 包装。 |
| `triton` | `triton` | 使用 [Triton 融合 kernel](../../../nanovllm/kernels/triton/activation.py)，通过 `triton.autotune` 选择行块、列块和 warp 数配置，详见[实现思路](../../kernels/triton/activation/SiluAndMul.md)。 |
| `cutedsl` | `cutedsl` | 使用 [CuTe DSL 融合 kernel](../../../nanovllm/kernels/cute/activation.py)，通过 `from_dlpack` 接收 Tensor，使用官方 `autotune_jit` 搜索每 CTA 行数和每线程元素数。 |

- Triton 与 CuTe 都采用 **persistent kernel（持久化 kernel）**：网格上限为 SM 数的四倍，额外行块由 kernel 内循环处理。
- 两个自定义后端当前支持连续的 CUDA FP16、BF16、FP32 输入，中间计算使用 FP32，写回时转换为输出精度。
- 首次调用可能触发编译和自动调优；后续相同条件复用已选配置。模型层在构造时绑定后端，普通 `forward()` 不重新选择后端。

## 性能测试方式

以下记录 **2026-10-19 的一次实测**，CuTe 使用 `from_dlpack` 和官方自动调优版本，Triton 当时搜索三种候选。当前 Triton 已扩展为 36 种候选，下表保留扩展前的横向对比记录。

- **环境**：WSL，`nano-vllm` Conda 环境，NVIDIA GeForce RTX 4060 Laptop GPU，24 个 SM，PyTorch `2.5.1+cu121`。
- **输入输出**：输入 `(4096, 8192)`，输出 `(4096, 4096)`，即 $T=4096$、$H=4096$。
- **执行方式**：直接调用原有 `layer(x)` 入口，**不捕获 CUDA Graph**；`torch` 后端使用 `torch.compile`。
- **计时方式**：预热 10 次，每个样本调用 50 次，采样 10 轮，使用 CUDA Event 折算每调用耗时，报告中位数。编译、自动调优、预热和数值校验均在计时之外。
- **缓存与发射**：重复使用相同输入，允许 GPU cache 复用；计时区间可能包含 Python 发射造成的 GPU 空闲时间。

在项目根目录、`nano-vllm` 环境中运行[测试脚本](../../../tests/test_ops/test_silu_and_mul.py)：

```shell
python tests/test_ops/test_silu_and_mul.py \
    --dtypes float16 bfloat16 float32 \
    --num-tokens 4096 --hidden-size 4096 \
    --warmup 10 --iters 50 --repeats 10 --seed 0
```

## 实测结果

单位为 **μs/次调用**，数值越小越快。

| 精度 | torch | triton | cutedsl |
| --- | ---: | ---: | ---: |
| FP16 | 467.514 | 458.250 | 463.893 |
| BF16 | 478.044 | 466.442 | 478.300 |
| FP32 | 934.871 | 940.039 | 932.925 |

全部 9 组输出均通过 CPU FP64 参考值校验，测试正常退出。

**本组输入下，三种后端性能接近**：Triton 在 FP16、BF16 下略快，FP32 的三个结果相差不到 1%。当前自定义 kernel 相对已编译的 PyTorch 实现没有明显的大幅加速，其他尺寸应以对应的实测结果为准。
