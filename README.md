<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Nano-vLLM：Qwen3 推理与自定义算子实验

基于 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 继续实现的轻量推理项目，参考 vLLM 的配置、执行器、Worker 与算子组织方式。当前重点是读清推理执行链、接入 Triton / CuTe DSL kernel，并通过独立测试比较实现。

## 当前实现范围

| 部分 | 已接通的功能 |
| --- | --- |
| 模型与加载 | 本地 Hugging Face Qwen3 普通 decoder，加载 safetensors 权重。 |
| 执行端 | 单进程、单 CUDA GPU 的 `UniProcExecutor`。 |
| 调度与缓存 | 同步连续批处理、分块 prefill、分页 KV cache、prefix cache。 |
| Attention | FlashAttention-2，支持混合 prefill/decode 与 GQA。 |
| CUDA Graph | 单 token decode 使用 `FULL_DECODE_ONLY`，prefill 和混合 batch 使用 eager。 |
| 自定义算子 | SiLU × Mul 提供 `torch`、`triton`、`cutedsl` 三种实现，两个自定义后端支持自动调优。 |
| 性能测试 | 本地吞吐、逐 token 延迟，以及算子数值与性能对比。 |

**TP 多进程执行尚未接通**：并行配置、部分权重分片与通信代码已经存在，端到端启动当前仍要求 `tensor_parallel_size=1`。后续将继续扩展 TP 执行器与自定义算子。

## 安装

使用 Linux 或 WSL2、Python 3.10～3.12，以及支持当前 FA2 路径的 NVIDIA GPU。完整步骤与环境检查见 [安装指南](docs/installation.md)。

先按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 安装适合驱动的 CUDA 版 PyTorch，再在本仓库根目录执行：

```shell
python -m pip install packaging setuptools wheel ninja
python -m pip install "flash-attn>=2.8.3,<2.9" --no-build-isolation
python -m pip install -e .
```

PyTorch 与 Transformers 保留最低版本约束，具体依赖以 [pyproject.toml](pyproject.toml) 为准，不锁死某个 PyTorch/CUDA wheel 组合。已有兼容环境可以直接复用。

| 安装选项 | 命令 | 增加的功能 |
| --- | --- | --- |
| 基础安装 | `python -m pip install -e .` | 单卡推理、FA2、Triton 与默认 PyTorch 算子。 |
| CuTe DSL | `python -m pip install -e ".[cutedsl]"` | 安装 `nvidia-cutlass-dsl`，启用 CuTe 算子后端。 |
| Benchmark | `python -m pip install -e ".[benchmark]"` | 安装 PyArrow，读取本地 Parquet 数据集。 |
| 两者一起 | `python -m pip install -e ".[cutedsl,benchmark]"` | 同时启用上述可选组件。 |

习惯 requirements 入口时，在完成 PyTorch 与 FA2 安装后执行 `python -m pip install -r requirements.txt`，它复用同一份项目依赖声明。

## 下载模型与首次运行

先准备本地模型。下面将 Qwen3-0.6B 下载到用户目录：

```shell
hf download Qwen/Qwen3-0.6B --local-dir "$HOME/models/Qwen3-0.6B"
```

通过 [example.py](example.py) 运行一组短文本生成。示例会应用 Qwen3 chat template、关闭 thinking，并在退出时清理 Engine：

```shell
VLLM_HOST_IP=127.0.0.1 python example.py \
    --model "$HOME/models/Qwen3-0.6B" \
    --gpu-memory-utilization 0.5 \
    --max-tokens 32
```

默认使用 eager，便于先验证权重、Attention 与采样流程。增加 `--cuda-graph` 可启用 decode 图捕获；首次捕获还会编译项目的 host C++ 扩展，需要 C++ 编译器与 Ninja，详见安装指南。

## 配置与算子后端

Python 调用方组装 `NanovllmConfig`，通过 `LLM(vllm_config=...)` 创建推理实例。模型路径、调度容量、显存预算和算子后端分别由子配置管理：

```python
from pathlib import Path

from nanovllm import LLM, SamplingParams
from nanovllm.config import (
    CacheConfig,
    KernelConfig,
    ModelConfig,
    NanovllmConfig,
    SchedulerConfig,
)

config = NanovllmConfig(
    model_config=ModelConfig(
        model=str(Path.home() / "models/Qwen3-0.6B"),
        enforce_eager=True,
        max_model_len=1024,
    ),
    scheduler_config=SchedulerConfig(
        max_num_batched_tokens=256,
        max_num_seqs=4,
    ),
    cache_config=CacheConfig(gpu_memory_utilization=0.5),
    kernel_config=KernelConfig(silu_and_mul_backend="triton"),
)
llm = LLM(vllm_config=config)
try:
    outputs = llm.generate(
        ["Hello, Nano-vLLM."],
        SamplingParams(temperature=0.6, max_tokens=32),
    )
    print(outputs[0]["text"])
finally:
    llm.close()
```

`silu_and_mul_backend` 可选 `torch`、`triton`、`cutedsl`，默认 `torch` 使用 `torch.compile`。后端在模型层构造时绑定；选用 `cutedsl` 前安装对应 extra。

## 测试与文档

在项目根目录执行算子测试：

```shell
python tests/test_ops/test_silu_and_mul.py --dtypes float16
```

该命令比较三个后端，需要安装 CuTe extra；仅测试基础安装可加 `--backends torch triton`。测试通过原有层入口执行，编译、调优与数值检查在正式计时之外，不捕获 CUDA Graph。

| 入口 | 内容 |
| --- | --- |
| [安装指南](docs/installation.md) | 依赖分组、工具链、安装顺序与环境验证。 |
| [实现文档索引](docs/README.md) | Executor、Worker、Attention、CUDA Graph 与 kernel 笔记。 |
| [Benchmark](benchmarks/README.md) | 本地 Alpaca / LongBench 吞吐与延迟测试。 |
| [SiLU × Mul 后端与性能](docs/ir/ops/activation.md) | 三种实现的测试口径及历史实测。 |

性能应结合 GPU、输入形状、精度和执行方式测量，记录各自的预热与缓存条件。

## 上游项目与许可证

项目基于 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)，执行协议与部分接口参考 [vLLM](https://github.com/vllm-project/vllm)。许可证见 [LICENSE](LICENSE)，移植文件保留各自的版权和许可证声明。
