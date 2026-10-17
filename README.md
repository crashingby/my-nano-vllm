<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

先组装 `NanovllmConfig`，再将完整配置传给 `LLM`。模型路径和执行选项由
`ModelConfig` 管理，批处理容量由 `SchedulerConfig` 管理；完整示例见 `example.py`。

```python
from nanovllm import LLM, SamplingParams
from nanovllm.config import ModelConfig, NanovllmConfig, SchedulerConfig

vllm_config = NanovllmConfig(
    model_config=ModelConfig(
        model="/YOUR/MODEL/PATH",
        enforce_eager=True,
        max_model_len=1024,
    ),
    scheduler_config=SchedulerConfig(
        max_num_batched_tokens=256,
        max_num_seqs=4,
    ),
)
llm = LLM(vllm_config=vllm_config)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
try:
    outputs = llm.generate(prompts, sampling_params)
    print(outputs[0]["text"])
finally:
    llm.close()
```

`cache_config`、`parallel_config`、`compilation_config`、`load_config` 和
`device_config` 同样通过 `NanovllmConfig` 传入，省略时使用各自的默认配置。
Engine、Executor 和 Worker 共享该配置对象，初始化得到的 EOS、KV 容量等状态也会
回写到这个对象。

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
