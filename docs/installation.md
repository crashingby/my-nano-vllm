# 安装与运行环境

本项目通过 [pyproject.toml](../pyproject.toml) 统一声明依赖；[requirements.txt](../requirements.txt) 仅提供可编辑安装入口。先准备 CUDA 版 PyTorch 和 FA2，再安装项目与所需 extra，可以明确控制二进制扩展使用的环境。

## 环境要求

| 项目 | 当前要求 |
| --- | --- |
| 系统 | Linux；当前开发与测试包含 WSL2 环境，按 Linux 路径安装。 |
| Python | `>=3.10,<3.13`，对应 Python 3.10～3.12。 |
| GPU | NVIDIA CUDA GPU；项目的 FA2 路径要求计算能力至少 SM 8.0，模型 Attention 使用 FP16/BF16。 |
| PyTorch | CUDA 构建，满足项目最低 API 要求；按驱动与平台选择 wheel。 |
| C++ 工具链 | 启用 CUDA Graph 或 spinloop 等待扩展时，需要 C++ 编译器与 Ninja。 |
| CUDA Toolkit | 从源码构建 FlashAttention 时需要可用的 `nvcc` 与兼容工具链。 |

系统驱动、PyTorch 自带的 CUDA runtime，以及 `nvcc` 所属的 Toolkit 是不同层次。安装 PyTorch 时使用 [官方安装页](https://pytorch.org/get-started/locally/) 生成适合当前机器的命令，先确认 GPU 可用。

## 必需依赖与可选组件

| 依赖 | 用途 | 安装范围 |
| --- | --- | --- |
| `torch` | Tensor、矩阵乘、分布式通信、局部编译与 CUDA Graph。 | 必需。 |
| `transformers` | Qwen3 配置、tokenizer 与 chat template。 | 必需。 |
| `triton` | KV 写入、slot mapping，以及 Triton 算子后端。 | 必需；即使 SiLU 选择 `torch`，执行链仍使用 Triton。 |
| `flash-attn` | 提供当前 Attention 调用的 `flash_attn_2_cuda` 扩展。 | 必需。 |
| `numpy`、`pydantic` | CPU 输入准备、元数据与配置类型。 | 必需。 |
| `safetensors`、`tqdm` | 本地权重读取与进度显示。 | 必需。 |
| `typing-extensions`、`cloudpickle` | 类型兼容、执行方法序列化。 | 必需。 |
| `pyzmq` | 多 Worker 消息队列通知与超大消息传递。 | 必需。 |
| `ninja` | 构建按需加载的 LibTorch host 扩展。 | 基础安装包含。 |
| `nvidia-cutlass-dsl` | CuTe DSL 的编译与运行支持，导入名为 `cutlass`。 | `cutedsl` extra。 |
| `pyarrow` | 读取 Benchmark 的 Parquet 数据集。 | `benchmark` extra；JSONL 读取使用标准库。 |

项目显式声明直接使用的依赖，避免依赖 Transformers 或其他包恰好间接安装它们。当前 prefix hash 使用标准库 SHA-256，基础依赖中不再保留未使用的 `xxhash`。

### 版本约束的含义

- **PyTorch、Transformers、Triton**：保留最低版本约束，没有固定为开发机的确切版本。Triton 应与选用的 PyTorch 依赖保持兼容，不单独强制升级到另一版本。
- **FA2**：当前保留 `flash-attn>=2.8.3,<2.9`，因为适配层直接调用扩展的参数 ABI，升级时需要重新核对接口。
- **CuTe DSL**：当前保留 `nvidia-cutlass-dsl>=4.8.1,<4.9`，实现使用官方 `autotune_jit` 及其缓存属性，升级时需要核对调优接口。
- **Pydantic**：`SkipValidation` 属于 v2 API，因此声明 `pydantic>=2`。

这些范围说明当前代码的依赖条件，不代表范围内所有版本组合都已完整测试。具体声明以 `pyproject.toml` 为准。

## 安装步骤

### 创建 Python 环境

下面使用 Conda；已有可用环境时直接激活即可：

```shell
conda create -n nano-vllm python=3.10 -y
conda activate nano-vllm
python -m pip install --upgrade pip
```

后面的项目命令均从本仓库根目录执行。

### 安装并检查 PyTorch

先按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 安装 CUDA 版 PyTorch。项目不指定统一的 `cuXXX` 下载地址，已安装的兼容版本可以保留。

```shell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

最后一项应为 `True`。只有 CPU 版 PyTorch，或仅能导入 PyTorch，还不足以运行本项目。

### 安装构建工具和 FlashAttention

先将构建依赖放入当前环境，再安装 FA2：

```shell
python -m pip install packaging setuptools wheel ninja
python -m pip install "flash-attn>=2.8.3,<2.9" --no-build-isolation
```

`--no-build-isolation` 沿用 [FlashAttention 官方安装方式](https://github.com/Dao-AILab/flash-attention/tree/v2.8.3.post1#installation-and-features)，让构建使用当前环境的 PyTorch 和构建依赖。安装过程可能取得匹配的预编译 wheel；进入源码构建时，需要可用的 CUDA Toolkit 与 C++ 编译器。

从源码构建前可检查：

```shell
nvcc --version
c++ --version
ninja --version
```

Ubuntu / WSL Ubuntu 缺少系统编译器时，可安装 `build-essential`。源码构建占用较多 CPU 内存时，按官方说明限制并行任务，例如：

```shell
MAX_JOBS=4 python -m pip install "flash-attn>=2.8.3,<2.9" --no-build-isolation
```

项目的 `_nanovllm_C` 扩展只编译 host C++，无需 `nvcc`；FA2 的源码构建则包含 CUDA，两者的工具链要求不同。

### 可选的 spinloop 等待扩展

共享内存消息队列默认使用 Python/ZMQ 等待路径。需要使用原生 CPU 等待循环时，
在启动进程前设置 `VLLM_USE_SPINLOOP_EXT=1`。仅设置此变量不会开启多 Worker；
它只影响使用消息队列的执行路径。

```shell
export VLLM_USE_SPINLOOP_EXT=1
python -c "from nanovllm.spinloop import spinloop; print(spinloop(b'0', lambda: True, timeout=0.1))"
```

首次导入使用当前 Python 的头文件，通过 PyTorch/Ninja 编译
[`csrc/spinloop.cpp`](../nanovllm/csrc/spinloop.cpp)，后续进程复用构建缓存。
无需安装 vLLM 或使用其 Python 3.11 ABI 产物，也不需要 `nvcc`。
可用 `TORCH_EXTENSIONS_DIR` 指定缓存目录；将 `VLLM_USE_SPINLOOP_EXT` 设为 `0`
或取消该变量可关闭队列的扩展路径。队列加载扩展失败时会输出原因并回退到
Python/ZMQ；直接导入 `nanovllm.spinloop` 失败则抛出 `ImportError`。

本机 Intel CPU 使用 PAUSE 回退路径；支持条件满足的 AMD CPU 可使用
MONITORX/MWAITX。具体收益需要单独测量，不改变共享内存队列协议。

### 安装项目

```shell
python -m pip install -e .
```

等价的 requirements 入口为：

```shell
python -m pip install -r requirements.txt
```

`-e` 表示可编辑安装，修改 Python 源码后继续使用当前工作区。依赖仍统一从 `pyproject.toml` 解析。

### 按需启用 CuTe DSL 或 Benchmark

```shell
# CuTe SiLU × Mul 后端。
python -m pip install -e ".[cutedsl]"

# Parquet Benchmark 数据读取。
python -m pip install -e ".[benchmark]"

# 同时安装两个可选组件。
python -m pip install -e ".[cutedsl,benchmark]"
```

CuTe 的 pip 包名为 **`nvidia-cutlass-dsl`**，Python 使用 `import cutlass.cute as cute`。[官方 wheel 包含生成 GPU kernel 所需的 DSL 组件](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html#installation)，使用本项目的 CuTe 实现无需另外克隆 CUTLASS 仓库；CUDA Python bindings 等包由其依赖带入。GPU 驱动仍需满足所装 DSL 的要求。

## 验证安装

### 检查基础导入

```shell
python - <<'PY'
import torch
import transformers
import triton
import flash_attn_2_cuda
from nanovllm import LLM, SamplingParams
from nanovllm.config import KernelConfig, NanovllmConfig

assert torch.cuda.is_available()
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Triton:", triton.__version__)
PY
```

检查 CuTe extra：

```shell
python -c "import cutlass.cute; from cutlass.cute.testing import autotune_jit"
```

### 验证 Graph 所需的 host 扩展

该命令会实际编译并加载项目的 LibTorch host 扩展，首次执行需要时间：

```shell
python -c "from nanovllm._C import load_extension; load_extension(); print('host extension ready')"
```

编译产物由 PyTorch 的扩展缓存复用，可通过 `TORCH_EXTENSIONS_DIR` 指定缓存目录。修改 PyTorch 或工具链后，应确认扩展能在新环境重新构建。

### 运行算子与模型

基础安装先验证 PyTorch 和 Triton 后端：

```shell
python tests/test_ops/test_silu_and_mul.py --backends torch triton --dtypes float16
```

安装 CuTe extra 后比较三个后端：

```shell
python tests/test_ops/test_silu_and_mul.py --dtypes float16 bfloat16 float32
```

模型下载与生成示例见 [README](../README.md#下载模型与首次运行)，吞吐与延迟测试见 [Benchmark 文档](../benchmarks/README.md)。

## 当前开发环境快照

以下版本读取自 2026-10-20 的 `nano-vllm` 环境，用于排查兼容性，不作为新安装时的固定版本清单：

| 组件 | 版本 |
| --- | --- |
| Python | 3.10 |
| PyTorch | `2.5.1+cu121` |
| Triton | `3.1.0` |
| Transformers | `5.15.0` |
| FlashAttention | `2.8.3.post1` |
| CuTe DSL | `4.8.1` |

安装或更换依赖后，以当前机器的导入检查、算子数值测试与模型生成为准。
