# CUDA Graph 与 eager 性能对比

本次测试记录同一连续批处理实现中，开启 decode CUDA Graph 相对 eager 的变化。
**两组均关闭异步调度**，只改变 `--enforce-eager`，分别运行吞吐和延迟 benchmark。
测试日期为 **2026-10-24**，时区为 Asia/Shanghai。

## 环境与代码版本

| 项目 | 本次环境 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，标称 8 GB 显存，NVIDIA-SMI 报告 8188 MiB，24 SM，计算能力 8.9。 |
| CPU | Intel Core i7-14700HX。 |
| 系统 | WSL2，Linux 6.6.114.1-microsoft-standard-WSL2，x86_64。 |
| 驱动 | Windows / WSL KMD 610.74；NVIDIA-SMI 610.53。 |
| Python | Conda `nano-vllm`，Python 3.10.20。 |
| PyTorch | 2.5.1+cu121，PyTorch CUDA runtime 为 12.1。 |
| Attention | FlashAttention 2.8.3.post1，运行 FA2 后端。 |
| 其他依赖 | Triton 3.1.0、Transformers 5.15.0、PyArrow 25.0.1。 |
| 模型 | 本地 Qwen3-0.6B，BF16，路径为 `/home/huangxy/models/Qwen3-0.6B`。 |



## 固定负载与配置

| 项目 | 配置 |
| --- | --- |
| 数据集 | 本地 Alpaca Parquet，按文件顺序接受前 128 条有效记录。 |
| 输入构造 | 使用 `instruction`，有 `input` 时以两个换行拼接；不使用参考 `output`。 |
| 对话模板 | Qwen3 tokenizer 的 user 模板，`add_generation_prompt=True`、`enable_thinking=False`。 |
| 请求数 | `num_prompts=128`。 |
| 最大活跃请求数 | `max_active=16`，闭环补充请求，保持最多 16 条活跃。 |
| 输入长度策略 | 包含模板后最多 256 token，超长时保留首尾；本轮实际没有跳过或截断。 |
| 输出长度 | 每条固定 128 token，`ignore_eos=True`，无额外停止 token。 |
| 采样 | `temperature=0.6`、`seed=0`。 |
| 上下文上限 | `max_model_len=1024`。 |
| 单轮 token 预算 | `max_num_batched_tokens=1024`。 |
| 并行方式 | 单进程、单 GPU，TP=1。 |
| KV cache | `block_size=256`，`gpu_memory_utilization=0.5`，按启动 profiling 分配。 |
| SiLU × Mul | 默认 `torch` 后端。 |
| 调度 | 显式传入 `--no-async-scheduling`，`async_scheduling=False`，不创建在途批次队列。 |

128 条输入共 **3,737 token**，长度最小 18、最大 115，均值 29.195，中位数 25.5。
每次运行都完成全部 128 条请求，产生 **16,384 个输出 token**。

输入预先编码后才开始计时。`seed` 固定生成随机数初态，不保证 eager 与 Graph
输出文本逐字相同；本次通过固定输出长度控制工作量，不进行生成质量比较。

### 两组具体区别

| 执行方式 | `enforce_eager` | 模型 forward |
| --- | --- | --- |
| eager | `True`。 | 不使用 CUDA Graph。 |
| CUDA Graph | `False`。 | `FULL_DECODE_ONLY`：uniform 单 token decode 回放完整模型图，prefill 与混合批次仍普通执行。 |

这里的 eager 只表示关闭模型级 CUDA Graph。RMSNorm、RoPE、默认 SiLU × Mul
和采样器已有的局部编译仍然保留，因此不是完全禁用 `torch.compile` 的对照。

默认 Graph 候选尺寸为 `[1, 2, 4, 8, 16, 24, 32]`；Dispatcher 再按请求容量 16
过滤，实际捕获 `[1, 2, 4, 8, 16]` 五档，日志确认 `FULL=5 (largest=16)`。
捕获覆盖模型 hidden-state forward，词表投影、采样、结果回传和 CPU 调度仍由
图外执行链处理。机制说明见 [CUDA Graph 总览](../cudagraph/overview.md)。

## 测量方法与统计口径

两个 benchmark 各自比较 eager 和 Graph，每种组合测量 **3 次**，总共 12 次。
每次执行原有 CLI，创建独立 Engine，结束后关闭进程，不使用 profiler，也不插入
额外设备同步或正式请求预热来改变原有计时路径。保留所有测量，不剔除某一轮结果。

| 重复轮次 | 吞吐入口运行顺序 | 延迟入口运行顺序 |
| --- | --- | --- |
| 1 | eager → Graph。 | Graph → eager。 |
| 2 | Graph → eager。 | eager → Graph。 |
| 3 | eager → Graph。 | Graph → eager。 |

每轮先完成吞吐的两次运行，再执行延迟的两次运行，期间始终只有一个 benchmark
进程使用 GPU。交替顺序用于减轻先后次序和机器状态变化的影响。

计时包含请求入队、Scheduler、输入准备、模型计算、采样、CPU 结果回收与循环内
统计；**不包含**数据读取、模板处理、编码、模型加载、Engine 启动 profiling、
kernel 预热、Graph 捕获、最终 Worker 清理和 JSON 写入。

- 输出吞吐为 `output_tokens / elapsed`，单位是整个 Engine 的输出 token/s。
- TTFT 从请求调用 `add_request()` 前开始，到首次观察到真实输出；包含 Engine 内
  的等待，不包含样本尚未提交时在 benchmark 外等待补位的时间。
- ITL 是相邻两个真实 token 的可见时间差；TPOT 是每请求首 token 之后的平均
  生成时间；E2EL 从请求入队前计至完成。延迟统一使用毫秒。
- 延迟入口在 `step_fn()` 返回后采集时间，包含本地执行与回收开销，不表示单个
  CUDA kernel 的耗时。各指标的完整定义见 [benchmark 说明](../../benchmarks/README.md)。

Engine 的启动预热不保证覆盖正式请求的所有首次调用开销；首批请求产生的开销
仍保留在测量中。两种入口的统计工作不同，吞吐提升在各自入口内部计算。

## 吞吐结果

下表单位为 **输出 token/s**。平均值为三次独立运行的算术平均，标准差使用总体标准差；
所有原始结果均保留，两个入口分别比较，不把它们的测量混在一起。
这里的标准差描述三次结果的离散程度，不是置信区间。

| 入口 | 执行方式 | 第 1 次 | 第 2 次 | 第 3 次 | 平均值 ± 标准差 |
| --- | --- | ---: | ---: | ---: | ---: |
| 吞吐 | eager | 513.3 | 532.6 | 533.2 | 526.4 ± 9.3 |
| 吞吐 | CUDA Graph | 1593.1 | 1584.2 | 1598.4 | 1591.9 ± 5.8 |
| 延迟 | eager | 546.7 | 538.9 | 507.3 | 531.0 ± 17.0 |
| 延迟 | CUDA Graph | 1560.8 | 1605.8 | 1607.7 | 1591.4 ± 21.7 |

| 入口 | CUDA Graph / eager | 输出吞吐提升 |
| --- | ---: | ---: |
| 吞吐 | 3.02× | +202.42% |
| 延迟 | 3.00× | +199.72% |

提升按 Graph 三次 TPS 均值除以 eager 三次 TPS 均值、再减一计算。

12 次运行的工作量全部一致：每次提交并完成 128 条请求，输入 3,737 token、
输出 16,384 token，实际安排输入计算 19,993 token，计时循环执行 1,024 步。
实际计算量为 3,737 + 128 × 127 = 19,993，每条请求最后采样的 token 不再作为
下一轮输入执行，因此不能直接用输入与输出总数之和代替模型计算量。

## 延迟结果

下面只使用延迟入口的原始时间戳。每个模式合并三次的 **384 条请求**，
TTFT、TPOT、E2EL 各有 384 个样本；ITL 合计 **48,768 个 token 间隔**。
均值和 P99 都从合并后的样本重新计算，没有平均三次的 P99。

单位为 **ms**，均值变化按 Graph / eager - 1 计算，负值表示延迟下降。

| 指标 | eager 均值 | Graph 均值 | 均值变化 | eager P99 | Graph P99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TTFT | 37.699 | 43.872 | +16.37% | 55.159 | 151.334 |
| TPOT | 30.104 | 9.788 | -67.49% | 32.489 | 10.764 |
| ITL | 30.104 | 9.788 | -67.49% | 46.990 | 12.123 |
| E2EL | 3860.922 | 1286.920 | -66.67% | 4161.525 | 1518.412 |

各次测量的主要延迟如下，便于直接观察运行间波动：

| 执行方式 | 重复轮次 | TTFT 均值 | TPOT 均值 | ITL P99 | E2EL 均值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| eager | 1 | 38.397 | 29.194 | 36.355 | 3746.017 |
| eager | 2 | 37.195 | 29.628 | 36.334 | 3799.996 |
| eager | 3 | 37.506 | 31.490 | 53.216 | 4036.754 |
| CUDA Graph | 1 | 48.062 | 9.952 | 12.605 | 1311.914 |
| CUDA Graph | 2 | 41.543 | 9.714 | 11.679 | 1275.194 |
| CUDA Graph | 3 | 42.012 | 9.698 | 11.570 | 1273.651 |

eager 第三次的 ITL P99 高于前两次，按原协议保留，没有剔除尾部较慢的运行。

### 首批请求的 TTFT

本轮没有给正式请求再做额外预热，首批 16 条请求的 TTFT 与后续请求不同。
分别合并三次首批的 48 条请求和后续的 336 条请求，观察到：

| 执行方式 | 首批 TTFT 均值 | 后续请求 TTFT 均值 |
| --- | ---: | ---: |
| eager | 53.269 | 35.475 |
| CUDA Graph | 117.005 | 33.425 |

这些首批开销保留在主结果中，不因它们偏大就删除样本。当前 Graph 策略没有
捕获 prefill；原始 12 次 benchmark 没有使用 profiler，下面的后续诊断单独记录。


## 结果解释与适用范围

本轮 Graph 将吞吐入口的平均输出吞吐提高到 eager 的 **3.02 倍**；
延迟入口的平均 ITL 从 **30.104 ms 降至 9.788 ms**，
下降 **67.49%**。
但平均 TTFT 从 **37.699 ms 增至 43.872 ms**，本次没有取得首 token 延迟收益。

这组输入较短、输出固定为 128 token，主要反映小模型持续 decode 的端到端效果。
Graph 用一次回放提交图内工作，减少重复的 Python 调用和 kernel 发射开销；
本次没有拆分 CPU/GPU 时间，不能把端到端倍率当作单个 kernel 的计算加速比。
结果也不能直接推广到长 prompt、其他并发、其他显卡或包含启动时间的场景。

本记录比较的是同一连续批处理版本的两种执行方式，不据此推算连续批处理相对
早期实现的收益。后续优化应保留相同负载和计时口径，再分别记录代码与环境变化。

## 复现命令

在仓库根目录执行。下面直接运行原有两个脚本，顺序与本次测量一致；唯一的模式
差别是 eager 添加 `--enforce-eager`。复测报告使用新目录，避免覆盖本次原始记录。

```shell
conda activate nano-vllm
export VLLM_HOST_IP=127.0.0.1
result_dir="benchmark_results/cuda_graph_recheck"
mkdir -p "$result_dir"

common_args=(
  --model-path /home/huangxy/models/Qwen3-0.6B
  --dataset-name alpaca
  --dataset-path /home/huangxy/datasets/alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
  --num-prompts 128
  --max-active 16
  --max-tokens 128
  --temperature 0.6
  --ignore-eos
  --seed 0
  --no-async-scheduling
  --max-model-len 1024
  --max-num-batched-tokens 1024
  --gpu-memory-utilization 0.5
  --max-prompt-tokens 256
  --long-prompt-policy truncate
)

run_pair() {
  local repetition="$1" benchmark="$2"
  shift 2
  local mode
  for mode in "$@"; do
    local extra_args=()
    if [[ "$mode" == "eager" ]]; then
      extra_args+=(--enforce-eager)
    fi
    python "benchmarks/benchmark_engine_${benchmark}.py" \
      "${common_args[@]}" "${extra_args[@]}" \
      --result-path "$result_dir/${benchmark}_${mode}_r${repetition}.json"
  done
}

run_pair 1 throughput eager graph
run_pair 1 latency graph eager
run_pair 2 throughput graph eager
run_pair 2 latency eager graph
run_pair 3 throughput eager graph
run_pair 3 latency graph eager
```

