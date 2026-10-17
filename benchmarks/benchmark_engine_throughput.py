import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from datasets import load_dataset
from transformers  import AutoTokenizer
from contextlib import contextmanager
from collections.abc import Generator
import torch
from nanovllm import LLM,SamplingParams
from nanovllm.config import (
    CacheConfig,
    ModelConfig,
    NanovllmConfig,
    ParallelConfig,
    SchedulerConfig,
)
from nanovllm.logger import init_logger

logger = init_logger(__name__)

@contextmanager
def nvtx_range(name:str)->Generator[None, None, None]:
    """在 Nsight Systems 中标记一段代码区间。

    这个工具只负责给 profiler 增加可读标签，不改变 benchmark 的执行逻辑。
    Nsight Systems 需要启动时带上 `--trace=nvtx,cuda,...` 才能看到这些区间。
    """
    # 当前 benchmark 必然依赖 CUDA；这里仍保留判断，避免纯 CPU 导入时报错。
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()


@dataclass
class BenchmarkConfig:
    """保存本次 Alpaca 推理测试的配置。"""

    model_path:str # 本地 Hugging Face 模型目录。
    dataset_path:str # 本地 Alpaca parquet 文件路径
    num_prompts:int # 本次最多读取多少条 prompt。
    max_active:int # engine 内同时保持多少个未完成请求。
    max_tokens: int  # 每个请求最多生成多少 token。
    temperature:float # 采样温度。
    enforce_eager: bool  # 是否关闭 CUDA graph，便于调试。
    max_model_len: int  # 单请求上下文上限，同时决定 Runner 请求行的固定容量。
    max_num_batched_tokens: int  # 单轮调度与 profiling 使用的 token 预算。
    gpu_memory_utilization: float  # 模型、激活、Graph 和 KV 共用的设备总显存预算比例。

@dataclass
class BenchmarkStats:
    """记录 benchmark 过程中累计出来的统计信息。"""

    num_submitted:int =0 # 已经提交到 engine 的请求数
    num_finished:int =0 # 已完成的请求数
    num_input_tokens:int =0 #所有提示词token数之和
    num_output_tokens:int =0 # 所有 completion token 数之和
    num_scheduled_tokens: int = 0  # step() 返回的真实调度 token 总数，含混合 batch。
    num_engine_steps:int =0 # 调用了多少次llm.step()

def parse_args()->BenchmarkConfig:
    """解析命令行参数，并转换成 benchmark 配置对象。"""
    parser=argparse.ArgumentParser(description="Benchmark nano-vLLM on local Alpaca parquet.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/huangxy/models/Qwen3-0.6B",
        help="本地模型目录"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/home/huangxy/datasets/alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet",
        help="本地 Alpaca parquet 文件路径。",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=128,
        help="最多读取多少条 Alpaca 样本。",
    )
    parser.add_argument(
        "--max-active",
        type=int,
        default=16,
        help="同时放进 engine 的最大未完成请求数。",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="每条请求最多生成多少 token。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="采样温度。",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="开启后禁用 CUDA graph，调试更直观但可能更慢。",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=4096,
        help="每请求的 prompt 与 output 总长度上限。",
    )
    parser.add_argument(
        "--max-num-batched-tokens", type=int, default=256,
        help="单轮真实 token 预算；长 prompt 使用 chunked prefill。",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.9,
        help="当前实例占设备总显存的预算比例。",
    )
    args=parser.parse_args()
    # 先拒绝无有效工作量的命令，避免空循环或直到加载模型后才暴露参数错误。
    for name in ("num_prompts", "max_active", "max_tokens"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return BenchmarkConfig(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        num_prompts=args.num_prompts,
        max_active=args.max_active,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

'''
Alpaca的格式如下
{
"instruction": "将以下句子翻译成英文",
"input": "今天天气很好。",
"output": "The weather is nice today."
}
'''

def build_alpaca_prompt(example:dict[str,str])->str:
    """把 Alpaca 的 instruction/input 字段拼成一个用户问题。"""
    instruction=example["instruction"].strip() # Alpaca 的主指令。
    input_text = example["input"].strip() # Alpaca 的可选输入上下文。
    if input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction

def load_prompts(config:BenchmarkConfig,tokenizer:AutoTokenizer)->list[str]:
    """从本地 parquet 读取 Alpaca 数据，并转成 Qwen3 chat prompt。"""
    dataset_file=Path(config.dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset parquet not found: {dataset_file}")

    # 用 datasets 的 parquet reader 直接读取本地文件，不需要再访问 Hugging Face 网络。
    dataset = load_dataset(
        "parquet",
        data_files=str(dataset_file),
        split="train",
    )
    # 限制样本数量，避免第一次 benchmark 直接把 5 万条全塞进来。
    num_prompts = min(config.num_prompts,len(dataset))
    selected_dataset = dataset.select(range(num_prompts))

    prompts:list[str]=[]
    for example in selected_dataset:
        user_prompt = build_alpaca_prompt(example=example)

        # 关闭 Qwen3 thinking，让较短 output 预算也能用于普通回答负载。
        chat_prompt = tokenizer.apply_chat_template(
            [{"role":"user","content":user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(chat_prompt)
    return prompts

def submit_requests(
    llm:LLM,
    prompts:list[str],
    sampling_params:SamplingParams,
    stats:BenchmarkStats,
    max_active:int,
    next_prompt_idx:int,
    num_active:int,
)->tuple[int,int]:
    """尽量向 engine 补请求，直到达到 max_active 或 prompt 用完。"""

    # 当 engine 里未完成请求不足时，继续从 prompt 列表里补新请求。
    while next_prompt_idx < len(prompts) and num_active<max_active:
        prompt = prompts[next_prompt_idx]
        # 同一份 token IDs 同时用于请求输入和逻辑输入量统计，避免重复编码。
        prompts_token_ids = llm.tokenizer.encode(prompt)
        llm.add_request(prompt=prompts_token_ids,sampling_params=sampling_params)

        stats.num_submitted+=1
        stats.num_input_tokens+=len(prompts_token_ids)

        next_prompt_idx+=1
        num_active+=1
    return next_prompt_idx,num_active

def run_benchmark(config:BenchmarkConfig)->None:
    """执行基于 add_request/step 的持续请求 benchmark。"""
    logger.info("Loading tokenizer from %s",config.model_path)
    with nvtx_range("load_tokenizer"):
        tokenizer = AutoTokenizer.from_pretrained(config.model_path,use_fast=True)

    logger.info("Loading Alpaca prompts from %s",config.dataset_path)
    with nvtx_range("load_prompts"):
        prompts=load_prompts(config,tokenizer=tokenizer)
    if not prompts:
        raise ValueError("The dataset must contain at least one prompt")

    logger.info(
        "Initializing LLM: model=%s num_prompts=%d max_active=%d max_tokens=%d",
        config.model_path,
        len(prompts),
        config.max_active,
        config.max_tokens,
    )
    with nvtx_range("init_llm"):
        # BenchmarkConfig 保存脚本选项；Engine 只接收已经组装好的完整推理配置。
        vllm_config = NanovllmConfig(
            model_config=ModelConfig(
                model=config.model_path,
                enforce_eager=config.enforce_eager,
                max_model_len=config.max_model_len,
            ),
            parallel_config=ParallelConfig(tensor_parallel_size=1),
            # 脚本并发上限同时约束 Runner 的行容量和 FULL decode 捕获尺寸。
            scheduler_config=SchedulerConfig(
                max_num_seqs=config.max_active,
                max_num_batched_tokens=config.max_num_batched_tokens,
            ),
            cache_config=CacheConfig(
                gpu_memory_utilization=config.gpu_memory_utilization,
            ),
        )
        llm = LLM(vllm_config=vllm_config)
    try:
        sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        stats = BenchmarkStats()
        next_prompt_idx = 0  # 下一个待提交的输入下标。
        num_active = 0  # 已提交但尚未收到完成结果的请求数。
        last_reported_finished = 0  # 防止同一完成数在多个 step 重复输出进度。
        # 只测请求提交和执行；模型加载、profiling、预热及捕获已经完成。
        start_time = perf_counter()

        # 首批请求先入队，后续每完成一批便补充请求，维持持续负载。
        with nvtx_range("submit_initial_requests"):
            next_prompt_idx, num_active = submit_requests(
                llm=llm,
                prompts=prompts,
                sampling_params=sampling_params,
                stats=stats,
                max_active=config.max_active,
                next_prompt_idx=next_prompt_idx,
                num_active=num_active,
            )
        with nvtx_range("benchmark_loop"):
            while next_prompt_idx < len(prompts) or not llm.is_finished():
                step_outputs, num_tokens = llm.step()
                stats.num_engine_steps += 1
                # 连续批处理允许 prefill/decode 混合；step() 返回非负的真实总量。
                stats.num_scheduled_tokens += num_tokens

                # 完成结果包含实际生成 token，EOS 提前结束时也据实计数。
                for _, completion_token_ids in step_outputs:
                    stats.num_finished += 1
                    stats.num_output_tokens += len(completion_token_ids)
                    num_active -= 1

                # 输出处理已释放完成请求的 KV 资源，下一步可以接纳新请求。
                next_prompt_idx, num_active = submit_requests(
                    llm=llm,
                    prompts=prompts,
                    sampling_params=sampling_params,
                    stats=stats,
                    max_active=config.max_active,
                    next_prompt_idx=next_prompt_idx,
                    num_active=num_active,
                )
                if (
                    stats.num_finished
                    and stats.num_finished % 32 == 0
                    and stats.num_finished != last_reported_finished
                ):
                    logger.info(
                        "Progress: finished=%d/%d active=%d submitted=%d",
                        stats.num_finished, len(prompts), num_active, stats.num_submitted,
                    )
                    last_reported_finished = stats.num_finished
        elapsed = perf_counter() - start_time
        # 最后的结束通知仍须下发给 Worker；清理耗时不纳入请求吞吐统计。
        llm.step()
        # 逻辑输入量包含命中 prefix cache 的 token，不等同于实际计算量。
        total_tokens = stats.num_input_tokens + stats.num_output_tokens

        logger.info("Benchmark finished.")
        logger.info("num_prompts: %d", len(prompts))
        logger.info("num_submitted: %d", stats.num_submitted)
        logger.info("num_finished: %d", stats.num_finished)
        logger.info("num_engine_steps: %d", stats.num_engine_steps)
        logger.info("input_tokens: %d", stats.num_input_tokens)
        logger.info("output_tokens: %d", stats.num_output_tokens)
        logger.info("scheduled_tokens: %d", stats.num_scheduled_tokens)
        logger.info("elapsed: %.3f s", elapsed)
        logger.info("requests/s: %.3f", stats.num_finished / elapsed)
        logger.info("total_tokens/s: %.3f", total_tokens / elapsed)
        logger.info("output_tokens/s: %.3f", stats.num_output_tokens / elapsed)
    finally:
        # 直接使用 add_request/step 的脚本也必须显式释放执行资源和通信组。
        llm.close()

def main() -> None:
    """benchmark 脚本入口。"""

    config = parse_args()
    run_benchmark(config)


if __name__ == "__main__":
    main()
