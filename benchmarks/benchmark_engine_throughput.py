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
from nanovllm.config import ParallelConfig
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

@dataclass
class BenchmarkStats:
    """记录 benchmark 过程中累计出来的统计信息。"""

    num_submitted:int =0 # 已经提交到 engine 的请求数
    num_finished:int =0 # 已完成的请求数
    num_input_tokens:int =0 #所有提示词token数之和
    num_output_tokens:int =0 # 所有 completion token 数之和
    num_prefill_tokens:int =0 # engine step 返回的 prefill token数累计
    num_decode_tokens:int =0 # engine step 返回的 decode token数累计
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
    args=parser.parse_args()
    return BenchmarkConfig(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        num_prompts=args.num_prompts,
        max_active=args.max_active,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        enforce_eager=args.enforce_eager,
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

        # nano-vLLM 内部会自己 encode 字符串；这里先套 chat template
        chat_prompt = tokenizer.apply_chat_template(
            [{"role":"user","content":user_prompt}],
            tokenize=False,
            add_generation_prompt=True
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
        # 直接在这里先编码
        prompts_token_ids = llm.tokenizer.encode(prompt)
        llm.add_request(prompt=prompt,sampling_params=sampling_params)

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

    logger.info(
        "Initializing LLM: model=%s num_prompts=%d max_active=%d max_tokens=%d",
        config.model_path,
        len(prompts),
        config.max_active,
        config.max_tokens,
    )
    with nvtx_range("init_llm"):
        llm=LLM(
            config.model_path,
            enforce_eager=config.enforce_eager,
            parallel_config=ParallelConfig(tensor_parallel_size=1),
        )
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    stats = BenchmarkStats()
    next_prompt_idx=0
    num_active=0
    last_reported_finished = 0
    start_time = perf_counter()

    # 先塞入第一批请求；否则 engine 为空时直接 step 会触发 scheduler 的 assert。
    with nvtx_range("submit_initial_requests"):
        next_prompt_idx,num_active=submit_requests(
            llm=llm,
            prompts=prompts,
            sampling_params=sampling_params,
            stats=stats,
            max_active=config.max_active,
            next_prompt_idx=next_prompt_idx,
            num_active=num_active
        )
    with nvtx_range("benchmark_loop"):
        while next_prompt_idx<len(prompts) or not llm.is_finished():
            step_outputs,num_tokens=llm.step()
            stats.num_engine_steps+=1

            # 这个正负的规则是nano-vllm定的
            # num_tokens > 0 表示本轮是 prefill；num_tokens < 0 表示本轮是 decode。
            if num_tokens > 0:
                stats.num_prefill_tokens += num_tokens
            else:
                stats.num_decode_tokens += -num_tokens

            # step_outputs 只包含本轮完成的请求；token_ids 是 completion 部分。
            for _, completion_token_ids in step_outputs:
                stats.num_finished += 1
                stats.num_output_tokens += len(completion_token_ids)
                num_active -= 1

            # 有请求完成后，继续补新请求，维持持续压测。
            next_prompt_idx, num_active = submit_requests(
                llm=llm,
                prompts=prompts,
                sampling_params=sampling_params,
                stats=stats,
                max_active=config.max_active,
                next_prompt_idx=next_prompt_idx,
                num_active=num_active,
            )
            # 每完成一小段打印一次进度，避免长时间运行时不知道卡在哪里。
            if (
                stats.num_finished
                and stats.num_finished % 32 == 0
                and stats.num_finished != last_reported_finished
            ):
                logger.info(
                    "Progress: finished=%d/%d active=%d submitted=%d",
                    stats.num_finished,
                    len(prompts),
                    num_active,
                    stats.num_submitted,
                )
                last_reported_finished = stats.num_finished
    elapsed = perf_counter() - start_time
    total_tokens = stats.num_input_tokens + stats.num_output_tokens

    logger.info("Benchmark finished.")
    logger.info("num_prompts: %d", len(prompts))
    logger.info("num_submitted: %d", stats.num_submitted)
    logger.info("num_finished: %d", stats.num_finished)
    logger.info("num_engine_steps: %d", stats.num_engine_steps)
    logger.info("input_tokens: %d", stats.num_input_tokens)
    logger.info("output_tokens: %d", stats.num_output_tokens)
    logger.info("prefill_tokens_by_engine: %d", stats.num_prefill_tokens)
    logger.info("decode_tokens_by_engine: %d", stats.num_decode_tokens)
    logger.info("elapsed: %.3f s", elapsed)
    logger.info("requests/s: %.3f", stats.num_finished / elapsed)
    logger.info("total_tokens/s: %.3f", total_tokens / elapsed)
    logger.info("output_tokens/s: %.3f", stats.num_output_tokens / elapsed)

def main() -> None:
    """benchmark 脚本入口。"""

    config = parse_args()
    run_benchmark(config)


if __name__ == "__main__":
    main()
