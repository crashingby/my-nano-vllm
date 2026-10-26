"""使用本地 Qwen3 执行单卡文本生成，默认先验证 eager 路径。"""

import argparse
import os

from nanovllm import LLM, SamplingParams
from nanovllm.config import (
    CacheConfig,
    ModelConfig,
    NanovllmConfig,
    SchedulerConfig,
)
from nanovllm.logger import init_logger
from transformers import AutoTokenizer

logger = init_logger(__name__)

def main() -> None:
    """读取模型目录和生成选项，输出结果后显式关闭 Engine。"""
    parser = argparse.ArgumentParser(description="Run nano-vLLM generation with local Qwen3.")
    parser.add_argument("--model", default="~/Models/Qwen3-0.6B", help="本地模型目录。")
    parser.add_argument("--max-tokens", type=int, default=64, help="每个请求最多生成的 token 数。")
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.9, help="当前实例的 GPU 显存预算比例。",
    )
    parser.add_argument("--cuda-graph", action="store_true", help="启用 FULL decode CUDA Graph。")
    parser.add_argument("--block-size", type=int, default=256, help="Manager KV block 的 token 数。")
    parser.add_argument(
        "--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True,
        help="开启前缀复用；--no-enable-prefix-caching 同时关闭 Request hash 生成。",
    )
    parser.add_argument("--prefix-match-unit", type=int, help="当前启用缓存时须等于 block-size。")
    parser.add_argument("--watermark", type=float, default=0.0, help="新请求接纳时的 KV 预留比例，范围 [0, 1)。")
    args = parser.parse_args()
    path = os.path.expanduser(args.model)
    # 当前示例只有同进程单卡 Worker，使用回环地址建立本地 process group 即可。
    os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
    logger.info("Loading tokenizer from %s", path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    # 各领域参数先由对应子配置解析，再聚合成贯穿 Engine 和 Worker 的总配置。
    vllm_config = NanovllmConfig(
        model_config=ModelConfig(
            model=path,
            enforce_eager=not args.cuda_graph,
            max_model_len=1024,
        ),
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=256,
            max_num_seqs=4,
            watermark=args.watermark,
        ),
        cache_config=CacheConfig(
            block_size=args.block_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=args.enable_prefix_caching,
            prefix_match_unit=args.prefix_match_unit,
        ),
    )
    logger.info("Initializing LLM")
    llm = LLM(vllm_config=vllm_config)
    try:
        sampling_params = SamplingParams(temperature=0.6, max_tokens=args.max_tokens)
        prompts = ["introduce yourself", "list all prime numbers within 100"]
        # Qwen3 使用 chat template；关闭 thinking 便于短输出预算下直接观察回答。
        formatted_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for prompt in prompts
        ]
        logger.info("Generating outputs")
        outputs = llm.generate(formatted_prompts, sampling_params)
        for prompt, output in zip(prompts, outputs):
            print(f"\nPrompt: {prompt!r}")
            print(f"Completion: {output['text']!r}")
    finally:
        # 生成失败也释放模型执行端和本进程创建的通信组。
        llm.close()


if __name__ == "__main__":
    main()
