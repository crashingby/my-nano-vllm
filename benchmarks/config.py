"""集中定义 benchmark 配置，以及命令行到完整推理配置的转换。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from nanovllm.config import (
    CacheConfig,
    ModelConfig,
    NanovllmConfig,
    SchedulerConfig,
)
from nanovllm.sampling_params import SamplingParams


@dataclass
class DatasetConfig:
    """描述数据来源和公共长度策略，不持有 tokenizer 或 Engine。

    Attributes:
        name: 注册的数据集名称，当前为 alpaca 或 longbench。
        path: 本地文件或目录；None 使用对应适配器的默认位置。
        subset: 可选子集名，例如 multi_news；是否支持由具体适配器检查。
        system_prompt: 可选的共享 system 文本，去除首尾空白后放在 user 消息前。
            None 保持原有单 user 模板，模板构造和编码均在推理计时之外。
        max_prompt_tokens: 应用 chat template 后的输入 token 上限；None 表示
            只受模型上下文减去目标输出长度后的剩余容量限制。
        long_prompt_policy: 超长输入的处理方式。error 报错、skip 跳过，truncate
            则从用户文本中部截断并重新应用 chat template，不裁剪对话边界。
    """

    name: str = "alpaca"  # 用于注册表选择适配器，不通过文件扩展名猜测数据语义。
    path: str | None = None  # 路径默认值归各数据集适配器所有。
    subset: str | None = None  # 子集选择不进入吞吐回放逻辑。
    system_prompt: str | None = None  # 直接接收共享文本，不依赖仓库中的提示词文件。
    max_prompt_tokens: int | None = None  # 单位是当前 tokenizer 的 token，不是字符。
    long_prompt_policy: Literal["error", "skip", "truncate"] = "error"

    def __post_init__(self) -> None:
        """规范化共享文本并校验长度选项；数据文件与字段稍后由适配器检查。"""
        if self.system_prompt is not None:
            self.system_prompt = self.system_prompt.strip()  # 报告与实际编码使用同一文本。
            if not self.system_prompt:
                raise ValueError("system_prompt must not be empty")
        if self.max_prompt_tokens is not None and self.max_prompt_tokens < 1:
            raise ValueError("max_prompt_tokens must be positive")
        if self.long_prompt_policy not in ("error", "skip", "truncate"):
            raise ValueError(f"Unknown long_prompt_policy: {self.long_prompt_policy}")


@dataclass
class BenchmarkConfig:
    """聚合完整推理配置、数据集配置和回放选项。

    vllm_config 直接传给 LLM，不在 benchmark 中另存一份模型、显存或调度容量。
    最大并发由其中的 scheduler_config.max_num_seqs 确定。数据集预处理在计时前
    完成，num_prompts 是最多选取的有效样本数，不会自动重复不足的样本。
    """

    vllm_config: NanovllmConfig  # 调用方可直接提供全部已支持的推理子配置。
    dataset_config: DatasetConfig = field(default_factory=DatasetConfig)
    num_prompts: int = 128  # 限制接受的样本数，不包含长度过滤掉的记录。
    sampling_params: SamplingParams = field(
        default_factory=lambda: SamplingParams(temperature=0.6, max_tokens=128),
    )
    seed: int = 0  # 在 Engine 启动后重置生成随机数，避免启动预热消耗影响初态。
    result_path: str | None = None  # 可选 JSON 报告路径，写入发生在性能计时结束后。

    def __post_init__(self) -> None:
        """检查回放规模与输出预算，避免模型加载后才发现无效工作量。"""
        if self.num_prompts < 1:
            raise ValueError("num_prompts must be positive")
        # 每条请求至少保留一个输入 token；完整输入是否装得下由数据集层检查。
        if self.sampling_params.max_tokens >= self.vllm_config.model_config.max_model_len:
            raise ValueError("max_tokens must be less than max_model_len")


def parse_args(
    argv: Sequence[str] | None = None,
    *,
    description: str = "Benchmark local Engine throughput.",
) -> BenchmarkConfig:
    """解析共享 CLI，组装可直接传入 Engine 的总配置。

    Args:
        argv: 显式参数序列；None 读取当前进程命令行。
        description: 当前吞吐或延迟入口在 --help 中显示的测试说明。

    Returns:
        config: 已完成配置校验的 BenchmarkConfig；不读取数据集或创建 GPU 资源。
            ModelConfig 构造会读取本地 HF 结构配置。
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model-path", default=str(Path.home() / "models/Qwen3-0.6B"))
    parser.add_argument("--dataset-name", default="alpaca", help="注册的数据集名称。")
    parser.add_argument("--dataset-path", help="本地数据文件或目录，省略时使用适配器默认值。")
    parser.add_argument("--dataset-subset", help="子集名，例如 LongBench 的 multi_news。")
    parser.add_argument(
        "--system-prompt", help="所有请求共享的 system 文本，省略时仅有 user 消息。",
    )
    parser.add_argument("--num-prompts", type=int, default=128, help="最多接受的有效样本数。")
    parser.add_argument("--max-active", type=int, default=16, help="Scheduler/Runner 请求容量。")
    parser.add_argument("--max-tokens", type=int, default=128, help="每请求目标输出长度上限。")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--ignore-eos", action="store_true", help="忽略 EOS，测试固定输出长度。")
    parser.add_argument("--seed", type=int, default=0, help="生成随机种子，不保证跨模式逐字一致。")
    parser.add_argument("--result-path", help="可选 JSON 报告路径，包含配置摘要与实际指标。")
    parser.add_argument("--enforce-eager", action="store_true", help="禁用 CUDA Graph。")
    parser.add_argument(
        "--async-scheduling", action=argparse.BooleanOptionalAction, default=False,
        help="启用异步调度与在途批次队列；默认关闭，可用 --no-async-scheduling 显式关闭。",
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    # 提高单轮预算，让 512-token 输入减少分块和首批排队，仍允许 CLI 显式覆盖。
    parser.add_argument(
        "--max-num-batched-tokens", type=int, default=1024,
        help="单轮共享 token 预算，默认 1024；增大可减少 prefill 分块轮数。",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--block-size", type=int, default=256, help="Manager KV block 的 token 数，当前须为 256 的倍数。")
    parser.add_argument(
        "--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True,
        help="开启前缀缓存；--no-enable-prefix-caching 同时关闭请求 hash 生成和缓存复用。",
    )
    parser.add_argument(
        "--prefix-match-unit", type=int,
        help="前缀 hash 的 token 粒度；当前单 group 启用缓存时须等于 block-size。",
    )
    parser.add_argument(
        "--watermark", type=float, default=0.0,
        help="接纳等待/恢复请求时预留的总 KV block 比例，范围 [0, 1)，默认关闭。",
    )
    parser.add_argument("--max-prompt-tokens", type=int, help="含 chat template 的输入 token 上限。")
    parser.add_argument(
        "--long-prompt-policy", choices=("error", "skip", "truncate"), default="error",
        help="超过输入容量时报错、跳过或中部截断；默认不改变原始文本。",
    )
    args = parser.parse_args(argv)
    # 子配置继续负责各自的约束；CLI 只将校验错误转换成带参数说明的错误输出。
    try:
        return BenchmarkConfig(
            vllm_config=NanovllmConfig(
                model_config=ModelConfig(
                    model=str(Path(args.model_path).expanduser()),
                    enforce_eager=args.enforce_eager,
                    max_model_len=args.max_model_len,
                ),
                scheduler_config=SchedulerConfig(
                    max_num_seqs=args.max_active,
                    max_num_batched_tokens=args.max_num_batched_tokens,
                    async_scheduling=args.async_scheduling,
                    watermark=args.watermark,
                ),
                cache_config=CacheConfig(
                    block_size=args.block_size,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    enable_prefix_caching=args.enable_prefix_caching,
                    prefix_match_unit=args.prefix_match_unit,
                ),
            ),
            dataset_config=DatasetConfig(
                name=args.dataset_name,
                path=args.dataset_path,
                subset=args.dataset_subset,
                system_prompt=args.system_prompt,
                max_prompt_tokens=args.max_prompt_tokens,
                long_prompt_policy=args.long_prompt_policy,
            ),
            num_prompts=args.num_prompts,
            sampling_params=SamplingParams(
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                ignore_eos=args.ignore_eos,
            ),
            seed=args.seed,
            result_path=args.result_path,
        )
    except ValueError as exc:
        parser.error(str(exc))
