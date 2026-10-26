"""统一数据集记录与可回放请求，集中处理 chat template 和输入长度。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from transformers import PreTrainedTokenizerBase

from benchmarks.config import DatasetConfig
from nanovllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class DatasetSample:
    """适配器输出的单轮用户文本，尚未套 chat template 或编码。

    sample_id 标识源文件中的记录，用于定位字段错误或超长输入；prompt 只含
    请求文本，不含数据集参考答案。记录按需生成，不要求保存整个原始数据集。
    """

    prompt: str  # 已按任务组织好的用户文本，由公共层统一添加对话格式。
    sample_id: str  # 例如 qasper:0，在同一适配器的数据流中保持唯一。


@dataclass(frozen=True)
class SampleRequest:
    """已经完成预处理、可直接交给 Engine 的 benchmark 请求。

    命名参考 vLLM 的 SampleRequest；当前仅支持 token IDs 形式的单轮文本输入。
    prompt 列表提交后不得修改，prompt_len 始终从同一列表计算，避免统计量失配。
    expected_output_len 是生成预算，不是数据集参考答案的长度。
    """

    prompt: list[int]  # 含 chat template 的最终 token IDs，回放时不再编码。
    expected_output_len: int  # 数据准备时已为该输出预算预留上下文容量。
    request_id: str  # 映射回源记录的稳定 ID，回放时作为 Engine request_id 使用。

    @property
    def prompt_len(self) -> int:
        """返回真实提交的逻辑输入长度，不包含 Graph padding。

        Returns:
            prompt_len: 最终提交的 prompt token ID 数量，不含 Graph padding。
        """
        return len(self.prompt)


class BenchmarkDataset(ABC):
    """定义数据集适配契约，共享编码、长度处理与样本数量限制。

    新数据集实现 load_data()，逐条输出 DatasetSample；文件格式、字段映射与
    任务文本属于子类。sample() 统一产生 SampleRequest，吞吐或未来延迟脚本
    因而无需识别 Alpaca、LongBench 或原始文件格式。
    """

    DEFAULT_DATASET_PATH: ClassVar[Path | None] = None

    def __init__(self, config: DatasetConfig) -> None:
        """保存数据集配置并解析本地路径，不加载模型或读取全部记录。"""
        self.config = config  # 长度策略和子集选择均来自这一配置引用。
        path = config.path if config.path is not None else self.DEFAULT_DATASET_PATH
        if path is None:
            raise ValueError(f"dataset_path is required for dataset {config.name!r}")
        self.dataset_path = Path(path).expanduser()
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
        self.system_prompt: str | None = config.system_prompt  # 配置已规范化；None 保持单 user 模板。

    @abstractmethod
    def load_data(self) -> Iterator[DatasetSample]:
        """读取本地记录并组织用户文本，不添加 chat template 或参考答案。

        Returns:
            samples: 逐条提供用户文本和 sample_id 的 DatasetSample 迭代器。
        """
        raise NotImplementedError

    def _encode_prompt(self, tokenizer: PreTrainedTokenizerBase, prompt: str) -> list[int]:
        """应用统一对话模板并编码，关闭 Qwen3 thinking。

        模板已经提供角色和特殊 token，因此 encode 不再重复添加特殊 token。
        该阶段在性能计时前执行，所有数据集使用相同的对话封装。

        Returns:
            prompt_ids: 应用统一聊天模板并编码后的 token ID 列表，包含模板特殊 token。
        """
        messages: list[dict[str, str]] = []  # system 位于可变 user 文本前，保留完整共享前缀。
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        chat_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return tokenizer.encode(chat_prompt, add_special_tokens=False)

    def _truncate_prompt(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prompt: str,
        max_prompt_tokens: int,
    ) -> list[int]:
        """保留用户文本首尾，再重新构建完整对话，保证最终 token 数不超限。

        这是吞吐负载的长度控制，会改变原始任务内容，不用于官方质量评分。
        先扣除空用户文本的模板开销，再从文本中部删减；重新编码后若边界分词
        增加 token，则继续缩减文本，直到满足包含模板在内的硬上限。

        Returns:
            prompt_ids: 保留用户文本首尾并重建模板后的 token ID 列表，最终长度不超过上限。
        """
        template_tokens = len(self._encode_prompt(tokenizer, ""))
        content_ids = tokenizer.encode(prompt, add_special_tokens=False)
        keep = min(len(content_ids), max_prompt_tokens - template_tokens)
        # 模板开销包含完整 system；只缩减 user，避免长度限制破坏共享前缀。
        # 每次失败都减少保留量，保证分词边界变化不会导致无限重试。
        while keep > 0:
            head = (keep + 1) // 2
            tail = keep // 2
            kept_ids = content_ids[:head] + (content_ids[-tail:] if tail else [])
            shortened_prompt = tokenizer.decode(kept_ids, skip_special_tokens=False)
            prompt_ids = self._encode_prompt(tokenizer, shortened_prompt)
            if len(prompt_ids) <= max_prompt_tokens:
                return prompt_ids
            keep -= max(1, len(prompt_ids) - max_prompt_tokens)
        raise ValueError(
            f"Input limit {max_prompt_tokens} cannot fit chat template and user text"
        )

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: int,
        max_model_len: int,
    ) -> list[SampleRequest]:
        """顺序读取并接受最多 num_requests 条合法请求，不自动重复数据。

        输入上限取 max_prompt_tokens 与 max_model_len - output_len 的较小值，
        为目标输出预留位置。error 策略遇到超限记录立即报错；skip 继续寻找后续
        样本；truncate 保留文本首尾。文件或字段错误始终报错，不按超长策略吞掉。

        Returns:
            requests: 按源顺序接受的合法 SampleRequest 列表，最多 num_requests 条，不重复补齐。

        Side Effects:
            读取本地数据、运行 tokenizer 并记录接受、跳过和截断数量；不启动 GPU。
            返回列表只保存最终输入 IDs，原始长文档不会另行保留在适配器中。
        """
        if num_requests < 1 or output_len < 1 or output_len >= max_model_len:
            raise ValueError("Request count/output length must be positive and fit the model")
        input_limit = max_model_len - output_len
        if self.config.max_prompt_tokens is not None:
            input_limit = min(input_limit, self.config.max_prompt_tokens)

        requests: list[SampleRequest] = []  # 只收集通过处理并准备回放的请求。
        num_skipped = 0  # 仅统计因长度策略跳过的记录，不包含字段错误。
        num_truncated = 0  # 统计实际缩短用户文本的记录。
        for sample in self.load_data():
            prompt_ids = self._encode_prompt(tokenizer, sample.prompt)
            if not prompt_ids:
                raise ValueError(f"Empty prompt after tokenization: {sample.sample_id}")
            if len(prompt_ids) > input_limit:
                if self.config.long_prompt_policy == "error":
                    raise ValueError(
                        f"Sample {sample.sample_id!r} has {len(prompt_ids)} prompt tokens; "
                        f"limit={input_limit}, output_len={output_len}, "
                        f"max_model_len={max_model_len}. Choose --long-prompt-policy "
                        "skip/truncate or increase --max-model-len."
                    )
                if self.config.long_prompt_policy == "skip":
                    num_skipped += 1
                    continue
                prompt_ids = self._truncate_prompt(tokenizer, sample.prompt, input_limit)
                num_truncated += 1
            requests.append(SampleRequest(
                prompt=prompt_ids,
                expected_output_len=output_len,
                request_id=f"{self.config.name}:{sample.sample_id}",
            ))
            if len(requests) >= num_requests:
                break

        logger.info(
            "Prepared %s: accepted=%d requested=%d skipped=%d truncated=%d input_limit=%d",
            self.config.name, len(requests), num_requests, num_skipped, num_truncated, input_limit,
        )
        if not requests:
            raise ValueError(f"No usable samples in {self.dataset_path}")
        if len(requests) < num_requests:
            logger.warning(
                "Requested %d samples but only %d are available; no oversampling is applied.",
                num_requests, len(requests),
            )
        return requests
