"""将本地 Alpaca Parquet 的 instruction/input 转换为统一用户文本。"""

from collections.abc import Iterator
from pathlib import Path

from benchmarks.datasets.base import BenchmarkDataset, DatasetSample
from benchmarks.datasets.io import iter_parquet


class AlpacaDataset(BenchmarkDataset):
    """读取单个本地 Parquet；output 参考答案不进入推理输入或生成预算。"""

    DEFAULT_DATASET_PATH = (
        Path.home() / "datasets/alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet"
    )

    def load_data(self) -> Iterator[DatasetSample]:
        """按源文件顺序组织 instruction 和可选 input，保留可定位的记录 ID。

        Yields:
            sample: 由 instruction 和可选 input 构造的 DatasetSample，保留源记录 ID。
        """
        if self.config.subset is not None:
            raise ValueError("AlpacaDataset does not support --dataset-subset")
        if not self.dataset_path.is_file() or self.dataset_path.suffix != ".parquet":
            raise ValueError(f"Alpaca requires a Parquet file: {self.dataset_path}")
        for index, record in enumerate(iter_parquet(self.dataset_path, ["instruction", "input"])):
            sample_id = f"{self.dataset_path.stem}:{index}"
            instruction = record["instruction"]
            input_text = record["input"]
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"Expected non-empty instruction at {sample_id}")
            # Alpaca 的无上下文记录通常使用空字符串；也允许本地转换文件中的 null。
            if input_text is None:
                input_text = ""
            if not isinstance(input_text, str):
                raise ValueError(f"Expected string input at {sample_id}")
            prompt = instruction.strip()
            if input_text.strip():
                prompt += f"\n\n{input_text.strip()}"
            yield DatasetSample(prompt=prompt, sample_id=sample_id)
