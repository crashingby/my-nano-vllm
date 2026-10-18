"""读取官方 LongBench JSONL，组织长上下文任务的吞吐负载。"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from benchmarks.datasets.base import BenchmarkDataset, DatasetSample
from benchmarks.datasets.io import iter_jsonl


class LongBenchDataset(BenchmarkDataset):
    """支持单文件、单子集和目录读取，不执行官方准确率评分。

    非空 input 作为上下文后的任务指令；无 input 的摘要任务补充摘要指令。
    这里使用统一吞吐模板，不声明与官方逐任务评测模板完全相同。
    """

    DEFAULT_DATASET_PATH = Path.home() / "datasets/LongBench/data"

    def _get_files(self) -> list[Path]:
        """解析输入文件，目录模式默认只选原版子集，避免混入重叠的 LongBench-E。"""
        path = self.dataset_path
        if path.is_file():
            if path.suffix != ".jsonl":
                raise ValueError(f"LongBench requires JSONL: {path}")
            if self.config.subset is not None and path.stem != self.config.subset.removesuffix(".jsonl"):
                raise ValueError("--dataset-subset does not match the JSONL filename")
            return [path]
        # 同时接受下载根目录 LongBench/ 和实际 JSONL 所在的 LongBench/data/。
        if (path / "data").is_dir():
            path = path / "data"
        if self.config.subset is not None:
            subset = self.config.subset.removesuffix(".jsonl")
            if not subset or Path(subset).name != subset:
                raise ValueError(f"Expected a subset name, got {self.config.subset!r}")
            files = [path / f"{subset}.jsonl"]  # 显式指定 qasper_e 等名称时允许 E 版本。
        else:
            files = [file for file in sorted(path.glob("*.jsonl")) if not file.stem.endswith("_e")]
        if not files:
            raise FileNotFoundError(f"No LongBench JSONL files in {path}")
        for file in files:
            if not file.is_file():
                raise FileNotFoundError(f"LongBench subset not found: {file}")
        return files

    @staticmethod
    def _build_prompt(record: dict[str, Any], sample_id: str) -> str:
        """组合 context 和任务，不读取 answers，避免把参考答案泄漏到输入。"""
        context = record.get("context")
        task_input = record.get("input")
        if not isinstance(context, str) or not context.strip():
            raise ValueError(f"Expected non-empty context at {sample_id}")
        if not isinstance(task_input, str):
            raise ValueError(f"Expected string input at {sample_id}")
        if task_input.strip():
            return f"{context.strip()}\n\n{task_input.strip()}"
        # GovReport/MultiNews/VCSUM 等摘要子集的 input 为空，不能仅发送裸文档。
        if record.get("language") == "zh":
            return f"请总结下面文本的主要内容：\n\n{context.strip()}\n\n总结："
        return f"Summarize the main points of the following text:\n\n{context.strip()}\n\nSummary:"

    def load_data(self) -> Iterator[DatasetSample]:
        """按文件名及文件内顺序逐条读取，sample_id 保留子集名和记录下标。"""
        for path in self._get_files():
            for index, record in enumerate(iter_jsonl(path)):
                sample_id = f"{path.stem}:{index}"
                yield DatasetSample(
                    prompt=self._build_prompt(record, sample_id),
                    sample_id=sample_id,
                )
