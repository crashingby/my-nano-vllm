"""只读取本地 JSONL/Parquet 记录；任务字段和 prompt 模板由适配器解释。"""

from collections.abc import Iterator
import json
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """逐行读取 JSON 对象，出错时保留文件名和物理行号。

    空白行不产生记录；不下载数据、不修改文件、不提前加载整个数据集。

    Yields:
        record: 从本地 JSONL 解码的一条非空 JSON 对象，文件错误保留物理行号。
    """
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            yield record


def iter_parquet(path: Path, columns: list[str]) -> Iterator[dict[str, Any]]:
    """按批读取指定列，避免 Hub 探测和 datasets 的额外缓存。

    Args:
        path: 本地 Parquet 文件。
        columns: 适配器需要的列；不读取参考答案等无关列。

    Yields:
        record: 从本地 Parquet 批次逐条转换的行字典，只包含请求的列。

    Raises:
        ImportError: 未安装可选 benchmark 依赖 pyarrow。
        ValueError: 文件缺少指定列。
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError('Install benchmark dependencies: pip install -e ".[benchmark]"') from exc

    with pq.ParquetFile(path) as parquet_file:
        missing = set(columns) - set(parquet_file.schema_arrow.names)
        if missing:
            raise ValueError(f"Missing Parquet columns in {path}: {sorted(missing)}")
        # 每批转为普通字典，公共数据集层不依赖 Arrow 的容器类型。
        for batch in parquet_file.iter_batches(batch_size=256, columns=columns):
            yield from batch.to_pylist()
