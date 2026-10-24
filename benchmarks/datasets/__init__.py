"""数据集公共接口与显式注册表；新增适配器不需要修改吞吐回放代码。"""

from benchmarks.config import DatasetConfig
from benchmarks.datasets.alpaca import AlpacaDataset
from benchmarks.datasets.base import BenchmarkDataset, DatasetSample, SampleRequest
from benchmarks.datasets.longbench import LongBenchDataset

# 注册表只负责名称到实现的映射，原始格式和默认路径由各适配器定义。
DATASET_REGISTRY: dict[str, type[BenchmarkDataset]] = {
    "alpaca": AlpacaDataset,
    "longbench": LongBenchDataset,
}


def get_dataset(config: DatasetConfig) -> BenchmarkDataset:
    """选择并构造本地适配器，不读取记录、不创建 tokenizer 或 Engine。

    Returns:
        dataset: 持有指定配置的本地数据集适配器；尚未读取样本或创建 Engine。

    Raises:
        ValueError: 名称未注册。
        FileNotFoundError: 适配器的数据路径不存在。
    """
    dataset_class = DATASET_REGISTRY.get(config.name)
    if dataset_class is None:
        raise ValueError(f"Unknown dataset {config.name!r}; available: {sorted(DATASET_REGISTRY)}")
    return dataset_class(config)


__all__ = [
    "AlpacaDataset", "BenchmarkDataset", "DATASET_REGISTRY", "DatasetSample",
    "LongBenchDataset", "SampleRequest", "get_dataset",
]
