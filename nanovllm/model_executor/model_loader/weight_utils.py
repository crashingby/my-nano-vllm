# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""本地 safetensors 文件筛选、惰性迭代与普通参数装填工具。"""

from collections.abc import Generator
import json
import os
import re

import torch
from safetensors.torch import safe_open
from tqdm.auto import tqdm

# 与 vLLM 相同的逐行文本进度格式，避免多进程日志前缀破坏终端动画。
_BAR_FORMAT = (
    "{desc}: {percentage:3.0f}% Completed | "
    "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\n"
)


def _natural_sort_key(filepath: str) -> list[str | int]:
    """生成按数字片段排序的文件名键，使 shard-2 排在 shard-10 之前。

    Args:
        filepath: checkpoint 文件路径。

    Returns:
        交替包含文本和整数的排序键，忽略父目录。
    """
    # s 是文件名按数字边界拆出的片段；数字转整数以避免纯字符串排序。
    return [
        int(s) if s.isdigit() else s
        for s in re.split(r"(\d+)", os.path.basename(filepath))
    ]


def filter_duplicate_safetensors_files(
    hf_weights_files: list[str], hf_folder: str, index_file: str
) -> list[str]:
    """根据本地分片索引排除未被引用的 safetensors 文件。

    Args:
        hf_weights_files: 文件模式匹配得到的候选路径列表。
        hf_folder: checkpoint 所在目录。
        index_file: 索引文件名，通常是 model.safetensors.index.json。

    Returns:
        索引引用的候选文件；不存在索引时原样返回输入列表。

    Raises:
        FileNotFoundError: 索引存在，但引用的某个权重分片不在本地。

    Side Effects:
        仅读取 JSON 和查询文件存在性，不下载、不删除、不读取权重 tensor。
    """
    # index_file_name 是本地完整索引路径，不是 Hub 资源名称。
    index_file_name = os.path.join(hf_folder, index_file)
    if not os.path.isfile(index_file_name):
        return hf_weights_files
    with open(index_file_name, encoding="utf-8") as f:
        # weight_map 把原始参数名称映射到保存它的 safetensors 文件名。
        weight_map = json.load(f)["weight_map"]
    weight_files_in_index: set[str] = set()
    for weight_name in weight_map:
        weight_files_in_index.add(os.path.join(hf_folder, weight_map[weight_name]))
    # 本地路径不会下载补齐分片，缺失时提前报错，防止部分融合权重被当成完整参数。
    for filename in weight_files_in_index:
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Checkpoint shard listed in index is missing: {filename}")
    # 保留 vLLM 的按文件筛选逻辑，不在这里重新映射或合并权重名称。
    hf_weights_files = [f for f in hf_weights_files if f in weight_files_in_index]
    return hf_weights_files


def enable_tqdm(use_tqdm_on_load: bool) -> bool:
    """仅在用户启用且为全局 rank 0（或尚未初始化 distributed）时显示进度。

    Args:
        use_tqdm_on_load: LoadConfig 中的文件加载进度开关。

    Returns:
        当前进程是否应输出进度条；不会初始化 distributed。
    """
    return use_tqdm_on_load and (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )


def safetensors_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """按分片顺序惰性读取普通 safetensors，不预取整个模型。

    Args:
        hf_weights_files: 已经完成索引过滤的文件列表。
        use_tqdm_on_load: 是否显示文件级加载进度。

    Yields:
        (原始 checkpoint 权重名称, CPU tensor)。设备搬运及 TP 切片由参数加载器负责。

    Side Effects:
        每次打开一个文件并在遍历结束或生成器关闭时释放对应文件上下文。
        当前仅保留 vLLM 的普通 safe_open 路径，不含 eager、预取、量化或 EP 过滤。
    """
    loading_desc = "Loading safetensors checkpoint shards"
    # sorted_files 是新列表，不改变调用方保存的文件顺序。
    sorted_files = sorted(hf_weights_files, key=_natural_sort_key)
    for st_file in tqdm(
        sorted_files,
        desc=loading_desc,
        disable=not enable_tqdm(use_tqdm_on_load),
        bar_format=_BAR_FORMAT,
    ):
        # safe_open 默认在 CPU 上提供 tensor，不在文件读取工具中决定目标 GPU。
        with safe_open(st_file, framework="pt") as f:
            for name in f.keys():
                param = f.get_tensor(name)
                yield name, param


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """将普通 checkpoint tensor 拷贝到参数，兼容单元素标量形状差异。

    Args:
        param: 已分配的目标参数存储。
        loaded_weight: 从 checkpoint 读取的 tensor。

    Side Effects:
        原地覆盖 param.data；copy_ 按目标参数转换设备与 dtype，不重新绑定 Parameter。

    Raises:
        AssertionError: 非标量权重形状与目标参数不一致。
    """
    try:
        if param.numel() == 1 and loaded_weight.numel() == 1:
            # 标量可能以 [] 或 [1] 保存，只要都只有一个元素就按目标形状装填。
            param.data.copy_(loaded_weight.view(param.shape))
        else:
            # 普通权重不允许依靠广播掩盖错误的 checkpoint 或模型形状。
            assert param.size() == loaded_weight.size(), (
                f"Attempted to load weight ({loaded_weight.size()}) "
                f"into parameter ({param.size()})"
            )
            param.data.copy_(loaded_weight)
    except Exception:
        # 保留 vLLM 的调试断点位置，异常仍原样抛给调用方。
        raise
