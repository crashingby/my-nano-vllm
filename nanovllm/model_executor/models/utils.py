# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""模型权重的名称映射与递归加载，沿用 vLLM 的流式分派协议。

WeightsMapper 负责改名、过滤条目和附加 shard_id；AutoWeightsLoader 按模块
前缀递归，把融合参数交给层的 load_weights，把普通参数交给 weight_loader。
本模块不读取文件、不创建 TP 组，实际切片与复制由各层的参数加载器执行。
当前省略 PP 缺失层和在线量化重载装饰器，保留通用映射、buffer 与共享权重处理。
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, TypeAlias

import torch
from torch import nn

from nanovllm.logger import init_logger
from nanovllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from nanovllm.model_executor.model_loader.weight_utils import default_weight_loader

if TYPE_CHECKING:
    from transformers.conversion_mapping import WeightRenaming

logger = init_logger(__name__)

ShardId: TypeAlias = str | int | tuple[int, ...]
"""单个融合投影标识；tuple 表示连续的部分融合区间。"""
ShardIds: TypeAlias = ShardId | list[ShardId]
"""一个目标投影，或同一源权重需要依次装填的多个目标投影。"""


@dataclass
class WeightsMapper:
    """按规则转换 checkpoint 名称，可过滤条目并附带融合投影标识。

    各规则依次执行；映射到 None 的条目不再加载。stacked 映射除改名外还
    记录 shard_id，例如 .q_proj -> (.qkv_proj, q)，供 QKV 层解释融合布局。
    实际 TP rank、目标偏移和参数复制不属于本类职责。
    """

    orig_to_new_renaming: list[WeightRenaming] = field(default_factory=list)
    # 保留上游接口，调用方可提供带 rename_source_key 的转换规则对象。
    orig_to_new_regex: Mapping[re.Pattern[str], str | None] = field(default_factory=dict)
    orig_to_new_substr: Mapping[str, str | None] = field(default_factory=dict)
    orig_to_new_stacked: Mapping[str, tuple[str, ShardIds]] = field(default_factory=dict)
    orig_to_new_prefix: Mapping[str, str | None] = field(default_factory=dict)
    orig_to_new_suffix: Mapping[str, str | None] = field(default_factory=dict)

    def __or__(self, other: WeightsMapper) -> WeightsMapper:
        """合并规则，同类映射中由右侧覆盖重名项，原映射器不变。

        Returns:
            mapper: 包含两组规则的新映射器，renaming 规则按左右顺序拼接。
        """
        return WeightsMapper(
            orig_to_new_renaming=[
                *self.orig_to_new_renaming,
                *other.orig_to_new_renaming,
            ],
            orig_to_new_regex={**self.orig_to_new_regex, **other.orig_to_new_regex},
            orig_to_new_substr={**self.orig_to_new_substr, **other.orig_to_new_substr},
            orig_to_new_stacked={**self.orig_to_new_stacked, **other.orig_to_new_stacked},
            orig_to_new_prefix={**self.orig_to_new_prefix, **other.orig_to_new_prefix},
            orig_to_new_suffix={**self.orig_to_new_suffix, **other.orig_to_new_suffix},
        )

    def map_name(self, key: str) -> str | None:
        """只转换名称，供不需要读取 tensor 或 shard_id 的调用方使用。

        Returns:
            mapped_name: 转换后的名称；条目被过滤时返回 None。
        """
        result = self._map_name_with_shard(key)
        return result[0] if result is not None else None

    def _map_name(self, key: str) -> str | None:
        """保留上游旧入口，返回 map_name 得到的 mapped_name。"""
        return self.map_name(key)

    def _map_name_with_shard(self, key: str) -> tuple[str, ShardIds | None] | None:
        """依次应用改名与过滤规则，同时保留融合投影标识。

        Returns:
            (mapped_name, shard_id): 转换后的名称与可选投影标识；整体为 None
                表示丢弃条目。标识为 list 时，同一源权重需装填多个投影。
        """
        if key.endswith(".kv_scale"):
            logger.warning_once(
                "DEPRECATED. Found kv_scale in the checkpoint. "
                "This format is deprecated in favor of separate k_scale and "
                "v_scale tensors and will be removed in a future release. "
                "Functionally, we will remap kv_scale to k_scale and duplicate "
                "k_scale to v_scale"
            )

        for renaming in self.orig_to_new_renaming:
            key, _ = renaming.rename_source_key(key)

        for pattern, new_key in self.orig_to_new_regex.items():
            if pattern.search(key):
                if new_key is None:
                    return None
                key = pattern.sub(new_key, key)

        for substr, new_key in self.orig_to_new_substr.items():
            if substr in key:
                if new_key is None:
                    return None
                key = key.replace(substr, new_key, 1)

        shard_id: ShardIds | None = None
        for substr, (new_key, new_shard_id) in self.orig_to_new_stacked.items():
            if substr in key:
                key = key.replace(substr, new_key, 1)
                shard_id = new_shard_id

        for prefix, new_key in self.orig_to_new_prefix.items():
            if key.startswith(prefix):
                if new_key is None:
                    return None
                key = key.replace(prefix, new_key, 1)

        for suffix, new_key in self.orig_to_new_suffix.items():
            if key.endswith(suffix):
                if new_key is None:
                    return None
                key = new_key.join(key.rsplit(suffix, 1))

        return key, shard_id

    def apply(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """惰性转换权重流，保留 tensor 存储，并在需要时附加投影标识。

        Yields:
            (out_name, shard_data): 转换后的名称和源 tensor；被过滤的条目不产出。

        Side Effects:
            有融合映射时写入 tensor.shard_id。多个目标使用不同 tensor 对象，
            后续对象由 detach 创建并共享底层存储，不复制整份 checkpoint。
        """
        for name, data in weights:
            result = self._map_name_with_shard(name)
            if result is None:
                continue
            out_name, shard_id = result
            if shard_id is None:
                yield out_name, data
                continue
            shard_ids = shard_id if isinstance(shard_id, list) else [shard_id]
            for i, one_shard_id in enumerate(shard_ids):
                # 每个目标要保留独立标识，避免修改同一对象覆盖先前的 shard_id。
                shard_data = data if i == 0 else data.detach()
                shard_data.shard_id = one_shard_id
                yield out_name, shard_data

    def apply_list(self, values: list[str]) -> list[str]:
        """转换名称列表并删除过滤项，返回 mapped_names。"""
        return [
            out_name
            for name in values
            if (out_name := self._map_name(name)) is not None
        ]

    def apply_dict(self, values: dict[str, Any]) -> dict[str, Any]:
        """只转换字典的键并删除过滤项，返回 mapped_values，保留值对象。"""
        return {
            out_name: value
            for name, value in values.items()
            if (out_name := self._map_name(name)) is not None
        }

    def get_rename_mapper(self) -> WeightsMapper:
        """生成只含改名规则的副本，供模块名称匹配与量化配置使用。

        Returns:
            mapper: 去掉 stacked 和 None 过滤规则的映射器，保留各原始投影名称。
        """
        remove_none = lambda d: {k: v for k, v in d.items() if v is not None}
        return replace(
            self,
            orig_to_new_regex=remove_none(self.orig_to_new_regex),
            orig_to_new_substr=remove_none(self.orig_to_new_substr),
            orig_to_new_stacked={},
            orig_to_new_prefix=remove_none(self.orig_to_new_prefix),
            orig_to_new_suffix=remove_none(self.orig_to_new_suffix),
        )


def _get_tied_embedding_params(module: nn.Module) -> dict[str, str]:
    """找出共享词表参数的别名，以模块遍历遇到的首个名称作为加载目标。

    Returns:
        aliased: 完整别名到首个完整名称的映射，例如 lm_head.weight 映射到
            model.embed_tokens.weight；按 Parameter 身份判断共享，而非比较数值。
    """
    canonical: dict[int, str] = {}
    aliased: dict[str, str] = {}
    for prefix, submodule in module.named_modules(remove_duplicate=False):
        if not isinstance(submodule, VocabParallelEmbedding):
            continue
        for name, param in submodule.named_parameters(remove_duplicate=False):
            qualname = f"{prefix}.{name}" if prefix else name
            if (first_name := canonical.setdefault(id(param), qualname)) != qualname:
                aliased[qualname] = first_name
    return aliased


class AutoWeightsLoader:
    """沿模型树递归消费权重流，优先使用子模块声明的 load_weights 接口。

    子模块未声明加载入口时继续递归，叶子参数使用自身的 weight_loader，
    没有专用回调则完整复制。当前实例所属根模块不调用自身 load_weights，
    避免模型在该接口内创建自动加载器后无限递归。

    Attributes:
        module: 本次加载的根模块，产出的参数名均相对于此根模块。
        aliased_params: 共享词表参数的别名到标准名称的映射，别名不重复装填。
        _skipped_aliases: 本次权重流中确实出现并被跳过的共享别名。
        _loaded_params_are_complete: 子模块均报告已加载参数名时为 True，
            用于判断是否有足够信息检查共享参数的标准名称确实被加载。
    """

    REMOVE_UNUSED_ROTARY_EMBEDS_MAPPER = WeightsMapper(
        orig_to_new_substr={
            "rotary_pos_emb.inv_freq": None,
            "rotary_emb.inv_freq": None,
            "rotary_emb.cos_cached": None,
            "rotary_emb.sin_cached": None,
        }
    )

    def __init__(
        self,
        module: nn.Module,
        *,
        ignore_unexpected_prefixes: list[str] | None = None,
        ignore_unexpected_suffixes: list[str] | None = None,
    ) -> None:
        """记录根模块、可忽略条目和共享关系，不消费或装填权重。"""
        super().__init__()
        self.module = module
        self.ignore_unexpected_prefixes = ignore_unexpected_prefixes or []
        self.ignore_unexpected_suffixes = ignore_unexpected_suffixes or []
        self.aliased_params = _get_tied_embedding_params(module)
        self._skipped_aliases: dict[str, str] = {}
        self._loaded_params_are_complete = True

    def _groupby_prefix(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, Iterable[tuple[str, torch.Tensor]]]]:
        """按名称的首段惰性分组，消费当前组后才能继续读取下一组。

        Yields:
            (prefix, child_weights): 首段模块名与去掉首段后的权重流；参数本身
                对应空余名。不要求全局排序，同一前缀可在后面再次出现。
        """
        weights_by_parts = (
            (weight_name.split(".", 1), weight_data)
            for weight_name, weight_data in weights
        )
        for prefix, group in itertools.groupby(weights_by_parts, key=lambda x: x[0][0]):
            yield (
                prefix,
                (
                    ("" if len(parts) == 1 else parts[1], weights_data)
                    for parts, weights_data in group
                ),
            )

    def _get_qualname(self, prefix: str, rest: str) -> str:
        """连接非空路径片段，返回相对于加载根模块的 qualname。"""
        if prefix == "":
            return rest
        if rest == "":
            return prefix
        return ".".join((prefix, rest))

    def _can_skip(self, qualname: str) -> bool:
        """返回 is_alias，表示该名称是已识别的共享词表参数别名。"""
        return qualname in self.aliased_params

    def _can_ignore_unexpected(self, qualname: str) -> bool:
        """返回 can_ignore，只对不存在的条目应用显式前缀/后缀忽略规则。"""
        iup = (qualname.startswith(p) for p in self.ignore_unexpected_prefixes)
        ius = (qualname.endswith(s) for s in self.ignore_unexpected_suffixes)
        return any(iup) or any(ius)

    def _load_param(
        self,
        base_prefix: str,
        param: nn.Parameter,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[str]:
        """校验叶子路径，调用参数加载器，逐条产出已加载的完整参数名。

        Yields:
            weight_qualname: 实际装填的参数路径；路径错误或不匹配时抛出异常。

        Side Effects:
            加载器原地修改参数；源和目标 dtype 不同时按上游规则记录日志。
        """
        for weight_name, weight_data in weights:
            weight_qualname = self._get_qualname(base_prefix, weight_name)
            if self._can_skip(weight_qualname):
                logger.debug("Skipping weight %s", weight_qualname)
                continue

            if weight_name != "":
                if self._can_ignore_unexpected(weight_qualname):
                    logger.debug("Ignoring weight %s", weight_qualname)
                    continue
                raise ValueError(
                    f"Attempted to load nested weight {weight_qualname!r} "
                    f"into a single parameter {base_prefix!r}"
                )

            if param.dtype != weight_data.dtype:
                exact_widening = weight_data.dtype in (
                    torch.float16,
                    torch.bfloat16,
                    torch.float32,
                ) and param.dtype in (torch.float32, torch.float64)
                (logger.debug if exact_widening else logger.warning)(
                    "Attempted to load weight %s with dtype %s into "
                    "parameter with dtype %s",
                    weight_qualname,
                    weight_data.dtype,
                    param.dtype,
                )
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight_data)
            logger.debug("Loaded weight %s with shape %s", weight_qualname, param.shape)
            yield weight_qualname

    def _add_loadable_non_param_tensors(
        self,
        module: nn.Module,
        child_params: dict[str, torch.Tensor],
    ) -> None:
        """把持久 buffer 与 BatchNorm 统计加入本层的可加载目标表。

        Side Effects:
            修改 child_params；非持久 buffer 与 PyTorch state_dict 一样排除，
            例如本项目运行时生成的 RoPE 缓存和空 KV cache 不作为训练参数加载。
        """
        non_persistent: set[str] = getattr(module, "_non_persistent_buffers_set", set())
        for buf_name, buf in module.named_buffers(recurse=False):
            if buf_name not in child_params and buf_name not in non_persistent:
                child_params[buf_name] = buf

        if isinstance(
            module,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.LazyBatchNorm1d,
                nn.LazyBatchNorm2d,
                nn.LazyBatchNorm3d,
                nn.SyncBatchNorm,
            ),
        ):
            module_state_dict = module.state_dict()
            for stat_name in ("running_mean", "running_var", "num_batches_tracked"):
                child_params[stat_name] = module_state_dict[stat_name]

    def _load_module(
        self,
        base_prefix: str,
        module: nn.Module,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[str]:
        """先尝试子模块专用加载入口，再按首段路径递归定位普通模块与参数。

        Yields:
            qualname: 子模块报告或叶子加载器产出的完整目标名称。

        Side Effects:
            消费当前前缀的权重流并原地装填；未知条目默认报错，只有共享别名
            或显式忽略规则命中的不存在条目才跳过。
        """
        if module != self.module:
            module_load_weights = getattr(module, "load_weights", None)
            if callable(module_load_weights):
                loaded_params = module_load_weights(weights)
                if loaded_params is None:
                    logger.warning(
                        "Unable to collect loaded parameters for module %s", module
                    )
                    self._loaded_params_are_complete = False
                else:
                    yield from map(
                        lambda x: self._get_qualname(base_prefix, x), loaded_params,
                    )

        child_modules = dict(module.named_children())
        child_params = dict(module.named_parameters(recurse=False))
        self._add_loadable_non_param_tensors(module, child_params)

        for child_prefix, child_weights in self._groupby_prefix(weights):
            prefix = self._get_qualname(base_prefix, child_prefix)
            if child_prefix in child_modules:
                yield from self._load_module(
                    prefix, child_modules[child_prefix], child_weights,
                )
            elif child_prefix in child_params:
                if self._can_skip(prefix):
                    logger.debug("Skipping param %s", prefix)
                    continue
                yield from self._load_param(prefix, child_params[child_prefix], child_weights)
            else:
                if self._can_skip(prefix):
                    logger.debug("Skipping missing %s", prefix)
                    continue
                can_ignore_module = self._can_ignore_unexpected(prefix + ".")
                can_ignore_param = self._can_ignore_unexpected(prefix)
                if can_ignore_module or can_ignore_param:
                    logger.debug("Ignoring missing %s", prefix)
                    continue

                named_parameters = module.named_parameters(recurse=True)
                desc_param_keys = {
                    maybe_prefix(base_prefix, k) for k, _ in named_parameters
                }
                msg = (
                    f"There is no module or parameter named {prefix!r} "
                    f"in {self.module._get_name()}. "
                    f"The available parameters belonging to {base_prefix} "
                    f"({module._get_name()}) are: {desc_param_keys}"
                )
                raise ValueError(msg)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        mapper: WeightsMapper | None = None,
    ) -> set[str]:
        """合并映射规则、过滤共享别名并递归装填，返回完整的已加载名称集合。

        Args:
            weights: 相对于本次根模块的 checkpoint 名称与 tensor 流。
            mapper: 模型特有的名称与融合标识映射，None 表示不额外改名。

        Returns:
            autoloaded_weights: 本次实际加载的目标名称集合；可包含持久 buffer，
                融合参数按目标名称去重，供外层 DefaultModelLoader 检查覆盖情况。

        Side Effects:
            单次消费权重流并原地修改参数，不回滚已完成的加载。忽略匹配
            .bias 后缀的不存在条目，丢弃可重建的 RoPE 缓存；共享参数须
            确实加载其标准名称。
        """
        self.ignore_unexpected_suffixes.append(".bias")
        mapper = mapper or WeightsMapper()
        modules = (self.module, *self.module.children())
        iterator = (m.quant_config for m in modules if hasattr(m, "quant_config"))
        if quant_config := next(iterator, None):
            mapper |= quant_config.get_cache_scale_mapper()
            mapper |= quant_config.get_checkpoint_weight_mapper()
            ignore_unexpected_suffixes = quant_config._ignore_unexpected_suffixes
            self.ignore_unexpected_suffixes.extend(ignore_unexpected_suffixes)
        mapper |= self.REMOVE_UNUSED_ROTARY_EMBEDS_MAPPER

        weights = mapper.apply(weights)
        weights = self._filter_skipped(weights)
        autoloaded_weights = set(self._load_module("", self.module, weights))
        self._check_skipped_aliases(autoloaded_weights)
        return autoloaded_weights

    def _filter_skipped(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """记录并跳过实际出现的共享别名，产出其余 (name, weight)。"""
        for name, weight in weights:
            if (canonical := self.aliased_params.get(name)) is not None:
                self._skipped_aliases[name] = canonical
            if self._can_skip(name):
                continue
            yield name, weight

    def _check_skipped_aliases(self, autoloaded_weights: set[str]) -> None:
        """检查跳过别名后，其共享的标准名称确实已加载，防止留下未初始化权重。"""
        if not self._loaded_params_are_complete:
            return
        for alias, canonical in self._skipped_aliases.items():
            if canonical not in autoloaded_weights:
                raise ValueError(
                    f"{alias!r} was skipped because it is tied to {canonical!r} "
                    f"in {self.module._get_name()}, but {canonical!r} was not "
                    "found in the checkpoint, so the tied weight is "
                    "uninitialized."
                )


def maybe_prefix(prefix: str, name: str) -> str:
    """连接非空 prefix 与 name，返回完整模块名称 qualname。"""
    return name if not prefix else f"{prefix}.{name}"
