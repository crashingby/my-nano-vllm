# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""统一配置类的 Pydantic dataclass 构造和字段校验策略。"""

from collections.abc import Callable
from dataclasses import field
from typing import Any, TypeVar, overload

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from pydantic.fields import Field as PydanticField
from typing_extensions import dataclass_transform

ConfigT = TypeVar("ConfigT")  # 保留装饰前后的配置类类型，供调用方和类型检查器使用。


@overload
@dataclass_transform(field_specifiers=(field, PydanticField))
def config(cls: type[ConfigT]) -> type[ConfigT]:
    """直接装饰配置类，采用默认校验策略。

    Returns:
        config_cls: 应用默认 Pydantic 策略后的配置 dataclass 类。
    """
    ...


@overload
@dataclass_transform(field_specifiers=(field, PydanticField))
def config(
    *,
    config: ConfigDict | None = None,
    **kwargs: Any,
) -> Callable[[type[ConfigT]], type[ConfigT]]:
    """携带 Pydantic 策略或 dataclass 参数创建配置装饰器。

    Returns:
        decorator: 接收配置类并应用指定校验策略和 dataclass 参数的装饰器。
    """
    ...


@dataclass_transform(field_specifiers=(field, PydanticField))
def config(
    cls: type[ConfigT] | None = None,
    *,
    config: ConfigDict | None = None,
    **kwargs: Any,
) -> type[ConfigT] | Callable[[type[ConfigT]], type[ConfigT]]:
    """将配置类转换为默认禁止未知构造字段的 Pydantic dataclass。

    对齐 vLLM 的同名装饰器，支持 ``@config`` 和 ``@config(...)`` 两种形式。
    字段类型与 ``Field`` 约束在 ``__post_init__`` 前完成校验；仍保留标准
    dataclass 的默认工厂、派生字段以及现有构造后的运行时状态回写。

    Args:
        cls: 直接使用装饰器时传入的配置类；带参数调用时为 None。
        config: 覆盖默认策略的 Pydantic 配置，例如允许 torch.dtype 等类型。
        **kwargs: 传给 Pydantic dataclass 的参数，例如 slots、kw_only 或 frozen。

    Returns:
        根据调用方式返回以下分支之一：
            config_cls: 直接传类时返回完成 dataclass 转换的配置类。
            decorator: 未传类时返回等待接收配置类的装饰器。

    Side Effects:
        为配置类建立构造期校验逻辑；默认不启用赋值校验，初始化后仍可回写
        设备、KV 容量和编译状态，不会在每次赋值时重复执行初始化过程。
    """
    # 每次调用独立合并策略，避免一个配置类的覆盖项影响其他配置类。
    merged_config = ConfigDict(extra="forbid")
    if config is not None:
        merged_config.update(config)

    def decorator(config_cls: type[ConfigT]) -> type[ConfigT]:
        """应用当前配置类的校验策略，并透传 dataclass 的结构参数。

        Returns:
            config_cls: 应用当前校验策略并完成 dataclass 转换的配置类。
        """
        return dataclass(config_cls, config=merged_config, **kwargs)

    # 带参数形式先保存策略，直接形式立即完成类的转换。
    if cls is None:
        return decorator
    return decorator(cls)
