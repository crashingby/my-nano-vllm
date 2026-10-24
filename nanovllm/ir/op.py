# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""定义算子语义、注册 provider，并按优先级或指定实现执行。

参考 vLLM 的 ir/op.py，保留 register_op、IrOp、IrOpImpl 的组织方式。
被 register_op 装饰的 PyTorch 函数自动成为 native 实现，Triton / CuTe DSL
通过 register_impl 登记具有相同签名和计算语义的 Python launcher。

本模块直接通过 Python 调用实现，schema 仅用于注册时检查接口一致性。
当前契约为 functional：实现应保持输入内容，返回结果遵守 native 的约定。
原地实现适配、torch.library 注册和模型级编译 lowering 由后续扩展处理。
注册和优先级设置在启动阶段完成；CUDA Graph 捕获后切换实现需要重新捕获。
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, ClassVar, overload

from torch.library import infer_schema

from nanovllm.logger import init_logger

logger = init_logger(__name__)

# 名称规则与 vLLM 一致，便于后续用于配置字段及 torch operator 名称。
_NAME_PATTERN = re.compile(r"[a-z_][a-z_0-9]*")
# native 由算子定义提供，unfused 保留给后续编译分解路径。
RESERVED_PROVIDERS = ("native", "unfused")


def _validate_name(name: str, entity_type: str) -> None:
    """校验算子或 provider 名称，非法值在修改注册表前报错。"""
    if not isinstance(name, str) or _NAME_PATTERN.fullmatch(name) is None:
        raise ValueError(
            f"Invalid {entity_type} name {name!r}; expected [a-z_][a-z_0-9]*"
        )


def _get_signature(fn: Callable[..., Any], description: str) -> inspect.Signature:
    """取得显式参数签名，拒绝当前算子协议尚未支持的可变或仅关键字参数。

    Returns:
        signature: 受当前 IR 协议支持的显式函数签名，不含可变或仅关键字参数。
    """
    signature = inspect.signature(fn)  # 仅在注册时反射，在线调用不进行参数绑定。
    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            raise ValueError(
                f"{description} has unsupported parameter {parameter.name!r}; "
                "IR ops require explicit positional parameters"
            )
    return signature


def _check_signature(
    signature: inspect.Signature,
    expected: inspect.Signature,
    description: str,
) -> None:
    """检查参数名、种类与默认值，返回类型和参数类型交给 schema 单独校验。"""
    actual_parameters = list(signature.parameters.values())
    expected_parameters = list(expected.parameters.values())
    if len(actual_parameters) != len(expected_parameters):
        raise ValueError(
            f"{description} must have {len(expected_parameters)} parameters; "
            f"got {len(actual_parameters)}"
        )
    # 相同的参数名和默认值保证 positional / keyword 两种调用具有相同含义。
    for actual, reference in zip(actual_parameters, expected_parameters):
        if (
            actual.name != reference.name
            or actual.kind != reference.kind
            or actual.default != reference.default
        ):
            raise ValueError(
                f"{description} parameter {actual!s} does not match "
                f"native parameter {reference!s}"
            )


@overload
def register_op(f: Callable[..., Any]) -> IrOp:
    """直接装饰 native 函数，并返回已注册的逻辑算子。

    Returns:
        op: 已注册的可调用 IrOp，保留 native 函数作为默认实现。
    """
    ...


@overload
def register_op(
    *,
    name: str | None = None,
) -> Callable[[Callable[..., Any]], IrOp]:
    """先配置算子名，再返回接收 native 函数的注册装饰器。

    Returns:
        decorator: 接收 native 函数并注册为 IrOp 的装饰器。
    """
    ...


def register_op(
    f: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
) -> IrOp | Callable[[Callable[..., Any]], IrOp]:
    """将一个带类型注解的 PyTorch 参考函数注册为逻辑算子。

    Args:
        f: native 函数；省略时返回装饰器，支持 @register_op(name="...")。
        name: 全局唯一算子名，省略时使用函数名。

    Returns:
        根据调用方式返回以下分支之一：
            op: 直接传入 native 函数时返回注册后的 IrOp。
            decorator: 未传 native 函数时返回等待接收函数的装饰器。

    Side Effects:
        成功创建后写入 IrOp.registry，并登记 native 实现。重复名称或无效签名
        在提交注册之前报错；导入与注册本身不编译 kernel 或初始化 CUDA。
    """
    def decorator(native_impl: Callable[..., Any]) -> IrOp:
        """创建完整算子后再写入注册表，失败时保持全局表原值。

        Returns:
            op: 完成校验并写入注册表的逻辑算子对象。
        """
        op_name = native_impl.__name__ if name is None else name
        _validate_name(op_name, "op")
        if op_name in IrOp.registry:
            raise ValueError(f"Op {op_name!r} is already registered")
        op = IrOp(op_name, native_impl)  # 注册函数同时定义公共签名与默认计算语义。
        IrOp.registry[op_name] = op
        return op

    if f is not None:
        return decorator(f)
    return decorator


class IrOp:
    """维护一个逻辑算子的 native 定义、各 provider 实现和默认选择顺序。

    impls 保存 provider 到 IrOpImpl 的映射；_priority_impls 保存启动时解析后的
    有效优先级。首次定义时仅包含 native，直接调用算子即执行参考函数。

    set_default / set_priority 修改此算子对象的进程内共享策略，适合当前同步
    单 Engine。模型层需要固定实现时，使用 get_impl() 绑定一个 IrOpImpl，
    该对象的调用不受后续默认优先级变化影响。
    """

    registry: ClassVar[dict[str, IrOp]] = {}  # 所有通过 register_op 登记的逻辑算子。

    def __init__(self, name: str, native_impl: Callable[..., Any]) -> None:
        """保存算子契约并创建 native 实现，注册全局名称由 register_op 负责。"""
        _validate_name(name, "op")
        self.name = name  # 配置和诊断日志使用的稳定算子名。
        self._py_signature = _get_signature(native_impl, f"Op {name!r}")
        # infer_schema 要求参数与返回类型完整，类型错误在模型启动前即可暴露。
        self._schema_str = infer_schema(native_impl, mutates_args=[])
        self.impls: dict[str, IrOpImpl] = {}  # 实现注册成功后才加入，避免半初始化条目。
        self._priority_impls: list[IrOpImpl] = []  # 空列表表示沿用 native 默认路径。
        self.impls["native"] = IrOpImpl(
            op=self,
            provider="native",
            impl_fn=native_impl,
            supported=True,
            supports_args=None,
        )

    def register_impl(
        self,
        provider: str,
        *,
        supported: bool = True,
        supports_args: Callable[..., bool] | None = None,
    ) -> Callable[[Callable[..., Any]], IrOpImpl]:
        """返回装饰器，为当前算子登记一个同签名、同语义的实现。

        Args:
            provider: 唯一实现名，例如 triton 或 cutedsl，保留名不可覆盖。
            supported: 硬件和依赖的静态可用性，在设备绑定后的启动阶段确定。
            supports_args: 可选输入条件函数，参数名和默认值与 native 一致；
                只检查 shape、dtype、stride 等元数据，避免在线读取设备数值。

        Returns:
            decorator: 将函数替换为 IrOpImpl 的装饰器；该对象可直接调用并持有原始 impl_fn。

        Side Effects:
            成功后更新 impls。已设置的优先级保持原值，新 provider 需要显式选择。
        """
        _validate_name(provider, "provider")
        if provider in RESERVED_PROVIDERS:
            raise ValueError(f"Provider {provider!r} is reserved for op {self.name!r}")

        def decorator(impl_fn: Callable[..., Any]) -> IrOpImpl:
            """检查实现契约并登记，已经选定的默认策略保持原值。

            Returns:
                impl: 完成校验并注册的 provider 实现对象，可直接调用。
            """
            impl = IrOpImpl(self, provider, impl_fn, supported, supports_args)
            self.impls[provider] = impl  # 构造期间完成重复、schema 与输入条件校验。
            if self._priority_impls:
                logger.warning(
                    "Registered provider %s for op %s after priority was set; "
                    "set priority again to select it.", provider, self.name,
                )
            return impl

        return decorator

    def get_impl(self, provider: str) -> IrOpImpl:
        """强制取得指定实现；不存在或静态不可用时报错，返回对象不会自动回退。

        模型构造或单算子 benchmark 可保存返回对象并直接调用。输入支持条件
        在该对象每次调用时检查，因为 packed token 数和布局可以逐轮变化。

        Returns:
            impl: 指定 provider 的实现对象，不存在或静态不可用时抛出异常。
        """
        if provider not in self.impls:
            raise ValueError(
                f"Unknown provider {provider!r} for op {self.name!r}; "
                f"registered providers: {list(self.impls)}"
            )
        impl = self.impls[provider]  # 返回已注册的 impl，不复制函数或建立另一份注册状态。
        if not impl.supported:
            raise ValueError(f"Provider {provider!r} is unavailable for op {self.name!r}")
        return impl

    def dispatch(self, *args: Any, **kwargs: Any) -> IrOpImpl:
        """按默认优先级选出首个支持当前输入的实现，保持本次调用的状态不变。

        优先级在启动时过滤静态不可用项；每轮只检查动态输入条件。实现执行时
        的异常交给调用方处理，输入不支持的回退发生在启动任何 kernel 之前。

        Returns:
            impl: 按当前优先级找到的首个支持本次输入的实现，必要时回退 native。
        """
        if not self._priority_impls:
            return self.impls["native"]  # 未配置优先级时提供确定的参考实现。
        for impl in self._priority_impls:
            if impl.supports_args(*args, **kwargs):
                return impl
        # 正常解析的列表末尾存在无条件实现，本分支保护内部策略不变量。
        raise RuntimeError(f"No implementation supports the arguments for op {self.name!r}")

    def _inner_call(self, *args: Any, **kwargs: Any) -> Any:
        """选择实现后调用原始函数，动态支持条件只在 dispatch 中计算一次。

        Returns:
            output: 选中实现的原始执行结果，不再添加包装。
        """
        impl = self.dispatch(*args, **kwargs)
        return impl.impl_fn(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """通过 Python 分派执行算子，返回选定 provider 的原始结果。

        Returns:
            output: 按当前 provider 策略执行算子得到的原始结果。
        """
        return self._inner_call(*args, **kwargs)

    def __repr__(self) -> str:
        """返回包含逻辑算子名称的简短调试表示。

        Returns:
            representation: 包含逻辑算子名称的简短调试字符串。
        """
        return f"IrOp({self.name!r})"

    def get_priority(self) -> list[str]:
        """返回解析后的默认 provider 顺序快照；尚未配置时返回空列表。

        Returns:
            providers: 当前默认 provider 优先级的字符串列表快照。
        """
        return [impl.provider for impl in self._priority_impls]

    def _filter_priority_impls(self, priority: list[str]) -> list[IrOpImpl]:
        """校验优先级并过滤不可用项，沿用 vLLM 自动追加 native 的回退规则。

        Returns:
            filtered_impls: 过滤静态不可用项并补齐 native 回退后的实现列表。
        """
        if isinstance(priority, str):
            raise ValueError(f"Priority for op {self.name!r} must be a list of provider names")
        # 先检查所有名称，保证前面的无条件实现不会掩盖后面的配置拼写错误。
        for provider in priority:
            if provider not in self.impls:
                raise ValueError(f"Unknown provider {provider!r} for op {self.name!r}")
        filtered_impls: list[IrOpImpl] = []  # 本地构造完毕后才交给调用方安装。
        for provider in priority:
            impl = self.impls[provider]
            if not impl.supported:
                continue  # 自动策略允许当前平台跳过不可用实现。
            filtered_impls.append(impl)
            if impl.supports_all_args:
                return filtered_impls  # 无条件实现之后的 provider 永远不会被选择。
        if priority:
            logger.warning(
                "Op %s priority %s requires native fallback; appended native.",
                self.name, priority,
            )
        filtered_impls.append(self.impls["native"])
        return filtered_impls

    def set_default(self, priority: list[str]) -> None:
        """安装进程内共享的默认策略；校验失败时保持旧策略。

        在 Worker 启动、预热和 Graph 捕获之前调用。注册表与图缓存的生命周期
        由调用方管理，本方法只更新 Python 策略，不重建已经捕获的 CUDA Graph。
        """
        self._priority_impls = self._filter_priority_impls(priority)
        logger.debug("Set priority for op %s to %s", self.name, self.get_priority())

    @contextmanager
    def set_priority(self, priority: list[str]) -> Generator[None, None, None]:
        """临时覆盖共享策略，正常或异常退出时恢复进入前的列表，支持嵌套。

        适合当前同步单算子实验；作用域修改共享对象，不提供线程间的策略隔离。
        已通过 get_impl() 绑定的实现继续使用原 provider。
        """
        previous_impls = self._priority_impls  # 保存原列表，嵌套作用域可逐层恢复。
        self._priority_impls = self._filter_priority_impls(priority)
        try:
            yield
        finally:
            self._priority_impls = previous_impls

    def supported_providers(self) -> list[str]:
        """按注册顺序列出静态可用 provider；具体输入支持范围由实现条件决定。

        Returns:
            providers: 按注册顺序排列的静态可用 provider 名称列表。
        """
        return [provider for provider, impl in self.impls.items() if impl.supported]


class IrOpImpl:
    """保存一个 provider 的 launcher、静态可用性和动态输入条件。

    impl_fn 执行计算，_supports_args 检查本次张量元数据，二者共享 native 参数
    名和默认值。直接调用本对象表示强制使用这个 provider，不执行其他实现。
    """

    def __init__(
        self,
        op: IrOp,
        provider: str,
        impl_fn: Callable[..., Any],
        supported: bool,
        supports_args: Callable[..., bool] | None,
    ) -> None:
        """校验实现契约并保存状态，实际加入 op.impls 由注册入口负责。"""
        _validate_name(provider, "provider")
        if provider in op.impls:
            raise ValueError(f"Provider {provider!r} is already registered for op {op.name!r}")
        if provider in RESERVED_PROVIDERS and provider != "native":
            raise ValueError(f"Provider {provider!r} is reserved for op {op.name!r}")
        if not isinstance(supported, bool):
            raise TypeError(f"supported for {op.name!r}/{provider} must be bool")
        description = f"Implementation {op.name!r}/{provider}"
        signature = _get_signature(impl_fn, description)
        _check_signature(signature, op._py_signature, description)
        schema = infer_schema(impl_fn, mutates_args=[])  # 校验 Tensor / 标量类型及返回结构。
        if schema != op._schema_str:
            raise ValueError(
                f"{description} schema {schema!r} does not match native {op._schema_str!r}"
            )
        if supports_args is not None:
            if not callable(supports_args):
                raise TypeError(f"supports_args for {op.name!r}/{provider} must be callable")
            # 输入条件允许省略类型注解，但其参数绑定规则必须与实际计算一致。
            support_signature = _get_signature(supports_args, f"supports_args for {description}")
            _check_signature(support_signature, op._py_signature, f"supports_args for {description}")
        self.op = op  # 指回逻辑算子，用于错误报告与接口语义关联。
        self.provider = provider  # 稳定实现名，可用于日志和 benchmark 标签。
        self.impl_fn = impl_fn  # 原始 Python launcher，不在注册时执行。
        self.supported = supported  # 静态硬件 / 依赖检查结果。
        self._supports_args = supports_args  # None 表示接受公共契约中的所有合法输入。

    @property
    def supports_all_args(self) -> bool:
        """返回实现是否无需额外输入筛选，供启动时截断无效的优先级后缀。

        Returns:
            supports_all_args: 实现没有额外输入筛选条件时为 True。
        """
        return self._supports_args is None

    def supports_args(self, *args: Any, **kwargs: Any) -> bool:
        """查询输入支持条件；条件函数的异常直接传播，不转化为静默回退。

        Returns:
            supports_args: 当前调用参数满足实现约束时为 True；筛选函数异常直接传播。
        """
        if self._supports_args is None:
            return True
        return self._supports_args(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """强制执行当前 provider，静态不可用或输入不支持时在启动 kernel 前报错。

        Returns:
            output: 当前 provider 的原始执行结果；不支持的输入在 kernel 启动前报错。
        """
        if not self.supported:
            raise ValueError(f"Provider {self.provider!r} is unavailable for op {self.op.name!r}")
        if not self.supports_args(*args, **kwargs):
            raise ValueError(
                f"Provider {self.provider!r} does not support arguments for op {self.op.name!r}"
            )
        return self.impl_fn(*args, **kwargs)
