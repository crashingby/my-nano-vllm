"""提供带进程身份、终端颜色和重复消息去重的项目日志。

调用方继续使用 init_logger(__name__) 获取标准库管理的 logger 对象。
decorate_logs() 通过 set_vllm_process_name() 更新当前进程身份，由 formatter
直接读取，不安装全局 LogRecord factory，也不改变其他库的 Logger 类型。
"""

from collections.abc import Callable, Hashable
from copy import copy
import logging
import os
from functools import lru_cache
from logging import Logger
from types import MethodType
from typing import TextIO, cast


_DATE_FORMAT = "%m-%d %H:%M:%S"  # 沿用当前短时间格式，便于观察相邻推理步骤。
_vllm_process_info: tuple[str, int] | None = None  # 当前进程的显示名与设置时 PID，名称沿用上游。


def set_vllm_process_name(process_name: str, *, skip_if_set: bool = False) -> None:
    """设置当前进程后续日志使用的显示名，供 decorate_logs() 调用。

    Args:
        process_name: 日志中的进程身份，例如 Worker、Worker_TP0。
        skip_if_set: 当前 PID 已设置过身份时保留原名；fork 继承的父进程身份
            不算已设置，子进程仍能安装自己的名字。

    Side Effects:
        更新本进程的日志显示状态，不修改操作系统进程标题或其他进程的日志。
    """
    global _vllm_process_info
    pid = os.getpid()  # 同时记录 PID，避免 fork 后把父进程身份误用到子进程。
    if skip_if_set and _vllm_process_info is not None and _vllm_process_info[1] == pid:
        return
    _vllm_process_info = (process_name, pid)


class NanoVllmFormatter(logging.Formatter):
    """输出进程名、PID 和源码位置，为多行消息与异常栈的每行补上前缀。

    在格式化阶段读取本进程身份，未设置时使用 LogRecord.processName。
    颜色只作用于显示字段，不改消息正文；处理记录副本，避免影响其他 handler。

    Attributes:
        use_color: 是否给进程身份、日志级别、时间和源码位置添加 ANSI 颜色。
    """

    COLORS = {
        logging.DEBUG: "\033[37m",  # DEBUG 白色。
        logging.INFO: "\033[32m",  # INFO 绿色。
        logging.WARNING: "\033[33m",  # WARNING 黄色。
        logging.ERROR: "\033[31m",  # ERROR 红色。
        logging.CRITICAL: "\033[35m",  # CRITICAL 品红色。
    }
    GREY = "\033[90m"  # 时间和源码位置使用灰色，降低视觉干扰。
    CYAN = "\033[36m"  # 进程身份使用青色，便于区分多 Worker 日志。
    RESET = "\033[0m"  # 每个着色字段结束后恢复终端状态。

    def __init__(self, use_color: bool = False) -> None:
        """保留标准库的消息及异常格式化，仅增加统一显示前缀。

        Args:
            use_color: 是否添加 ANSI 颜色，由日志输出流和环境设置决定。
        """
        super().__init__(fmt="%(message)s", datefmt=_DATE_FORMAT)  # 异常栈仍由标准库生成。
        self.use_color: bool = use_color  # 在创建 handler 时确定，不给每条日志重复探测终端。

    def format(self, record: logging.LogRecord) -> str:
        """格式化一条日志，为消息及异常栈的每一行添加相同进程前缀。

        Args:
            record: 标准库生成的日志记录，含创建时的进程、级别和调用位置。

        Returns:
            formatted_message: 带完整前缀的日志文本，不包含 handler 追加的末尾换行。

        Side Effects:
            标准库在记录副本上生成 message 和异常文本，不修改其他 handler
            可能继续使用的原始 record，也不修改全局日志记录工厂。
        """
        record = copy(record)  # Formatter 会缓存 message / exc_text，副本隔离这些修改。
        process_info = _vllm_process_info  # 同一次格式化使用同一份进程名与 PID 快照。
        process_name = record.processName  # 尚未 decorate 或 fork 后默认采用标准库进程名。
        if process_info is not None and process_info[1] == record.process:
            process_name = process_info[0]  # 只使用与该日志记录 PID 匹配的显示身份。
        process_prefix = f"({process_name} pid={record.process})"
        level_name = record.levelname  # 保留原始级别名称，不修改共享记录的 levelname。
        timestamp = self.formatTime(record, self.datefmt)  # 使用记录创建时间，而非输出时刻。
        file_info = f"[{record.filename}:{record.lineno}]"  # 指向实际日志调用位置。
        if self.use_color:
            process_prefix = f"{self.CYAN}{process_prefix}{self.RESET}"
            if color := self.COLORS.get(record.levelno):
                level_name = f"{color}{level_name}{self.RESET}"
            timestamp = f"{self.GREY}{timestamp}{self.RESET}"
            file_info = f"{self.GREY}{file_info}{self.RESET}"
        prefix = f"{process_prefix} {level_name} {timestamp} {file_info} "
        message = super().format(record)  # 统一处理格式参数、exception 和 stack_info。
        return prefix + message.replace("\n", "\n" + prefix)  # 空消息也保留前缀，异常栈逐行标记来源。


def _use_color(stream: TextIO) -> bool:
    """根据显式开关和输出流是否为终端决定是否启用颜色。

    Args:
        stream: handler 实际写入的文本流，当前默认为 sys.stderr。

    Returns:
        use_color: NO_COLOR 或 NANOVLLM_LOGGING_COLOR=0 时为 False；显式设置
            NANOVLLM_LOGGING_COLOR=1 或 FORCE_COLOR 时为 True；否则按 isatty()
            判断，重定向到文件或管道时自动输出纯文本。关闭设置优先。
    """
    color = os.getenv("NANOVLLM_LOGGING_COLOR")  # 沿用项目环境变量前缀，仅增加颜色开关。
    if "NO_COLOR" in os.environ or color == "0":
        return False
    if color == "1" or os.getenv("FORCE_COLOR", "0") not in {"", "0"}:
        return True
    return stream.isatty()


def _configure_root_logger() -> None:
    """按项目日志级别安装默认终端输出，并配置 nanovllm logger 的级别。

    Side Effects:
        模块导入时执行，为尚未配置的根 logger 添加带项目 formatter 的 stderr
        handler。保留 basicConfig 的既有语义：根 logger 已有 handler 时不覆盖
        调用方配置，但仍更新 nanovllm logger 的级别。不安装 LogRecord factory。
    """
    level = os.getenv(key="NANOVLLM_LOGGING_LEVEL", default="INFO").upper()
    if not logging.getLogger().handlers:
        handler = logging.StreamHandler()  # 使用标准库默认 stderr，颜色判断与实际输出流一致。
        handler.setFormatter(NanoVllmFormatter(use_color=_use_color(handler.stream)))
        logging.basicConfig(level=level, handlers=[handler])  # 沿用根 logger 输出路径，避免重复打印。
    logging.getLogger("nanovllm").setLevel(level=level)  # 不依赖其他库是否先初始化了根日志。


@lru_cache
def _log_once(pid: int, level: int, logger: Logger, msg: str, *args: Hashable) -> None:
    """按进程、级别、logger、模板和参数缓存调用，命中时不重复输出。

    Args:
        pid: 调用进程的 PID，仅参与缓存键，避免 fork 子进程沿用父进程的去重结果。
        level: 标准库日志级别。
        logger: 实际负责输出的 logger。
        msg: 交给日志系统格式化的消息模板。
        *args: 必须可哈希的模板参数，也是缓存键的一部分。

    Side Effects:
        缓存未命中时输出日志；不同 Worker 各自去重，不做 TP rank 筛选。
        lru_cache 容量有限，条目淘汰后仍可能再次输出。
    """
    logger.log(level, msg, *args, stacklevel=3)  # 跳过便捷方法与本函数，定位实际调用行。


class NanoVllmLogger(Logger):
    """声明便捷方法及其类型，由 init_logger() 绑定到真实 Logger 实例。

    不调用 setLoggerClass()，因此已有 logger 和其他库的 Logger 类型保持原样。
    四种级别共用带 PID 的去重缓存，只抑制同一进程内重复的调用。
    """

    def debug_once(self, msg: str, *args: Hashable) -> None:
        """在缓存未命中时输出 DEBUG；msg 和 args 的语义与 logger.debug 相同。"""
        _log_once(os.getpid(), logging.DEBUG, self, msg, *args)

    def info_once(self, msg: str, *args: Hashable) -> None:
        """在缓存未命中时输出 INFO；msg 和 args 的语义与 logger.info 相同。"""
        _log_once(os.getpid(), logging.INFO, self, msg, *args)

    def warning_once(self, msg: str, *args: Hashable) -> None:
        """在缓存未命中时输出 WARNING；msg 和 args 的语义与 logger.warning 相同。"""
        _log_once(os.getpid(), logging.WARNING, self, msg, *args)

    def error_once(self, msg: str, *args: Hashable) -> None:
        """在缓存未命中时输出 ERROR；msg 和 args 的语义与 logger.error 相同。"""
        _log_once(os.getpid(), logging.ERROR, self, msg, *args)


_METHODS_TO_PATCH: dict[str, Callable[..., None]] = {
    "debug_once": NanoVllmLogger.debug_once,
    "info_once": NanoVllmLogger.info_once,
    "warning_once": NanoVllmLogger.warning_once,
    "error_once": NanoVllmLogger.error_once,
}


_configure_root_logger()  # 配置模块默认输出；spawn 子进程重新导入时也建立自己的 handler。


def init_logger(name: str) -> NanoVllmLogger:
    """取得命名 logger，并像 vLLM 一样在实例上绑定四种级别的 once 方法。

    Args:
        name: 通常使用调用模块的 __name__。

    Returns:
        logger: 标准库管理的同名实例，保留原有对象身份、handler 和日志级别，
            并具备实际可调用的 debug_once / info_once / warning_once / error_once。

    Side Effects:
        绑定便捷方法，但不替换全局 LoggerClass 或 LogRecord factory；重复获取
        同一名称仍返回同一对象，之前已创建的 logger 也能补齐这些方法。
    """
    logger = logging.getLogger(name)  # 复用标准库的命名对象，调用方继续保存原来的 logger 引用。
    for method_name, method in _METHODS_TO_PATCH.items():
        setattr(logger, method_name, MethodType(method, logger))  # 给已有对象补方法，cast 只用于类型提示。
    return cast(NanoVllmLogger, logger)
