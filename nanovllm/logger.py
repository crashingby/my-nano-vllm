"""配置项目日志格式，并提供缓存去重和命名 logger 的类型入口。"""
import logging
import os
from functools import lru_cache
from logging import Logger
from typing import Hashable, cast


# 每次日志输出的统一格式
_FORMAT = (
    "%(levelname)s %(asctime)s "
    "[%(filename)s:%(lineno)d] %(message)s"
)
# 时间的格式
_DATE_FORMAT = "%m-%d %H:%M:%S"

# 配置根日志
def _configure_root_logger() -> None:
    """读取 NANOVLLM_LOGGING_LEVEL 并配置根 logger 与 nanovllm 日志级别。

    模块导入时执行。basicConfig 遵循标准库语义：根 logger 已有 handler 时不会
    强制替换 handler，但 nanovllm logger 的级别仍会更新。
    """
    level = os.getenv(key="NANOVLLM_LOGGING_LEVEL", default="INFO").upper()
    logging.basicConfig(
        level=level,
        format=_FORMAT,
        datefmt=_DATE_FORMAT,
    )
    logging.getLogger("nanovllm").setLevel(level=level)

# 重复一样的日志，只会输出一次
@lru_cache
def _log_once(level: int, logger: Logger, msg: str, *args: Hashable) -> None:
    """按日志级别、logger、模板和参数缓存调用，命中时不重复输出。

    Args:
        level: 标准库日志级别。
        logger: 实际负责输出的 logger。
        msg: 交给日志系统格式化的消息模板。
        *args: 必须可哈希的模板参数，也是缓存键的一部分。

    Side Effects:
        缓存未命中时输出日志。lru_cache 容量有限，条目淘汰后仍可能再次输出，
        因此该函数不保证进程整个生命周期中绝对只打印一次。
    """
    logger.log(level, msg, *args, stacklevel=3)


class NanoVllmLogger(Logger):
    """为 Logger 声明 info_once/warning_once 便捷接口，去重由共享缓存实现。"""
    def info_once(self, msg: str, *args: Hashable) -> None:
        """在缓存未命中时输出 INFO，模板参数需要可哈希。"""
        _log_once(logging.INFO, self, msg, *args)

    def warning_once(self, msg: str, *args: Hashable) -> None:
        """在缓存未命中时输出 WARNING，模板参数需要可哈希。"""
        _log_once(logging.WARNING, self, msg, *args)


_configure_root_logger()

# 使用日志的源文件都要通过它获得日志类
def init_logger(name: str) -> NanoVllmLogger:
    """取得命名 logger，并按 NanoVllmLogger 接口提供类型标注。

    Args:
        name: 通常使用调用模块的 __name__。

    Returns:
        logger: logging.getLogger 管理的同名实例。cast 只影响静态类型，本函数不会替换
            LoggerClass 或给已有 logger 动态添加方法。
    """
    return cast(NanoVllmLogger, logging.getLogger(name))
