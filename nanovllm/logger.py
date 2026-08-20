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
    logger.log(level, msg, *args, stacklevel=3)


class NanoVllmLogger(Logger):
    def info_once(self, msg: str, *args: Hashable) -> None:
        _log_once(logging.INFO, self, msg, *args)

    def warning_once(self, msg: str, *args: Hashable) -> None:
        _log_once(logging.WARNING, self, msg, *args)


_configure_root_logger()

# 使用日志的源文件都要通过它获得日志类
def init_logger(name: str) -> NanoVllmLogger:
    return cast(NanoVllmLogger, logging.getLogger(name))
