import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from functools import cache

from nanovllm.logger import init_logger

DEVICE_CONTROL_ENV_VAR = "CPU_VISIBLE_MEMORY_NODES"

logger = init_logger(__name__)


def _read_int_file(path: str) -> int | None:
    try:
        with open(path) as f:
            value = f.read().strip()
        if not value or value == "max":
            return None
        return int(value)
    except (OSError, ValueError):
        return None

@cache
def get_cgroup_memory_limit() -> int | None:
    """Return the cgroup memory limit in bytes, or None.

    Supports both cgroup v2 (unified) and v1. Returns None when
    not running under a constrained cgroup (e.g. bare metal, or limit
    reported as `max`/an unrealistically large value).
    """
    if sys.platform != "linux":
        return None

    # cgroup v2 unified hierarchy
    v2_limit = _read_int_file("/sys/fs/cgroup/memory.max")
    if v2_limit is not None:
        return v2_limit

    # cgroup v1
    v1_limit = _read_int_file("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if v1_limit is not None:
        # cgroup v1 reports a huge sentinel (close to PAGE_COUNTER_MAX)
        # when unlimited. Treat absurdly large values as "no limit".
        if v1_limit >= (1 << 62):
            return None
        return v1_limit

    return None

def get_cgroup_memory_usage() -> int | None:
    """Return the current cgroup memory usage in bytes, or None.

    The usage value is intentionally read on every call because cgroup
    memory usage changes while the process is running.
    """
    if sys.platform != "linux":
        return None

    # cgroup v2 unified hierarchy
    if _read_int_file("/sys/fs/cgroup/memory.max") is not None:
        return _read_int_file("/sys/fs/cgroup/memory.current")

    # cgroup v1
    v1_limit = _read_int_file("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if v1_limit is not None and v1_limit < (1 << 62):
        return _read_int_file("/sys/fs/cgroup/memory/memory.usage_in_bytes")

    return None

def check_cgroup_memory_available(
    required_bytes: int,
    allocation_name: str,
) -> None:
    """Log cgroup memory headroom for an upcoming allocation.

    Args:
        required_bytes: Bytes required by the allocation.
        allocation_name: Human-readable name used in log messages.

    Low headroom logs a warning, but does not reject the allocation because
    cgroup usage can include reclaimable memory. If the cgroup limit or usage
    cannot be read, the check is skipped.

    """
    cgroup_limit = get_cgroup_memory_limit()
    cgroup_usage = get_cgroup_memory_usage()
    if cgroup_limit is None or cgroup_usage is None:
        return

    cgroup_available = max(0, cgroup_limit - cgroup_usage)
    mib = 1 << 20
    remaining_bytes = cgroup_available - required_bytes
    log_fn = logger.debug if remaining_bytes >= 0 else logger.warning
    status = (
        "current headroom meets the requested allocation"
        if remaining_bytes >= 0
        else "current headroom is below the requested allocation; allocation "
        "will still be attempted because cgroup usage may be reclaimable"
    )
    log_fn(
        "Cgroup memory preflight for %s: %.0f MiB required, %.0f MiB current "
        "usage, %.0f MiB available under %.0f MiB limit, %.0f MiB remaining "
        "after allocation based on current usage; %s.",
        allocation_name,
        required_bytes / mib,
        cgroup_usage / mib,
        cgroup_available / mib,
        cgroup_limit / mib,
        remaining_bytes / mib,
        status,
    )
