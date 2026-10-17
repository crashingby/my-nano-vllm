import contextlib
import time
from collections.abc import Generator

from nanovllm.config import  NanovllmConfig
from nanovllm.logger import init_logger

logger = init_logger(__name__)

# Shared global so backends.py can read the start time for Dynamo timing.
torch_compile_start_time: float = 0.0

cudagraph_capturing_enabled: bool = True

def validate_cudagraph_capturing_enabled() -> None:
    # used to monitor whether a cudagraph capturing is legal at runtime.
    # should be called before any cudagraph capturing.
    # if an illegal cudagraph capturing happens, raise an error.
    global cudagraph_capturing_enabled
    if not cudagraph_capturing_enabled:
        raise RuntimeError(
            "CUDA graph capturing detected at an inappropriate "
            "time. This operation is currently disabled."
        )

def set_cudagraph_capturing_enabled(enabled: bool) -> None:
    global cudagraph_capturing_enabled
    cudagraph_capturing_enabled = enabled