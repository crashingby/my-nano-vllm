"""比较大输入下 PyTorch、Triton 和 CuTe DSL 的 SiLU × Mul 性能。

在项目根目录运行：
    python tests/test_ops/test_silu_and_mul.py
    python tests/test_ops/test_silu_and_mul.py --dtypes float16 --num-tokens 1024 4096 16384
    python tests/test_ops/test_silu_and_mul.py --num-tokens 8192 --hidden-size 8192

默认输入 [4096, 8192]，输出 [4096, 4096]，测试三个实现和三种精度。
PyTorch 使用模型层的 torch.compile。编译、调优、预热和数值检查均在计时之外；
不捕获 CUDA Graph，直接调用原有 layer(x) 入口。CUDA Event 测量多次调用，
再折算为每次调用耗时，区间包含 Python 发射造成的 GPU 空闲时间。
固定输入重复使用，GPU cache 自然复用；有效带宽按两次读取、一次写入估算。
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import random
import statistics
import sys

import torch
import torch.nn.functional as F

# 支持从项目根目录直接运行文件，无需安装 pytest。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanovllm.config.kernel import SiluAndMulBackend
from nanovllm.model_executor.layers.activation import SiluAndMul
from nanovllm.platforms import current_platform


DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
# 低精度 native 计算可能在 SiLU 后先舍入，融合实现保留 FP32 中间结果。
TOLERANCES: dict[torch.dtype, tuple[float, float]] = {
    torch.float16: (2e-3, 1e-5),
    torch.bfloat16: (2e-2, 1e-4),
    torch.float32: (1e-5, 1e-6),
}
REFERENCE_CHUNK_ROWS = 256  # 分块参考计算，限制 CPU FP64 临时张量大小。


@dataclass
class BenchmarkResult:
    """保存一个实现的每调用耗时分布及逻辑数据量对应的有效带宽。"""

    backend: SiluAndMulBackend
    median_us: float
    min_us: float
    max_us: float
    effective_gbps: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析大输入尺寸和测量参数，默认运行三个实现的完整性能对比。"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--backends", nargs="+", choices=("torch", "triton", "cutedsl"),
        default=["torch", "triton", "cutedsl"],
    )
    parser.add_argument("--dtypes", nargs="+", choices=tuple(DTYPES), default=list(DTYPES))
    parser.add_argument(
        "--num-tokens", nargs="+", type=int, default=[4096],
        help="输入行数 T，可传多个值，例如 1024 4096 16384。",
    )
    parser.add_argument("--hidden-size", type=int, default=4096, help="H，输入行宽为 2H。")
    parser.add_argument("--warmup", type=int, default=10, help="计时前的直接调用预热次数。")
    parser.add_argument("--iters", type=int, default=50, help="每轮采样直接调用算子的次数。")
    parser.add_argument("--repeats", type=int, default=10, help="每个实现的计时采样次数。")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if any(size <= 0 for size in args.num_tokens):
        parser.error("--num-tokens must contain positive integers")
    for name in ("hidden_size", "warmup", "iters", "repeats"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    # 去除重复项，保留用户给出的测试顺序。
    args.num_tokens = list(dict.fromkeys(args.num_tokens))
    args.backends = list(dict.fromkeys(args.backends))
    args.dtypes = list(dict.fromkeys(args.dtypes))
    return args


def reference(x_cpu: torch.Tensor) -> torch.Tensor:
    """分块计算 CPU FP64 参考值，按输出精度保存结果以限制内存占用。"""
    num_tokens, input_width = x_cpu.shape
    expected = torch.empty((num_tokens, input_width // 2), dtype=x_cpu.dtype)
    for begin in range(0, num_tokens, REFERENCE_CHUNK_ROWS):
        end = min(begin + REFERENCE_CHUNK_ROWS, num_tokens)
        gate, up = x_cpu[begin:end].double().chunk(2, dim=-1)
        expected[begin:end] = F.silu(gate) * up
    return expected


def check_output(
    output: torch.Tensor, x: torch.Tensor, expected: torch.Tensor,
) -> float:
    """在计时前检查元数据、独立显存和全部输出元素，返回最大绝对误差。"""
    assert output.shape == (x.shape[0], x.shape[1] // 2), output.shape
    assert output.dtype == x.dtype, (output.dtype, x.dtype)
    assert output.device == x.device, (output.device, x.device)
    assert output.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
    rtol, atol = TOLERANCES[x.dtype]
    max_error = 0.0
    for begin in range(0, x.shape[0], REFERENCE_CHUNK_ROWS):
        end = min(begin + REFERENCE_CHUNK_ROWS, x.shape[0])
        actual = output[begin:end].detach().cpu()
        target = expected[begin:end]
        torch.testing.assert_close(actual, target, rtol=rtol, atol=atol)
        max_error = max(max_error, (actual.double() - target.double()).abs().max().item())
    return max_error


def prepare_benchmark(
    layer: SiluAndMul,
    x: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    """通过模型原有入口完成编译和预热，返回最后一次输出供数值检查。

    使用与计时一致的输入形状，提前完成 JIT、Triton autotune 和 CuTe 编译。
    所有调用使用当前 PyTorch stream，返回前同步以暴露异步执行错误。
    """
    for _ in range(args.warmup):
        output = layer(x)
    torch.cuda.synchronize(x.device)
    return output


def summarize(
    backend: SiluAndMulBackend, samples_us: list[float], logical_bytes: int,
) -> BenchmarkResult:
    """汇总每调用耗时，带宽按逻辑字节数计算，单位为十进制 GB/s。"""
    median_us = statistics.median(samples_us)
    if median_us <= 0:
        raise RuntimeError("CUDA Event measured nonpositive time; increase --iters")
    return BenchmarkResult(
        backend=backend,
        median_us=median_us,
        min_us=min(samples_us),
        max_us=max(samples_us),
        effective_gbps=logical_bytes / (median_us * 1e-6) / 1e9,
    )


def measure(
    layers: list[SiluAndMul],
    x: torch.Tensor,
    args: argparse.Namespace,
    logical_bytes: int,
) -> list[BenchmarkResult]:
    """逐轮打乱后端测量顺序，直接调用 layer(x) 并记录每调用耗时。

    每轮每个实现各测一次，减少固定顺序带来的温度和频率变化偏差。
    结束时等待 end Event，再读取 GPU 时间线上的区间；区间可能包含 host
    发射不及时造成的空闲，不能将结果等同于隔离后的单 kernel 执行时间。
    """
    samples: dict[SiluAndMulBackend, list[float]] = {layer.backend: [] for layer in layers}
    order = list(layers)
    rng = random.Random(args.seed)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # 在测量外初始化 Event 资源，避免首个样本承担初始化开销。
    start.record()
    end.record()
    end.synchronize()
    for _ in range(args.repeats):
        rng.shuffle(order)
        for layer in order:
            start.record()
            for _ in range(args.iters):
                output = layer(x)  # 包含原有入口的分派、输出分配及 kernel 发射。
            end.record()
            end.synchronize()
            elapsed_us = start.elapsed_time(end) * 1000 / args.iters
            samples[layer.backend].append(elapsed_us)
            del output  # 采样完成后释放最后一个输出，避免跨后端持有额外显存。
    return [summarize(layer.backend, samples[layer.backend], logical_bytes) for layer in layers]


def run_case(
    layers: list[SiluAndMul],
    dtype: torch.dtype,
    num_tokens: int,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    """共享一份大输入，准备并校验全部实现后统一计时，打印结果比较表。"""
    x = torch.randn((num_tokens, 2 * args.hidden_size), dtype=dtype, device=device)
    x_cpu = x.cpu()  # 保存输入供参考计算及输入保持检查使用。
    expected = reference(x_cpu)
    logical_bytes = (x.numel() + expected.numel()) * x.element_size()
    print(
        f"\n{dtype}: input={tuple(x.shape)}, output={tuple(expected.shape)}, "
        f"input+output={logical_bytes / (1 << 20):.1f} MiB/call",
        flush=True,
    )
    for layer in layers:
        print(f"PREP {layer.backend}: 编译/调优、预热及校验……", flush=True)
        try:
            output = prepare_benchmark(layer, x, args)
            max_error = check_output(output, x, expected)
            torch.testing.assert_close(x.cpu(), x_cpu, rtol=0, atol=0)
        except Exception as exc:
            raise RuntimeError(
                f"{layer.backend} failed: dtype={dtype}, shape={tuple(x.shape)}"
            ) from exc
        del output
        print(f"PASS {layer.backend}: max_abs_error={max_error:.3e}", flush=True)
    # CPU 校验会拉长间隔，再通过原有入口预热后开始统一计时。
    for _ in range(3):
        for layer in layers:
            layer(x)
    torch.cuda.synchronize(device)
    results = measure(layers, x, args, logical_bytes)
    baseline = next((item.median_us for item in results if item.backend == "torch"), None)
    print(
        f"{'backend':<10} {'median(us)':>12} {'min(us)':>12} "
        f"{'max(us)':>12} {'GB/s*':>12} {'speedup':>10}"
    )
    for item in results:
        speedup = f"{baseline / item.median_us:.2f}x" if baseline is not None else "N/A"
        print(
            f"{item.backend:<10} {item.median_us:>12.3f} {item.min_us:>12.3f} "
            f"{item.max_us:>12.3f} {item.effective_gbps:>12.2f} {speedup:>10}",
            flush=True,
        )


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    """初始化设备，按精度和 token 数执行性能对比，任何异常直接使脚本失败。"""
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用，请在能运行 nano-vllm 的 GPU 环境启动此脚本。")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("--device 必须指定 CUDA 设备，例如 cuda:0。")
    torch.cuda.set_device(device)
    device = torch.device("cuda", torch.cuda.current_device())
    if "bfloat16" in args.dtypes and not torch.cuda.is_bf16_supported():
        raise SystemExit("设备不支持 BF16，请使用 --dtypes float16 float32。")
    torch.manual_seed(args.seed)
    num_sms = current_platform.num_compute_units(device.index)
    print(f"GPU: {torch.cuda.get_device_name(device)}; SMs: {num_sms}; PyTorch: {torch.__version__}")
    print(f"每轮直接调用 {args.iters} 次，采样 {args.repeats} 轮；编译、调优和预热在计时之外。")
    print(
        "不使用 CUDA Graph；torch = torch.compile；计时可能包含 host 发射间隙。\n"
        "GB/s* 按两次读取、一次写入估算，数据可复用 GPU cache。",
        flush=True,
    )
    layers: list[SiluAndMul] = []
    for backend in args.backends:
        try:
            layers.append(SiluAndMul(backend=backend).eval())
        except ImportError as exc:
            raise RuntimeError(
                f"无法加载 {backend}；CuTe DSL 可用 python -m pip install -e '.[cutedsl]' 安装。"
            ) from exc
    # 多精度和多尺寸产生多个 Dynamo guards，保留足够缓存以持续使用编译路径。
    with torch._dynamo.config.patch(cache_size_limit=64):
        for name in args.dtypes:
            for num_tokens in args.num_tokens:
                run_case(layers, DTYPES[name], num_tokens, device, args)
    print("\n性能测试完成，所有计时前的数值检查通过。", flush=True)


if __name__ == "__main__":
    main()
