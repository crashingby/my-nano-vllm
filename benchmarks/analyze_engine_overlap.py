"""分析 profile_engine_overlap.py 的四组 SQLite，统计跨批次准备重叠。

只读取 Nsight 导出文件，不导入推理依赖或访问 GPU。时间戳以 ns 运算，输出为 ms。
固定比较 eager/Graph 与同步/异步；稳态口径为相邻两批均有 16 个单 token 请求。
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
from statistics import fmean, median
from typing import Any


Interval = tuple[int, int]
Range = dict[str, Any]
RangeIndex = dict[int, tuple[list[int], list[Range]]]
PREP_STAGES = {"schedule", "prep_state", "prep_inputs", "prep_slots", "prep_metadata"}
COMPUTE_STAGES = {"forward", "logits", "sample"}
PARENT_STAGES = {"execute", "sample_tokens", "reclaim"}
CASES = ("eager_sync", "eager_async", "graph_sync", "graph_async")


def merge(intervals: list[Interval]) -> list[Interval]:
    """合并重叠或相接区间，避免重复计算并行 kernel 和嵌套范围。

    Returns:
        merged: 合并重叠或相接范围后的有序区间列表，不含零长和反向区间。
    """
    merged: list[Interval] = []
    for start, end in sorted((a, b) for a, b in intervals if b > a):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def intersect(left: list[Interval], right: list[Interval]) -> list[Interval]:
    """对两组区间先求并集，再用双指针计算交集。

    Returns:
        overlaps: 两组区间并集的交集列表，不重复计算重叠部分。
    """
    left, right = merge(left), merge(right)
    overlaps: list[Interval] = []
    i = j = 0
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if end > start:
            overlaps.append((start, end))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return overlaps


def duration_ms(intervals: list[Interval]) -> float:
    """返回区间并集长度，将 Nsight 的纳秒转换为毫秒。

    Returns:
        elapsed_ms: 区间并集的总经过时间，单位为毫秒。
    """
    return sum(end - start for start, end in merge(intervals)) / 1e6


def build_index(ranges: list[Range], stages: set[str]) -> RangeIndex:
    """按主机线程建立范围索引；同一阶段集合中的范围必须互不嵌套。

    Returns:
        index: 主机线程 ID 到有序起点列表及阶段范围列表的索引。
    """
    grouped: dict[int, list[Range]] = defaultdict(list)
    for row in ranges:
        if row["stage"] in stages:
            grouped[row["tid"]].append(row)
    index: RangeIndex = {}
    for tid, rows in grouped.items():
        rows.sort(key=lambda row: row["start"])
        # 查找依赖非嵌套前提；观测入口变化时应报错，而非静默归错批次。
        assert all(a["end"] <= b["start"] for a, b in zip(rows, rows[1:]))
        index[tid] = ([row["start"] for row in rows], rows)
    return index


def lookup(index: RangeIndex, tid: int, timestamp: int) -> Range | None:
    """查找指定线程和 API 发起时间所在的 NVTX 范围。

    Returns:
        host_range: 指定线程中包含 API 发起时间的 NVTX 范围；未找到时为 None。
    """
    if tid not in index:
        return None
    starts, rows = index[tid]
    position = bisect_right(starts, timestamp) - 1
    if position >= 0 and timestamp < rows[position]["end"]:
        return rows[position]
    return None


def summarize(pairs: list[Range]) -> dict[str, int | float]:
    """汇总批次对，分母分别使用 CPU 准备并集和上一批 GPU 计算并集。

    Returns:
        summary: 批次对数量、准备与计算时间总量、重叠时长及重叠比例的字典。
    """
    cpu = sum(pair["cpu_prep_ms"] for pair in pairs)
    gpu = sum(pair["previous_gpu_compute_ms"] for pair in pairs)
    overlap = sum(pair["overlap_ms"] for pair in pairs)
    return {
        "pairs": len(pairs), "cpu_prep_ms": cpu, "gpu_compute_ms": gpu,
        "overlap_ms": overlap, "overlap_per_pair_ms": overlap / len(pairs),
        "cpu_prep_overlap_percent": 100 * overlap / cpu,
        "gpu_compute_overlap_percent": 100 * overlap / gpu,
    }


def analyze(root: Path, name: str) -> dict[str, Any]:
    """关联单组主机阶段与 GPU 时间线，保留批次对以便复核。

    Args:
        root: 四组 .sqlite 与同名 .trace.json 所在目录。
        name: CASES 中的执行方式和调度组合。

    Returns:
        result: 计时窗口、归属覆盖率、稳态重叠、事件等待及代表批次。

    Side Effects:
        只读打开 SQLite；发现缺失批次、范围嵌套或归属不足时抛出异常。
    """
    database = (root / f"{name}.sqlite").resolve()
    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        return _analyze_connection(connection, root, name)
    finally:
        connection.close()


def _analyze_connection(
    connection: sqlite3.Connection, root: Path, name: str,
) -> dict[str, Any]:
    """读取 Nsight 事件并按 correlation ID 和 batch ID 计算交集。

    Returns:
        result: 该组合的 GPU 活动、批次归属、重叠、等待及代表批次的分析字典。
    """
    tables = {row[0] for row in connection.execute(
        "select name from sqlite_master where type='table'"
    )}
    strings = {row["id"]: row["value"] for row in connection.execute(
        "select id,value from StringIds"
    )}
    ranges: list[Range] = []
    loop: Interval | None = None
    for row in connection.execute(
        "select start,end,text,textId,globalTid from NVTX_EVENTS where end is not null"
    ):
        label = row["text"] or strings.get(row["textId"], "")
        if label == "benchmark_loop":
            loop = (row["start"], row["end"])
        if "|" not in label:
            continue
        stage, batch = label.rsplit("|", 1)
        if batch.isdigit():
            ranges.append(dict(start=row["start"], end=row["end"], stage=stage,
                               batch=int(batch), tid=row["globalTid"]))
    assert loop is not None, f"{name}: missing benchmark_loop"
    phase = build_index(ranges, COMPUTE_STAGES)
    parent = build_index(ranges, PARENT_STAGES)

    runtime = [dict(row) for row in connection.execute(
        "select start,end,globalTid,correlationId,nameId from CUPTI_ACTIVITY_KIND_RUNTIME"
    )]
    api = {row["correlationId"]: row for row in runtime}
    kernels = [dict(row) for row in connection.execute(
        "select start,end,correlationId from CUPTI_ACTIVITY_KIND_KERNEL"
    )]
    compute: dict[int, list[Interval]] = defaultdict(list)
    num_mapped = 0
    for kernel in kernels:
        call = api.get(kernel["correlationId"])
        if call is None:
            continue
        if lookup(parent, call["globalTid"], call["start"]) is not None:
            num_mapped += 1
        active = lookup(phase, call["globalTid"], call["start"])
        if active is not None:
            compute[active["batch"]].append((kernel["start"], kernel["end"]))

    copies = [dict(row) for row in connection.execute(
        "select start,end,correlationId,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"
    )]
    d2h: dict[int, list[Interval]] = defaultdict(list)
    for copy in copies:
        call = api.get(copy["correlationId"])
        if copy["copyKind"] == 2 and call is not None:
            host = lookup(parent, call["globalTid"], call["start"])
            if host is not None:
                d2h[host["batch"]].append((copy["start"], copy["end"]))
    prep: dict[int, list[Interval]] = defaultdict(list)
    cpu_submit: dict[int, list[Interval]] = defaultdict(list)
    for row in ranges:
        if row["stage"] in PREP_STAGES:
            prep[row["batch"]].append((row["start"], row["end"]))
        if row["stage"] in COMPUTE_STAGES:
            cpu_submit[row["batch"]].append((row["start"], row["end"]))

    metadata = json.loads((root / f"{name}.trace.json").read_text())
    batches = {row["batch_id"]: row for row in metadata["batches"]}
    active_ids = sorted(key for key, batch in batches.items() if batch["num_tokens"] > 0)
    assert all(key in compute for key in active_ids), f"{name}: missing GPU batch"
    pairs: list[Range] = []
    copied_overlap: list[Interval] = []
    gpu_cross: list[Interval] = []
    for previous, current in zip(active_ids, active_ids[1:]):
        overlap = intersect(prep[current], compute[previous])
        pairs.append({
            "previous_batch": previous, "current_batch": current,
            "steady_decode": all(batches[key]["uniform_decode"] and
                                 batches[key]["num_reqs"] == 16
                                 for key in (previous, current)),
            "cpu_prep_ms": duration_ms(prep[current]),
            "previous_gpu_compute_ms": duration_ms(compute[previous]),
            "overlap_ms": duration_ms(overlap),
            "cpu_prep_intervals": merge(prep[current]),
            "previous_gpu_intervals": merge(compute[previous]),
            "overlap_intervals": overlap,
        })
        copied_overlap += intersect(d2h[previous], compute[current])
        gpu_cross += intersect(compute[previous], compute[current])
    steady = [pair for pair in pairs if pair["steady_decode"]]
    assert steady, f"{name}: no steady decode pairs"
    target = median(pair["overlap_ms"] for pair in steady)
    representative = min(steady, key=lambda pair: abs(pair["overlap_ms"] - target))

    # GPU 忙碌覆盖率按所有设备活动的并集计算，kernel 间隙不算忙碌。
    activities = [(row["start"], row["end"]) for row in kernels + copies]
    if "CUPTI_ACTIVITY_KIND_MEMSET" in tables:
        activities += [(row["start"], row["end"]) for row in connection.execute(
            "select start,end from CUPTI_ACTIVITY_KIND_MEMSET"
        )]
    busy = duration_ms(intersect(activities, [loop]))
    waits = dict(input_prep_event_ms=0.0, output_event_ms=0.0, other_event_ms=0.0)
    for call in runtime:
        if "EventSynchronize" not in strings.get(call["nameId"], ""):
            continue
        host = lookup(parent, call["globalTid"], call["start"])
        key = "other_event_ms"
        if host is not None and host["stage"] == "execute":
            key = "input_prep_event_ms"
        elif host is not None and host["stage"] == "reclaim":
            key = "output_event_ms"
        waits[key] += (call["end"] - call["start"]) / 1e6
    reclaims = {row["batch"]: row["start"] for row in ranges
                if row["stage"] == "reclaim"}
    # 回收早于 D2H 完成时记 0；真正阻塞时间另外计入 output_event_ms。
    lags = [max(0, reclaims[batch] - max(end for _, end in intervals)) / 1e6
            for batch, intervals in d2h.items() if batch in reclaims and intervals]
    result = {
        "name": name, "loop_ms": duration_ms([loop]), "gpu_busy_ms": busy,
        "gpu_busy_percent": 100 * busy / duration_ms([loop]),
        "gpu_kernel_count": len(kernels), "batch_attributed_kernel_count": num_mapped,
        "attribution_percent": 100 * num_mapped / len(kernels),
        "cpu_compute_submission_per_steady_batch_ms": fmean(
            duration_ms(cpu_submit[pair["current_batch"]]) for pair in steady
        ),
        "ready_to_reclaim": {
            "batches": len(lags), "mean_ms": fmean(lags) if lags else None,
            "median_ms": median(lags) if lags else None,
            "max_ms": max(lags) if lags else None,
        },
        "all_pairs": summarize(pairs), "steady_decode": summarize(steady),
        "input_and_output_event_waits": waits,
        "d2h_gpu_ms": duration_ms([span for spans in d2h.values() for span in spans]),
        "d2h_with_next_compute_overlap_ms": duration_ms(copied_overlap),
        "different_batch_compute_overlap_ms": duration_ms(gpu_cross),
        "representative": representative, "pairs": pairs,
    }
    assert result["attribution_percent"] > 99.9, f"{name}: insufficient attribution"
    if name.endswith("_sync"):
        assert result["steady_decode"]["overlap_ms"] < 0.01, name
    return result


def main() -> None:
    """读取四组诊断文件，打印摘要并在同目录写入 overlap_summary.json。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="四组 SQLite 和 trace JSON 所在目录")
    root = parser.parse_args().directory
    results = {name: analyze(root, name) for name in CASES}
    for result in results.values():
        print(json.dumps({key: value for key, value in result.items()
                          if key not in {"pairs", "representative"}}, ensure_ascii=False))
    (root / "overlap_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
