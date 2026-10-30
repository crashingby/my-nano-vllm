"""从 Nsight SQLite 复算 TP 通信占比，区分 kernel 累计时间和墙钟并集。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sqlite3
from typing import Any


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """合并重叠的纳秒区间，同一 GPU 多个 stream 或多个 GPU 都只计一次。"""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def interval_duration(intervals: list[tuple[int, int]]) -> int:
    """返回区间并集的纳秒长度，避免把通信重叠误算为墙钟开销。"""
    return sum(end - start for start, end in merge_intervals(intervals))


def analyze(sqlite_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """只统计 tp_profile 范围内的 GPU kernel，返回汇总与逐 kernel 名称计数。

    NCCL kernel 的执行时间包含传输、协议处理及等待，不能直接解释为链路传输
    时间。进程 IPC、CPU/Gloo 和普通 H2D/D2H 不属于本统计的 NCCL 分子。
    """
    connection = sqlite3.connect(f"file:{sqlite_path.resolve()}?mode=ro", uri=True)
    try:
        ranges = list(connection.execute(
            "SELECT start, end FROM NVTX_EVENTS WHERE text = 'tp_profile' AND end IS NOT NULL"
        ))
        if len(ranges) != 1:
            raise ValueError(f"Expected one completed tp_profile NVTX range, got {len(ranges)}")
        window_start, window_end = ranges[0]
        window_ns = window_end - window_start
        if window_ns <= 0:
            raise ValueError("tp_profile must have positive duration")

        devices: dict[int, dict[str, Any]] = defaultdict(lambda: {
            "all": [], "nccl": [], "kernel_count": 0, "nccl_kernel_count": 0,
            "kernel_sum_ns": 0, "nccl_sum_ns": 0, "graph_kernel_count": 0,
            "graph_nccl_kernel_count": 0,
        })
        kernels: dict[str, dict[str, Any]] = {}
        all_intervals: list[tuple[int, int]] = []
        nccl_intervals: list[tuple[int, int]] = []
        compute_intervals: list[tuple[int, int]] = []
        # 使用 Graph node 的实际 kernel 活动，不使用 cudaGraphLaunch API 的主机耗时。
        rows = connection.execute(
            "SELECT k.start, k.end, k.deviceId, k.graphNodeId, s.value "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.shortName "
            "WHERE k.end > ? AND k.start < ? ORDER BY k.start",
            (window_start, window_end),
        )
        for start, end, device_id, graph_node_id, name in rows:
            start, end = max(start, window_start), min(end, window_end)
            duration = end - start  # 边界上的 kernel 仅保留落入回放窗口的部分。
            is_nccl = "nccl" in name.lower()
            is_graph = graph_node_id is not None and graph_node_id > 0
            device = devices[device_id]
            device["all"].append((start, end))
            device["kernel_count"] += 1
            device["kernel_sum_ns"] += duration
            device["graph_kernel_count"] += int(is_graph)
            all_intervals.append((start, end))
            if is_nccl:
                device["nccl"].append((start, end))
                device["nccl_kernel_count"] += 1
                device["nccl_sum_ns"] += duration
                device["graph_nccl_kernel_count"] += int(is_graph)
                nccl_intervals.append((start, end))
            else:
                compute_intervals.append((start, end))
            aggregate = kernels.setdefault(name, {"name": name, "is_nccl": is_nccl, "count": 0, "duration_ns": 0})
            aggregate["count"] += 1
            aggregate["duration_ns"] += duration

        kernel_sum = sum(device["kernel_sum_ns"] for device in devices.values())
        if not kernel_sum:
            raise ValueError("No GPU kernels found inside tp_profile")
        nccl_sum = sum(device["nccl_sum_ns"] for device in devices.values())
        gpu_reports = []
        for device_id, device in sorted(devices.items()):
            busy_ns = interval_duration(device.pop("all"))
            nccl_busy_ns = interval_duration(device.pop("nccl"))
            gpu_reports.append({
                "device_id": device_id, **device,
                "busy_union_ns": busy_ns, "nccl_busy_union_ns": nccl_busy_ns,
                "nccl_kernel_sum_share_percent": device["nccl_sum_ns"] / device["kernel_sum_ns"] * 100,
                "nccl_wall_percent": nccl_busy_ns / window_ns * 100,
            })

        nccl_union = interval_duration(nccl_intervals)
        compute_union = interval_duration(compute_intervals)
        all_union = interval_duration(all_intervals)
        # |A ∩ B| = |A| + |B| - |A ∪ B|；跨 GPU 同时计算和通信的区间不能重复扣除。
        overlap_ns = nccl_union + compute_union - all_union
        graph_calls = list(connection.execute(
            "SELECT s.value, COUNT(*), SUM(MIN(r.end, ?) - MAX(r.start, ?)) "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id = r.nameId "
            "WHERE r.end > ? AND r.start < ? AND s.value LIKE '%GraphLaunch%' GROUP BY s.value",
            (window_end, window_start, window_start, window_end),
        ))
        report = {
            "source": str(sqlite_path),
            "measurement": "NCCL GPU kernel activities within parent tp_profile NVTX range",
            "window_start_ns": window_start, "window_end_ns": window_end,
            "window_duration_s": window_ns / 1e9,
            "kernel_count": sum(device["kernel_count"] for device in devices.values()),
            "nccl_kernel_count": sum(device["nccl_kernel_count"] for device in devices.values()),
            "kernel_sum_ns": kernel_sum, "nccl_sum_ns": nccl_sum,
            "nccl_kernel_sum_share_percent": nccl_sum / kernel_sum * 100,
            "any_gpu_nccl_union_ns": nccl_union,
            "any_gpu_nccl_wall_percent": nccl_union / window_ns * 100,
            "any_gpu_kernel_union_ns": all_union,
            "nccl_compute_overlap_ns": overlap_ns,
            "nccl_without_traced_compute_ns": nccl_union - overlap_ns,
            "graph_launch_apis": [{"name": name, "count": count, "duration_ns": duration} for name, count, duration in graph_calls],
            "per_gpu": gpu_reports,
            "notes": [
                "跨 GPU/stream 累加的 kernel 时间不是墙钟时间。",
                "NCCL kernel 时间包含协议处理和等待，不能直接推出 PCIe 带宽或关键路径损失。",
                "普通 memcpy、CPU/Gloo 与多进程 RPC 不在 NCCL kernel 分子内。",
                "node tracing 增加采集开销，正式 TPS/延迟应读取未采集的 benchmark JSON。",
            ],
        }
        kernel_rows = sorted(kernels.values(), key=lambda row: row["duration_ns"], reverse=True)
        for row in kernel_rows:
            row["mean_us"] = row["duration_ns"] / row["count"] / 1000
            row["kernel_sum_share_percent"] = row["duration_ns"] / kernel_sum * 100
        return report, kernel_rows
    finally:
        connection.close()


def main() -> None:
    """导出可复算的通信 JSON 与 kernel CSV，不修改输入 SQLite。"""
    parser = argparse.ArgumentParser(description="Analyze TP NCCL kernels in an Nsight SQLite export.")
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    args = parser.parse_args()
    report, rows = analyze(args.sqlite)
    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    args.result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with args.result_path.with_suffix(".kernels.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: report[key] for key in (
        "window_duration_s", "kernel_count", "nccl_kernel_count",
        "nccl_kernel_sum_share_percent", "any_gpu_nccl_wall_percent", "graph_launch_apis",
    )}, indent=2))


if __name__ == "__main__":
    main()
