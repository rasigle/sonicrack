"""Benchmark the sounddevice callback formatting path.

Run from the repository root:

    uv run --extra audio-io python scripts/benchmarks/audio_callback_benchmark.py
"""

from __future__ import annotations

import argparse
import statistics
import time
import tracemalloc

import numpy as np

from scripts.benchmarks._benchmark_output import print_benchmark_table
from src.audio_io import AudioOutput


def benchmark_buffer_size(buffer_size: int, iterations: int) -> dict[str, float]:
    source = np.zeros((buffer_size, 2), dtype=np.float32)
    outdata = np.zeros((buffer_size, 2), dtype=np.float32)
    audio = AudioOutput(buffer_size=buffer_size, audio_callback=lambda n: source[:n])
    audio._master_volume_smoothing_samples = 0

    durations_ns: list[int] = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        audio._sounddevice_callback(outdata, buffer_size, None, None)
        durations_ns.append(time.perf_counter_ns() - start)

    durations_us = [duration / 1000.0 for duration in durations_ns]
    return {
        "min_us": min(durations_us),
        "mean_us": statistics.fmean(durations_us),
        "max_us": max(durations_us),
    }


def track_allocations(buffer_size: int, iterations: int) -> dict[str, int]:
    source = np.zeros((buffer_size, 2), dtype=np.float32)
    outdata = np.zeros((buffer_size, 2), dtype=np.float32)
    audio = AudioOutput(buffer_size=buffer_size, audio_callback=lambda n: source[:n])
    audio._master_volume_smoothing_samples = 0

    for _ in range(32):
        audio._sounddevice_callback(outdata, buffer_size, None, None)

    tracemalloc.start()
    for _ in range(iterations):
        audio._sounddevice_callback(outdata, buffer_size, None, None)
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "current_bytes": current_bytes,
        "peak_bytes": peak_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--buffer-sizes",
        nargs="+",
        type=int,
        default=[64, 128, 256, 512, 1024, 2048],
    )
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--track-allocations", action="store_true", default=True)
    args = parser.parse_args()

    if args.track_allocations:
        print("buffer_size,min_us,mean_us,max_us,current_bytes,peak_bytes")
    else:
        print("buffer_size,min_us,mean_us,max_us")

    rows: list[dict[str, object]] = []

    for buffer_size in args.buffer_sizes:
        result = benchmark_buffer_size(buffer_size, args.iterations)

        row: dict[str, object] = {
            "buffer_size": buffer_size,
            "min_us": result["min_us"],
            "mean_us": result["mean_us"],
            "max_us": result["max_us"],
        }

        if args.track_allocations:
            allocation_result = track_allocations(buffer_size, args.iterations)
            row.update(
                {
                    "current_bytes": allocation_result["current_bytes"],
                    "peak_bytes": allocation_result["peak_bytes"],
                }
            )

        rows.append(row)

    print_benchmark_table(
        title="Audio Callback Benchmark",
        rows=rows,
        include_allocations=args.track_allocations,
    )


if __name__ == "__main__":
    main()
