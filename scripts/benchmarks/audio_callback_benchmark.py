"""Benchmark the sounddevice callback formatting path.

Run from the repository root:

    uv run --extra audio-io python scripts/benchmarks/audio_callback_benchmark.py
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--buffer-sizes",
        nargs="+",
        type=int,
        default=[64, 128, 256, 512, 1024, 2048],
    )
    parser.add_argument("--iterations", type=int, default=5000)
    args = parser.parse_args()

    print("buffer_size,min_us,mean_us,max_us")
    for buffer_size in args.buffer_sizes:
        result = benchmark_buffer_size(buffer_size, args.iterations)
        print(
            f"{buffer_size},"
            f"{result['min_us']:.3f},"
            f"{result['mean_us']:.3f},"
            f"{result['max_us']:.3f}"
        )


if __name__ == "__main__":
    main()
