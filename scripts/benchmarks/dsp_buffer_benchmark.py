"""Benchmark core DSP buffer rendering paths.

Run from the repository root:

    uv run python scripts/benchmarks/dsp_buffer_benchmark.py --track-allocations
"""

from __future__ import annotations

import argparse
import statistics
import time
import tracemalloc
from collections.abc import Callable

import numpy as np

from src.engine import (
    Chain,
    Delay,
    ModulatedVolume,
    Reverb,
    SawtoothOscillator,
    SineOscillator,
    SquareOscillator,
    TriangleOscillator,
    Volume,
)
from src.engine.filter import ButterworthFilter

RenderFn = Callable[[], np.ndarray]


def _render_cases(buffer_size: int) -> dict[str, RenderFn]:
    filter_input = np.zeros(buffer_size, dtype=np.float32)
    butterworth = ButterworthFilter(cutoff=1200, order=4, sample_rate=44100)
    sine = SineOscillator(frequency=440, gain_db=-12)
    sawtooth = SawtoothOscillator(frequency=173, gain_db=-12, mode="analog")
    triangle = TriangleOscillator(frequency=211, gain_db=-12, mode="analog")
    square = SquareOscillator(frequency=137, gain_db=-12, mode="vcv")
    delay = Delay(
        SineOscillator(frequency=220, gain_db=-12),
        delay_time=0.01,
        feedback=0.35,
        mix=0.55,
    )
    reverb = Reverb(
        SineOscillator(frequency=330, gain_db=-12),
        room_size=0.65,
        damping=0.35,
        mix=0.4,
    )
    modulated_chain = Chain(
        SineOscillator(frequency=330, gain_db=-12),
        ModulatedVolume(SineOscillator(frequency=3, amplitude=0.25, gain_db=None)),
        Volume(gain_db=-3),
    )

    return {
        "sine": lambda: sine.get_samples_vectorized(buffer_size),
        "sawtooth_analog": lambda: sawtooth.get_samples_vectorized(buffer_size),
        "triangle_analog": lambda: triangle.get_samples_vectorized(buffer_size),
        "square_vcv": lambda: square.get_samples_vectorized(buffer_size),
        "butterworth": lambda: butterworth.scale_vectorized(filter_input),
        "delay": lambda: delay.get_samples_vectorized(buffer_size),
        "reverb": lambda: reverb.get_samples_vectorized(buffer_size),
        "modulated_chain": lambda: modulated_chain.get_samples_vectorized(buffer_size),
    }


def benchmark_render(render: RenderFn, iterations: int) -> dict[str, float]:
    durations_ns: list[int] = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        render()
        durations_ns.append(time.perf_counter_ns() - start)

    durations_us = [duration / 1000.0 for duration in durations_ns]
    return {
        "min_us": min(durations_us),
        "mean_us": statistics.fmean(durations_us),
        "max_us": max(durations_us),
    }


def track_allocations(render: RenderFn, iterations: int) -> dict[str, int]:
    for _ in range(32):
        render()

    tracemalloc.start()
    for _ in range(iterations):
        render()
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
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--track-allocations", action="store_true")
    args = parser.parse_args()

    if args.track_allocations:
        print("component,buffer_size,min_us,mean_us,max_us,current_bytes,peak_bytes")
    else:
        print("component,buffer_size,min_us,mean_us,max_us")

    for buffer_size in args.buffer_sizes:
        for name, render in _render_cases(buffer_size).items():
            result = benchmark_render(render, args.iterations)
            row = (
                f"{name},{buffer_size},"
                f"{result['min_us']:.3f},"
                f"{result['mean_us']:.3f},"
                f"{result['max_us']:.3f}"
            )
            if args.track_allocations:
                allocation_result = track_allocations(render, args.iterations)
                row += (
                    f",{allocation_result['current_bytes']},"
                    f"{allocation_result['peak_bytes']}"
                )
            print(row)


if __name__ == "__main__":
    main()
