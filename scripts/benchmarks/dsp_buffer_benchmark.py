from __future__ import annotations

import argparse
import statistics
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from _benchmark_output import print_benchmark_table

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
from src.engine.dsp.filters.butterworth import ButterworthFilter

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


def collect_rows(
    buffer_sizes: list[int],
    iterations: int,
    include_allocations: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for buffer_size in buffer_sizes:
        print(f"Collecting rows for {buffer_size} bytes")
        for name, render in _render_cases(buffer_size).items():
            result = benchmark_render(render, iterations)

            row: dict[str, object] = {
                "component": name,
                "buffer_size": buffer_size,
                "min_us": result["min_us"],
                "mean_us": result["mean_us"],
                "max_us": result["max_us"],
            }

            if include_allocations:
                allocation_result = track_allocations(render, iterations)
                row.update(
                    {
                        "current_bytes": allocation_result["current_bytes"],
                        "peak_bytes": allocation_result["peak_bytes"],
                    }
                )

            rows.append(row)

    return rows


def plot_metric(
    rows: list[dict[str, object]],
    metric_key: str,
    metric_label: str,
    title: str,
    output_path: Path,
) -> None:
    components = sorted({str(row["component"]) for row in rows})

    plt.subplots(figsize=(10, 6))

    for component in components:
        component_rows = [row for row in rows if str(row["component"]) == component]
        component_rows.sort(key=lambda row: int(cast(int, row["buffer_size"])))

        x = [int(cast(int, row["buffer_size"])) for row in component_rows]
        y = [float(cast(float, row[metric_key])) for row in component_rows]

        plt.plot(x, y, marker="o", label=component)

    plt.xscale("log", base=2)
    plt.xticks(sorted({int(cast(int, row["buffer_size"])) for row in rows}))
    plt.xlabel("Buffer Size")
    plt.ylabel(metric_label)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_results(
    rows: list[dict[str, object]],
    *,
    include_allocations: bool,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_metric(
        rows,
        metric_key="mean_us",
        metric_label="Mean Time (us)",
        title="DSP Benchmark: Mean Time vs Buffer Size",
        output_path=output_dir / "dsp_mean_us.png",
    )

    plot_metric(
        rows,
        metric_key="min_us",
        metric_label="Min Time (us)",
        title="DSP Benchmark: Min Time vs Buffer Size",
        output_path=output_dir / "dsp_min_us.png",
    )

    plot_metric(
        rows,
        metric_key="max_us",
        metric_label="Max Time (us)",
        title="DSP Benchmark: Max Time vs Buffer Size",
        output_path=output_dir / "dsp_max_us.png",
    )

    if include_allocations:
        plot_metric(
            rows,
            metric_key="peak_bytes",
            metric_label="Peak Bytes",
            title="DSP Benchmark: Peak Allocation vs Buffer Size",
            output_path=output_dir / "dsp_peak_bytes.png",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--buffer-sizes",
        nargs="+",
        type=int,
        default=[64, 128, 256, 512, 1024, 2048],
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--track-allocations", action="store_true", default=True)
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("benchmark_plots"),
    )
    args = parser.parse_args()

    rows = collect_rows(
        buffer_sizes=args.buffer_sizes,
        iterations=args.iterations,
        include_allocations=args.track_allocations,
    )

    print_benchmark_table(
        title="DSP Buffer Benchmark",
        rows=rows,
        include_allocations=args.track_allocations,
        include_component=True,
    )

    if args.plot:
        plot_results(
            rows,
            include_allocations=args.track_allocations,
            output_dir=args.plot_dir,
        )
        print(f"\nSaved plots to: {args.plot_dir}")


if __name__ == "__main__":
    main()
