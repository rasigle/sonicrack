#!/usr/bin/env python
"""Benchmark engine components using PresetBuilder.

This script demonstrates how to benchmark audio patches using the
PresetBuilder API, which works without GUI dependencies.

Usage:
    python scripts/benchmarks/benchmark_engine_patches.py
"""

from __future__ import annotations

import statistics
import time

from src.engine.presets.preset_builder import PresetBuilder


def benchmark_patch(patch, buffer_size: int = 512, iterations: int = 1000):
    """Benchmark a patch built with PresetBuilder.

    Args:
        patch: Built patch from PresetBuilder
        buffer_size: Audio buffer size
        iterations: Number of iterations

    Returns:
        Dict with benchmark results
    """
    # Warmup
    for _ in range(10):
        patch.get_samples_vectorized(buffer_size)

    # Benchmark
    times_ns = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        audio = patch.get_samples_vectorized(buffer_size)
        times_ns.append(time.perf_counter_ns() - start)

    times_us = [t / 1000.0 for t in times_ns]
    budget_us = (buffer_size / 44100) * 1e6

    return {
        "min_us": min(times_us),
        "mean_us": statistics.fmean(times_us),
        "median_us": statistics.median(times_us),
        "max_us": max(times_us),
        "budget_us": budget_us,
        "headroom_pct": 100 * (1 - statistics.fmean(times_us) / budget_us),
    }


def main():
    """Run benchmark examples."""
    print("=" * 70)
    print("ENGINE PATCH BENCHMARKS (PresetBuilder)")
    print("=" * 70)
    print()

    # Example 1: Simple Oscillator
    print("1. Simple Sine Oscillator")
    print("-" * 70)
    patch1 = PresetBuilder().sine(440, amplitude=0.8).build()
    results1 = benchmark_patch(patch1)
    print(f"  Mean: {results1['mean_us']:.1f}µs")
    print(f"  Budget: {results1['budget_us']:.1f}µs")
    print(f"  Headroom: {results1['headroom_pct']:.1f}%")
    print()

    # Example 2: Oscillator + ADSR
    print("2. Oscillator + ADSR Envelope")
    print("-" * 70)
    patch2 = PresetBuilder().sine(440, amplitude=0.8).adsr(0.1, 0.2, 0.7, 0.3).build()
    results2 = benchmark_patch(patch2)
    print(f"  Mean: {results2['mean_us']:.1f}µs")
    print(f"  Budget: {results2['budget_us']:.1f}µs")
    print(f"  Headroom: {results2['headroom_pct']:.1f}%")
    print()

    # Example 3: Oscillator + Volume + Panner
    print("3. Oscillator + Volume + Panner")
    print("-" * 70)
    patch3 = (
        PresetBuilder().triangle(120, amplitude=0.8).volume(0.7).panner(0.5).build()
    )
    results3 = benchmark_patch(patch3)
    print(f"  Mean: {results3['mean_us']:.1f}µs")
    print(f"  Budget: {results3['budget_us']:.1f}µs")
    print(f"  Headroom: {results3['headroom_pct']:.1f}%")
    print()

    # Example 4: Complex Chain with Filter
    print("4. Complex Processing Chain with Filter")
    print("-" * 70)
    patch4 = (
        PresetBuilder()
        .sawtooth(110, amplitude=0.7)
        .adsr(0.05, 0.1, 0.8, 0.2)
        .acid_filter(cutoff=1000, resonance=0.7)
        .volume(0.7)
        .build()
    )
    results4 = benchmark_patch(patch4)
    print(f"  Mean: {results4['mean_us']:.1f}µs")
    print(f"  Budget: {results4['budget_us']:.1f}µs")
    print(f"  Headroom: {results4['headroom_pct']:.1f}%")
    print()

    # Example 5: Multiple Oscillators
    print("5. Multiple Oscillators Mixed")
    print("-" * 70)
    patch5 = (
        PresetBuilder()
        .sine(440, amplitude=0.4)
        .sine(880, amplitude=0.3)
        .sine(220, amplitude=0.3)
        .build()
    )
    results5 = benchmark_patch(patch5)
    print(f"  Mean: {results5['mean_us']:.1f}µs")
    print(f"  Budget: {results5['budget_us']:.1f}µs")
    print(f"  Headroom: {results5['headroom_pct']:.1f}%")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_results = [results1, results2, results3, results4, results5]
    avg_time = statistics.mean(r["mean_us"] for r in all_results)
    avg_headroom = statistics.mean(r["headroom_pct"] for r in all_results)
    print(f"Average processing time: {avg_time:.1f}µs")
    print(f"Average headroom: {avg_headroom:.1f}%")
    print()
    print("All patches processed successfully! ✅")


if __name__ == "__main__":
    main()
