#!/usr/bin/env python
"""Benchmark audio patches for performance testing.

This tool measures the processing time for rendering audio patches,
helping identify performance bottlenecks and validate optimizations.

Usage:
    python scripts/benchmarks/benchmark_patch.py <patch_file.apr>
    python scripts/benchmarks/benchmark_patch.py --all
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path


def benchmark_patch_file(
    patch_path: Path,
    buffer_size: int = 512,
    iterations: int = 1000,
    sample_rate: int = 44100,
) -> dict[str, float]:
    """Benchmark a patch file.

    Args:
        patch_path: Path to the patch file
        buffer_size: Audio buffer size in samples
        iterations: Number of iterations to run
        sample_rate: Sample rate in Hz

    Returns:
        Dictionary containing benchmark results
    """
    from src.gui.utils.patch_loader import HeadlessPatchRenderer

    # Load the patch
    renderer = HeadlessPatchRenderer()
    if not renderer.load_patch(patch_path):
        raise RuntimeError(f"Failed to load patch: {patch_path}")

    print(f"Loaded patch: {patch_path.name}")
    print(f"  Modules: {renderer.get_module_count()}")
    print(f"  Connections: {renderer.get_connection_count()}")

    # Warmup: render a few buffers first
    try:
        for _ in range(5):
            renderer.render(buffer_size, sample_rate)
    except Exception as e:
        print(f"⚠️  Warmup failed: {e}")
        print("   Patch may not be compatible with headless rendering")

    # Benchmark rendering
    times_ns = []
    successful_renders = 0

    for i in range(iterations):
        try:
            start = time.perf_counter_ns()
            _ = renderer.render(buffer_size, sample_rate)
            elapsed = time.perf_counter_ns() - start
            times_ns.append(elapsed)
            successful_renders += 1
        except Exception as e:
            if i == 0:
                # Print error on first failure
                print(f"⚠️  Render failed: {e}")
                print("   Continuing with what renders successfully...")
            continue

    if not times_ns:
        raise RuntimeError("No successful renders - patch may be incompatible")

    if successful_renders < iterations:
        print(f"⚠️  {iterations - successful_renders}/{iterations} renders failed")

    times_us = [t / 1000.0 for t in times_ns]
    budget_us = (buffer_size / sample_rate) * 1e6

    return {
        "min_us": min(times_us),
        "mean_us": statistics.fmean(times_us),
        "median_us": statistics.median(times_us),
        "max_us": max(times_us),
        "p95_us": (
            statistics.quantiles(times_us, n=20)[18]
            if len(times_us) >= 20
            else max(times_us)
        ),  # 95th percentile
        "p99_us": (
            statistics.quantiles(times_us, n=100)[98]
            if len(times_us) >= 100
            else max(times_us)
        ),  # 99th percentile
        "budget_us": budget_us,
        "headroom_pct": 100 * (1 - statistics.fmean(times_us) / budget_us),
        "buffer_size": buffer_size,
        "iterations": successful_renders,
    }


def print_benchmark_results(patch_name: str, results: dict[str, float]) -> None:
    """Print benchmark results in a readable format."""
    print(f"\n{'=' * 70}")
    print(f"BENCHMARK RESULTS: {patch_name}")
    print(f"{'=' * 70}")
    print(f"Buffer Size:     {results['buffer_size']} samples")
    print(f"Iterations:      {results['iterations']}")
    print("")
    print("Processing Time:")
    print(f"  Min:           {results['min_us']:.1f} µs")
    print(f"  Mean:          {results['mean_us']:.1f} µs")
    print(f"  Median:        {results['median_us']:.1f} µs")
    print(f"  Max:           {results['max_us']:.1f} µs")
    print(f"  95th percentile: {results['p95_us']:.1f} µs")
    print(f"  99th percentile: {results['p99_us']:.1f} µs")
    print("")
    print(f"Time Budget:     {results['budget_us']:.1f} µs")
    print(f"CPU Headroom:    {results['headroom_pct']:.1f}%")

    # Status indicator
    if results["headroom_pct"] > 50:
        status = "✅ EXCELLENT"
    elif results["headroom_pct"] > 20:
        status = "✓  GOOD"
    elif results["headroom_pct"] > 0:
        status = "⚠️  MARGINAL"
    else:
        status = "❌ OVERBUDGET"

    print(f"Status:          {status}")
    print(f"{'=' * 70}\n")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark audio patches for performance testing"
    )
    parser.add_argument(
        "patch_file",
        nargs="?",
        type=Path,
        help="Path to patch file to benchmark",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=512,
        help="Audio buffer size in samples (default: 512)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of benchmark iterations (default: 1000)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Sample rate in Hz (default: 44100)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Benchmark all patches in examples/patches/",
    )

    args = parser.parse_args()

    if args.all:
        # Benchmark all patches
        patches_dir = Path("examples/patches")
        if not patches_dir.exists():
            print(f"Error: Patches directory not found: {patches_dir}")
            return 1

        patch_files = list(patches_dir.glob("*.apr"))
        if not patch_files:
            print(f"No .apr patch files found in {patches_dir}")
            return 1

        print(f"Found {len(patch_files)} patches to benchmark")
        for patch_file in sorted(patch_files):
            results = benchmark_patch_file(
                patch_file,
                args.buffer_size,
                args.iterations,
                args.sample_rate,
            )
            print_benchmark_results(patch_file.name, results)
    elif args.patch_file:
        # Benchmark single patch
        if not args.patch_file.exists():
            print(f"Error: Patch file not found: {args.patch_file}")
            return 1

        results = benchmark_patch_file(
            args.patch_file,
            args.buffer_size,
            args.iterations,
            args.sample_rate,
        )
        print_benchmark_results(args.patch_file.name, results)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
