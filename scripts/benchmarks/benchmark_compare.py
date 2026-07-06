#!/usr/bin/env python
"""Benchmark multiple patches and create comparison plots.

This tool benchmarks multiple patches, saves results to JSON, and creates
comprehensive comparison plots for performance analysis.

Usage:
    # Benchmark all patches and create plots
    python scripts/benchmarks/benchmark_compare.py --all

    # Benchmark specific patches
    python scripts/benchmarks/benchmark_compare.py patch1.apr patch2.apr

    # Load existing results and regenerate plots
    python scripts/benchmarks/benchmark_compare.py --load results.json

    # Custom output directory
    python scripts/benchmarks/benchmark_compare.py --all --output-dir ./my_benchmarks
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def benchmark_patch_file(
    patch_path: Path,
    buffer_size: int = 512,
    iterations: int = 1000,
    sample_rate: int = 44100,
    verbose: bool = True,
) -> dict[str, Any]:
    """Benchmark a single patch file.

    Args:
        patch_path: Path to the patch file
        buffer_size: Audio buffer size in samples
        iterations: Number of iterations to run
        sample_rate: Sample rate in Hz
        verbose: Print progress information

    Returns:
        Dictionary containing benchmark results
    """
    from src.gui.utils.patch_loader import HeadlessPatchRenderer

    # Load the patch
    renderer = HeadlessPatchRenderer()
    if not renderer.load_patch(patch_path):
        raise RuntimeError(f"Failed to load patch: {patch_path}")

    if verbose:
        print(f"  Benchmarking: {patch_path.name}")
        print(f"    Modules: {renderer.get_module_count()}")
        print(f"    Connections: {renderer.get_connection_count()}")

    # Warmup: render a few buffers first
    try:
        for _ in range(5):
            renderer.render(buffer_size, sample_rate)
    except Exception as e:
        if verbose:
            print(f"    ⚠️  Warmup failed: {e}")

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
            if i == 0 and verbose:
                print(f"    ⚠️  Render failed: {e}")
            continue

    if not times_ns:
        raise RuntimeError(f"No successful renders for {patch_path.name}")

    if successful_renders < iterations and verbose:
        print(f"    ⚠️  {iterations - successful_renders}/{iterations} renders failed")

    times_us = [t / 1000.0 for t in times_ns]
    budget_us = (buffer_size / sample_rate) * 1e6

    return {
        "patch_name": patch_path.stem,
        "patch_file": str(patch_path),
        "min_us": min(times_us),
        "mean_us": statistics.fmean(times_us),
        "median_us": statistics.median(times_us),
        "max_us": max(times_us),
        "std_us": statistics.stdev(times_us) if len(times_us) > 1 else 0.0,
        "p95_us": (
            statistics.quantiles(times_us, n=20)[18]
            if len(times_us) >= 20
            else max(times_us)
        ),
        "p99_us": (
            statistics.quantiles(times_us, n=100)[98]
            if len(times_us) >= 100
            else max(times_us)
        ),
        "budget_us": budget_us,
        "headroom_pct": 100 * (1 - statistics.fmean(times_us) / budget_us),
        "buffer_size": buffer_size,
        "sample_rate": sample_rate,
        "iterations": successful_renders,
        "modules": renderer.get_module_count(),
        "connections": renderer.get_connection_count(),
        "timestamp": datetime.now().isoformat(),
    }


def create_comparison_plots(results: list[dict[str, Any]], output_dir: Path) -> None:
    """Create comprehensive comparison plots.

    Args:
        results: List of benchmark results
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort by mean processing time
    results = sorted(results, key=lambda r: r["mean_us"])

    patch_names = [r["patch_name"] for r in results]
    mean_times = [r["mean_us"] for r in results]
    budgets = [r["budget_us"] for r in results]
    headrooms = [r["headroom_pct"] for r in results]
    modules = [r["modules"] for r in results]

    # Create figure with subplots
    plt.figure(figsize=(16, 12))

    # 1. Mean processing time comparison
    ax1 = plt.subplot(3, 2, 1)
    bars = ax1.barh(patch_names, mean_times, color="steelblue")
    ax1.axvline(
        budgets[0],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Time Budget",
        alpha=0.7,
    )
    ax1.set_xlabel("Processing Time (µs)", fontsize=11)
    ax1.set_title("Mean Processing Time per Patch", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(axis="x", alpha=0.3)

    # Color bars based on performance
    for bar, headroom in zip(bars, headrooms, strict=False):
        if headroom > 50:
            bar.set_color("green")
        elif headroom > 20:
            bar.set_color("yellowgreen")
        elif headroom > 0:
            bar.set_color("orange")
        else:
            bar.set_color("red")

    # 2. CPU Headroom comparison
    ax2 = plt.subplot(3, 2, 2)
    bars = ax2.barh(patch_names, headrooms)
    ax2.axvline(0, color="red", linestyle="-", linewidth=2, alpha=0.5)
    ax2.axvline(20, color="orange", linestyle="--", linewidth=1, alpha=0.5)
    ax2.axvline(50, color="green", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_xlabel("CPU Headroom (%)", fontsize=11)
    ax2.set_title("CPU Headroom per Patch", fontsize=12, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    # Color bars
    for bar, headroom in zip(bars, headrooms, strict=False):
        if headroom > 50:
            bar.set_color("green")
        elif headroom > 20:
            bar.set_color("yellowgreen")
        elif headroom > 0:
            bar.set_color("orange")
        else:
            bar.set_color("red")

    # 3. Processing time distribution (box plot)
    ax3 = plt.subplot(3, 2, 3)
    data_for_box = []
    for r in results:
        # Create representative distribution from stats
        data_for_box.append(
            [r["min_us"], r["median_us"], r["mean_us"], r["p95_us"], r["max_us"]]
        )

    bp = ax3.boxplot(data_for_box, vert=False, patch_artist=True)
    ax3.set_yticklabels(patch_names)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    ax3.set_xlabel("Processing Time (µs)", fontsize=11)
    ax3.set_title("Processing Time Distribution", fontsize=12, fontweight="bold")
    ax3.grid(axis="x", alpha=0.3)

    # 4. Complexity vs Performance
    ax4 = plt.subplot(3, 2, 4)
    scatter = ax4.scatter(
        modules,
        mean_times,
        s=100,
        alpha=0.6,
        c=headrooms,
        cmap="RdYlGn",
        edgecolors="black",
        linewidth=1,
    )
    ax4.set_xlabel("Number of Modules", fontsize=11)
    ax4.set_ylabel("Mean Processing Time (µs)", fontsize=11)
    ax4.set_title("Complexity vs Performance", fontsize=12, fontweight="bold")
    ax4.grid(alpha=0.3)

    # Add patch name labels
    for i, name in enumerate(patch_names):
        ax4.annotate(
            name,
            (modules[i], mean_times[i]),
            fontsize=8,
            alpha=0.7,
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.colorbar(scatter, ax=ax4, label="CPU Headroom (%)")

    # 5. Performance metrics table
    ax5 = plt.subplot(3, 2, 5)
    ax5.axis("off")

    table_data = []
    for r in results[:10]:  # Top 10 patches
        status = (
            "✅"
            if r["headroom_pct"] > 50
            else (
                "✓"
                if r["headroom_pct"] > 20
                else "⚠️"
                if r["headroom_pct"] > 0
                else "❌"
            )
        )
        table_data.append(
            [
                r["patch_name"][:15],
                f"{r['mean_us']:.1f}",
                f"{r['headroom_pct']:.1f}%",
                status,
            ]
        )

    table = ax5.table(
        cellText=table_data,
        colLabels=["Patch", "Mean (µs)", "Headroom", "Status"],
        cellLoc="left",
        loc="center",
        colWidths=[0.4, 0.2, 0.2, 0.2],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(weight="bold", color="white")

    ax5.set_title("Top 10 Performing Patches", fontsize=12, fontweight="bold", pad=20)

    # 6. Summary statistics
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis("off")

    avg_time = statistics.mean(mean_times)
    avg_headroom = statistics.mean(headrooms)
    best_patch = results[0]["patch_name"]
    worst_patch = results[-1]["patch_name"]

    passing_patches = sum(1 for r in results if r["headroom_pct"] > 0)
    excellent_patches = sum(1 for r in results if r["headroom_pct"] > 50)

    summary_text = f"""
    BENCHMARK SUMMARY
    {"=" * 40}

    Total Patches:           {len(results)}
    Passing (>0% headroom):  {passing_patches}
    Excellent (>50% headroom): {excellent_patches}

    Average Processing Time: {avg_time:.1f} µs
    Average CPU Headroom:    {avg_headroom:.1f}%

    Best Performer:          {best_patch}
    Slowest Patch:           {worst_patch}

    Buffer Size:             {results[0]["buffer_size"]} samples
    Sample Rate:             {results[0]["sample_rate"]} Hz
    Time Budget:             {results[0]["budget_us"]:.1f} µs
    """

    ax6.text(
        0.1,
        0.5,
        summary_text,
        fontsize=10,
        family="monospace",
        verticalalignment="center",
    )

    plt.tight_layout()

    # Save plot
    output_file = output_dir / "benchmark_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\n📊 Saved comparison plot: {output_file}")

    # Create individual metric plots
    create_detailed_plots(results, output_dir)


def create_detailed_plots(results: list[dict[str, Any]], output_dir: Path) -> None:
    """Create additional detailed plots.

    Args:
        results: List of benchmark results
        output_dir: Directory to save plots
    """
    # Percentile comparison
    fig, ax = plt.subplots(figsize=(12, 8))

    patch_names = [r["patch_name"] for r in results]
    x = np.arange(len(patch_names))
    width = 0.2

    ax.bar(
        x - width * 1.5, [r["min_us"] for r in results], width, label="Min", alpha=0.8
    )
    ax.bar(
        x - width * 0.5,
        [r["median_us"] for r in results],
        width,
        label="Median",
        alpha=0.8,
    )
    ax.bar(
        x + width * 0.5,
        [r["p95_us"] for r in results],
        width,
        label="95th %ile",
        alpha=0.8,
    )
    ax.bar(
        x + width * 1.5, [r["max_us"] for r in results], width, label="Max", alpha=0.8
    )

    ax.axhline(
        results[0]["budget_us"],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Time Budget",
    )

    ax.set_xlabel("Patch", fontsize=11)
    ax.set_ylabel("Processing Time (µs)", fontsize=11)
    ax.set_title("Processing Time Percentiles", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(patch_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "benchmark_percentiles.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"📊 Saved percentiles plot: {output_file}")

    plt.close("all")


def save_results_json(results: list[dict[str, Any]], output_file: Path) -> None:
    """Save benchmark results to JSON file.

    Args:
        results: List of benchmark results
        output_file: Path to output JSON file
    """
    data = {
        "benchmark_date": datetime.now().isoformat(),
        "total_patches": len(results),
        "results": results,
        "summary": {
            "avg_processing_time_us": statistics.mean(r["mean_us"] for r in results),
            "avg_headroom_pct": statistics.mean(r["headroom_pct"] for r in results),
            "best_patch": min(results, key=lambda r: r["mean_us"])["patch_name"],
            "slowest_patch": max(results, key=lambda r: r["mean_us"])["patch_name"],
        },
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"💾 Saved results: {output_file}")


def load_results_json(input_file: Path) -> list[dict[str, Any]]:
    """Load benchmark results from JSON file.

    Args:
        input_file: Path to input JSON file

    Returns:
        List of benchmark results
    """
    with open(input_file) as f:
        data = json.load(f)

    return data["results"]


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark multiple patches and create comparison plots"
    )
    parser.add_argument(
        "patches",
        nargs="*",
        type=Path,
        help="Patch files to benchmark",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Benchmark all patches in examples/patches/",
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
        "--output-dir",
        type=Path,
        default=Path("scripts/benchmarks/benchmark_plots"),
        help="Output directory for plots and results",
    )
    parser.add_argument(
        "--load",
        type=Path,
        help="Load existing results from JSON and regenerate plots",
    )

    args = parser.parse_args()

    # Load existing results
    if args.load:
        if not args.load.exists():
            print(f"Error: Results file not found: {args.load}")
            return 1

        print(f"Loading results from: {args.load}")
        results = load_results_json(args.load)
        create_comparison_plots(results, args.output_dir)
        return 0

    # Determine which patches to benchmark
    patch_files = []

    if args.all:
        patches_dir = Path("examples/patches")
        if not patches_dir.exists():
            print(f"Error: Patches directory not found: {patches_dir}")
            return 1

        patch_files = sorted(patches_dir.glob("*.apr"))
        if not patch_files:
            print(f"No .apr patch files found in {patches_dir}")
            return 1
    elif args.patches:
        patch_files = args.patches
    else:
        parser.print_help()
        return 1

    # Benchmark all patches
    print(f"\n{'=' * 70}")
    print(f"BENCHMARKING {len(patch_files)} PATCHES")
    print(f"{'=' * 70}\n")

    results = []
    failed_patches = []

    for i, patch_file in enumerate(patch_files, 1):
        print(f"[{i}/{len(patch_files)}]", end=" ")

        if not patch_file.exists():
            print(f"⚠️  File not found: {patch_file}")
            failed_patches.append(str(patch_file))
            continue

        try:
            result = benchmark_patch_file(
                patch_file,
                args.buffer_size,
                args.iterations,
                args.sample_rate,
                verbose=True,
            )
            results.append(result)

            # Print quick summary
            status = (
                "✅"
                if result["headroom_pct"] > 50
                else (
                    "✓"
                    if result["headroom_pct"] > 20
                    else "⚠️"
                    if result["headroom_pct"] > 0
                    else "❌"
                )
            )
            print(
                f"    {status} {result['mean_us']:.1f}µs "
                + f"({result['headroom_pct']:.1f}% headroom)\n"
            )

        except Exception as e:
            print(f"    ❌ Failed: {e}\n")
            failed_patches.append(str(patch_file))
            continue

    if not results:
        print("Error: No successful benchmarks")
        return 1

    # Save results and create plots
    print(f"\n{'=' * 70}")
    print("GENERATING REPORTS")
    print(f"{'=' * 70}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = args.output_dir / f"benchmark_results_{timestamp}.json"

    save_results_json(results, results_file)
    create_comparison_plots(results, args.output_dir)

    # Print summary
    print(f"\n{'=' * 70}")
    print("BENCHMARK COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total patches:       {len(patch_files)}")
    print(f"Successful:          {len(results)}")
    print(f"Failed:              {len(failed_patches)}")

    if failed_patches:
        print("\nFailed patches:")
        for patch in failed_patches:
            print(f"  - {patch}")

    avg_time = statistics.mean(r["mean_us"] for r in results)
    avg_headroom = statistics.mean(r["headroom_pct"] for r in results)
    print(f"\nAverage time:        {avg_time:.1f}µs")
    print(f"Average headroom:    {avg_headroom:.1f}%")
    print(f"\nResults saved to:    {args.output_dir}")
    print(f"{'=' * 70}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
