#!/usr/bin/env python
"""Compare benchmark results across multiple runs.

This tool loads multiple benchmark result files and creates comparison
plots to analyze performance changes over time or between different
configurations.

Usage:
    # Compare two benchmark runs
    python scripts/benchmarks/compare_benchmark_runs.py \
        results_before.json results_after.json

    # Compare multiple runs with custom labels
    python scripts/benchmarks/compare_benchmark_runs.py \
        --labels "Before" "After" "Optimized" \
        run1.json run2.json run3.json

    # Output to custom directory
    python scripts/benchmarks/compare_benchmark_runs.py \
        --output-dir ./comparisons \
        baseline.json optimized.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_benchmark_results(file_path: Path) -> dict[str, Any]:
    """Load benchmark results from JSON file.

    Args:
        file_path: Path to JSON results file

    Returns:
        Dictionary with benchmark data
    """
    with open(file_path) as f:
        return json.load(f)


def extract_common_patches(runs: list[dict[str, Any]]) -> list[str]:
    """Extract patch names that appear in all runs.

    Args:
        runs: List of benchmark run data

    Returns:
        List of common patch names
    """
    if not runs:
        return []

    # Get patch names from first run
    common_patches = {r["patch_name"] for r in runs[0]["results"]}

    # Intersect with all other runs
    for run in runs[1:]:
        run_patches = {r["patch_name"] for r in run["results"]}
        common_patches &= run_patches

    return sorted(common_patches)


def get_patch_results(
    run_data: dict[str, Any], patch_name: str
) -> dict[str, Any] | None:
    """Get results for a specific patch from a run.

    Args:
        run_data: Benchmark run data
        patch_name: Name of the patch

    Returns:
        Patch results or None if not found
    """
    for result in run_data["results"]:
        if result["patch_name"] == patch_name:
            return result
    return None


def create_comparison_plots(
    runs: list[dict[str, Any]], labels: list[str], output_dir: Path
) -> None:
    """Create comprehensive comparison plots.

    Args:
        runs: List of benchmark run data
        labels: Labels for each run
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    common_patches = extract_common_patches(runs)

    if not common_patches:
        print("Error: No common patches found across all runs")
        return

    print(f"Comparing {len(common_patches)} common patches across {len(runs)} runs")

    # Create main comparison figure
    plt.subplots(figsize=(18, 12))

    # 1. Mean processing time comparison
    ax1 = plt.subplot(2, 3, 1)
    x = np.arange(len(common_patches))
    width = 0.8 / len(runs)

    for i, (run, label) in enumerate(zip(runs, labels, strict=False)):
        times = [get_patch_results(run, p)["mean_us"] for p in common_patches]
        offset = (i - len(runs) / 2) * width + width / 2
        ax1.bar(x + offset, times, width, label=label, alpha=0.8)

    ax1.set_xlabel("Patch", fontsize=10)
    ax1.set_ylabel("Mean Processing Time (µs)", fontsize=10)
    ax1.set_title("Processing Time Comparison", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(common_patches, rotation=45, ha="right", fontsize=8)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # 2. CPU Headroom comparison
    ax2 = plt.subplot(2, 3, 2)
    ax2.axhline(0, color="red", linestyle="-", linewidth=1, alpha=0.5)
    ax2.axhline(20, color="orange", linestyle="--", linewidth=1, alpha=0.5)
    ax2.axhline(50, color="green", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_xlabel("Patch", fontsize=10)
    ax2.set_ylabel("CPU Headroom (%)", fontsize=10)
    ax2.set_title("CPU Headroom Comparison", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(common_patches, rotation=45, ha="right", fontsize=8)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # 3. Performance change heatmap (if 2 runs)
    if len(runs) == 2:
        ax3 = plt.subplot(2, 3, 3)

        # Calculate percentage changes
        changes = []
        for patch in common_patches:
            before = get_patch_results(runs[0], patch)["mean_us"]
            after = get_patch_results(runs[1], patch)["mean_us"]
            pct_change = ((after - before) / before) * 100
            changes.append(pct_change)

        ax3.axvline(0, color="black", linestyle="-", linewidth=2)
        ax3.set_xlabel("Performance Change (%)", fontsize=10)
        ax3.set_title(
            f"Change: {labels[0]} → {labels[1]}", fontsize=12, fontweight="bold"
        )
        ax3.grid(axis="x", alpha=0.3)

        # Add value labels
        for i, (_, change) in enumerate(zip(common_patches, changes, strict=False)):
            label = f"{change:+.1f}%"
            color = "white" if abs(change) > 15 else "black"
            ax3.text(
                change / 2,
                i,
                label,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color=color,
            )
    else:
        ax3 = plt.subplot(2, 3, 3)
        ax3.axis("off")
        ax3.text(
            0.5,
            0.5,
            "Change heatmap\navailable for\n2-run comparisons",
            ha="center",
            va="center",
            fontsize=11,
        )

    # 4. Average metrics comparison
    ax4 = plt.subplot(2, 3, 4)

    metric_names = ["Avg Time (µs)", "Avg Headroom (%)", "Passing Patches"]
    x_metrics = np.arange(len(metric_names))
    width_metric = 0.8 / len(runs)

    for i, (run, label) in enumerate(zip(runs, labels, strict=False)):
        avg_time = run["summary"]["avg_processing_time_us"]
        avg_headroom = run["summary"]["avg_headroom_pct"]
        passing = sum(1 for r in run["results"] if r["headroom_pct"] > 0)

        # Normalize for visualization
        normalized_values = [
            avg_time / 1000,  # Convert to ms for better scale
            avg_headroom,
            passing / len(run["results"]) * 100,  # As percentage
        ]

        offset = (i - len(runs) / 2) * width_metric + width_metric / 2
        ax4.bar(
            x_metrics + offset, normalized_values, width_metric, label=label, alpha=0.8
        )

    ax4.set_ylabel("Value", fontsize=10)
    ax4.set_title("Summary Metrics", fontsize=12, fontweight="bold")
    ax4.set_xticks(x_metrics)
    ax4.set_xticklabels(
        ["Avg Time\n(ms)", "Avg Headroom\n(%)", "Passing\n(%)"], fontsize=9
    )
    ax4.legend()
    ax4.grid(axis="y", alpha=0.3)

    # 5. Percentile comparison for selected patches
    ax5 = plt.subplot(2, 3, 5)

    # Select 5 most interesting patches (highest variability)
    if len(runs) >= 2:
        variability = {}
        for patch in common_patches:
            times = [get_patch_results(run, patch)["mean_us"] for run in runs]
            variability[patch] = max(times) - min(times)

        top_patches = sorted(variability.items(), key=lambda x: -x[1])[:5]
        selected_patches = [p[0] for p in top_patches]
    else:
        selected_patches = common_patches[:5]

    x_sel = np.arange(len(selected_patches))

    for i, (run, label) in enumerate(zip(runs, labels, strict=False)):
        p95_times = [get_patch_results(run, p)["p95_us"] for p in selected_patches]
        offset = (i - len(runs) / 2) * width + width / 2
        ax5.bar(x_sel + offset, p95_times, width, label=label, alpha=0.8)

    ax5.set_xlabel("Patch", fontsize=10)
    ax5.set_ylabel("95th Percentile Time (µs)", fontsize=10)
    ax5.set_title(
        "95th Percentile - Top Variable Patches", fontsize=12, fontweight="bold"
    )
    ax5.set_xticks(x_sel)
    ax5.set_xticklabels(selected_patches, rotation=45, ha="right", fontsize=8)
    ax5.legend()
    ax5.grid(axis="y", alpha=0.3)

    # 6. Summary table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")

    table_data = []
    for run, label in zip(runs, labels, strict=False):
        table_data.append(
            [
                label[:20],
                f"{run['summary']['avg_processing_time_us']:.1f}",
                f"{run['summary']['avg_headroom_pct']:.1f}%",
                run["summary"]["best_patch"][:15],
                run["summary"]["slowest_patch"][:15],
            ]
        )

    table = ax6.table(
        cellText=table_data,
        colLabels=[
            "Run",
            "Avg Time\n(µs)",
            "Avg Headroom",
            "Best Patch",
            "Slowest Patch",
        ],
        cellLoc="left",
        loc="center",
        colWidths=[0.2, 0.15, 0.15, 0.25, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)

    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(weight="bold", color="white")

    ax6.set_title("Run Comparison Summary", fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout()

    # Save plot
    output_file = output_dir / "benchmark_run_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\n📊 Saved comparison plot: {output_file}")

    # Create detailed change analysis if 2 runs
    if len(runs) == 2:
        create_detailed_change_analysis(runs, labels, common_patches, output_dir)

    plt.close("all")


def create_detailed_change_analysis(
    runs: list[dict[str, Any]], labels: list[str], patches: list[str], output_dir: Path
) -> None:
    """Create detailed analysis of changes between two runs.

    Args:
        runs: List of 2 benchmark run data
        labels: Labels for each run
        patches: List of patch names to analyze
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Calculate changes
    changes_data = []
    for patch in patches:
        before = get_patch_results(runs[0], patch)
        after = get_patch_results(runs[1], patch)

        time_change = ((after["mean_us"] - before["mean_us"]) / before["mean_us"]) * 100
        headroom_change = after["headroom_pct"] - before["headroom_pct"]

        changes_data.append(
            {
                "patch": patch,
                "time_change_pct": time_change,
                "headroom_change": headroom_change,
                "before_time": before["mean_us"],
                "after_time": after["mean_us"],
            }
        )

    # Sort by absolute change
    changes_data.sort(key=lambda x: abs(x["time_change_pct"]), reverse=True)

    # 1. Top improvements and regressions
    ax1 = axes[0, 0]
    top_n = min(10, len(changes_data))

    improvements = [d for d in changes_data if d["time_change_pct"] < 0][:top_n]
    regressions = [d for d in changes_data if d["time_change_pct"] > 0][:top_n]

    y_pos = np.arange(len(improvements) + len(regressions))
    changes = [d["time_change_pct"] for d in improvements] + [
        d["time_change_pct"] for d in regressions
    ]
    names = [d["patch"] for d in improvements] + [d["patch"] for d in regressions]
    colors = ["green"] * len(improvements) + ["red"] * len(regressions)

    ax1.barh(y_pos, changes, color=colors, alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel("Performance Change (%)", fontsize=10)
    ax1.set_title(
        f"Top Changes: {labels[0]} → {labels[1]}", fontsize=11, fontweight="bold"
    )
    ax1.axvline(0, color="black", linestyle="-", linewidth=2)
    ax1.grid(axis="x", alpha=0.3)

    # 2. Scatter: Before vs After
    ax2 = axes[0, 1]
    before_times = [d["before_time"] for d in changes_data]
    after_times = [d["after_time"] for d in changes_data]
    colors_scatter = [
        "green" if d["time_change_pct"] < 0 else "red" for d in changes_data
    ]

    ax2.scatter(before_times, after_times, c=colors_scatter, alpha=0.6, s=100)

    # Add diagonal line (no change)
    max_time = max(max(before_times), max(after_times))
    ax2.plot([0, max_time], [0, max_time], "k--", alpha=0.3, linewidth=2)

    ax2.set_xlabel(f"{labels[0]} Time (µs)", fontsize=10)
    ax2.set_ylabel(f"{labels[1]} Time (µs)", fontsize=10)
    ax2.set_title("Before vs After Performance", fontsize=11, fontweight="bold")
    ax2.grid(alpha=0.3)

    # 3. Change distribution histogram
    ax3 = axes[1, 0]
    all_changes = [d["time_change_pct"] for d in changes_data]

    ax3.hist(all_changes, bins=20, color="steelblue", alpha=0.7, edgecolor="black")
    ax3.axvline(0, color="red", linestyle="--", linewidth=2)
    ax3.axvline(
        np.mean(all_changes),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(all_changes):.1f}%",
    )
    ax3.set_xlabel("Performance Change (%)", fontsize=10)
    ax3.set_ylabel("Number of Patches", fontsize=10)
    ax3.set_title("Distribution of Performance Changes", fontsize=11, fontweight="bold")
    ax3.legend()
    ax3.grid(axis="y", alpha=0.3)

    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis("off")

    impr_pat = sum(1 for d in changes_data if d["time_change_pct"] < 0)
    regr_pat = sum(1 for d in changes_data if d["time_change_pct"] > 0)
    unchanged_patches = sum(1 for d in changes_data if abs(d["time_change_pct"]) < 1)

    avg_change = np.mean(all_changes)
    median_change = np.median(all_changes)
    max_improvement = min(all_changes)
    max_regression = max(all_changes)

    summary_text = f"""
    CHANGE ANALYSIS SUMMARY
    {"=" * 40}

    Total Patches:         {len(changes_data)}
    Improved:              {impr_pat} ({impr_pat / len(changes_data) * 100:.1f}%)
    Regressed:             {regr_pat} ({regr_pat / len(changes_data) * 100:.1f}%)
    Unchanged (±1%):       {unchanged_patches}

    Average Change:        {avg_change:+.2f}%
    Median Change:         {median_change:+.2f}%

    Best Improvement:      {max_improvement:.2f}%
    Worst Regression:      {max_regression:+.2f}%

    Overall Status:        {"✅ IMPROVED" if avg_change < 0 else "❌ REGRESSED"}
    """

    ax4.text(
        0.1,
        0.5,
        summary_text,
        fontsize=9,
        family="monospace",
        verticalalignment="center",
    )

    plt.tight_layout()

    output_file = output_dir / "benchmark_change_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"📊 Saved change analysis: {output_file}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare benchmark results across multiple runs"
    )
    parser.add_argument(
        "result_files",
        nargs="+",
        type=Path,
        help="Benchmark result JSON files to compare",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Labels for each run (default: auto-generated)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scripts/benchmarks/benchmark_plots"),
        help="Output directory for comparison plots",
    )

    args = parser.parse_args()

    # Validate input files
    for file_path in args.result_files:
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return 1

    # Load all runs
    print(f"\nLoading {len(args.result_files)} benchmark runs...")
    runs = []
    for file_path in args.result_files:
        print(f"  Loading: {file_path}")
        runs.append(load_benchmark_results(file_path))

    # Generate labels
    if args.labels:
        if len(args.labels) != len(runs):
            print(
                f"Error: Number of labels ({len(args.labels)}) must match "
                + f"number of result files ({len(runs)})"
            )
            return 1
        labels = args.labels
    else:
        # Auto-generate labels from filenames
        labels = [
            f.stem.replace("benchmark_results_", "Run ") for f in args.result_files
        ]

    # Create comparison plots
    print("\nGenerating comparison plots...")
    create_comparison_plots(runs, labels, args.output_dir)

    print(f"\n{'=' * 70}")
    print("COMPARISON COMPLETE")
    print(f"{'=' * 70}")
    print(f"Compared {len(runs)} runs")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'=' * 70}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
