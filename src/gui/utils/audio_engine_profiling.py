"""Profiling tools for audio engine performance analysis.

This module provides profiling capabilities for the audio engine, allowing
real-time measurement of module processing times, memory usage, and bottleneck
identification.

Usage:
    # Enable profiling for a render cycle
    context = ProfilingRenderContext(num_samples=512)
    # ... render modules ...
    report = context.get_report()
    print(report)
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import numpy as np

from src.gui.audio_engine import RenderContext


class ProfilingRenderContext(RenderContext):
    """RenderContext with per-module timing and performance tracking.

    Extends the standard RenderContext to add detailed profiling information
    about module processing times, call counts, and performance bottlenecks.

    Attributes:
        enable_profiling: Whether profiling is active
        timings: Per-module timing measurements (in milliseconds)
        call_counts: Number of times each module was called
        total_time: Total processing time for all modules
    """

    def __init__(self, num_samples: int, enable_profiling: bool = True):
        """Initialize profiling context.

        Args:
            num_samples: Number of samples to render
            enable_profiling: Enable profiling (default: True)
        """
        super().__init__(num_samples)
        self.enable_profiling = enable_profiling
        self.timings: dict[str, list[float]] = defaultdict(list)
        self.call_counts: dict[str, int] = defaultdict(int)
        self.total_time: float = 0.0
        self._render_start_time: float = 0.0

    def begin_render(self) -> None:
        """Mark the start of a render cycle."""
        if self.enable_profiling:
            self._render_start_time = time.perf_counter_ns()

    def end_render(self) -> None:
        """Mark the end of a render cycle."""
        if self.enable_profiling and self._render_start_time > 0:
            elapsed_ms = (time.perf_counter_ns() - self._render_start_time) / 1e6
            self.total_time = elapsed_ms

    def time_module(self, module_name: str, operation: callable) -> Any:
        """Time a module operation.

        Args:
            module_name: Name of the module being profiled
            operation: Callable to execute and time

        Returns:
            Result of the operation
        """
        if not self.enable_profiling:
            return operation()

        # Time the operation
        start = time.perf_counter_ns()
        result = operation()
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6

        # Track statistics
        self.timings[module_name].append(elapsed_ms)
        self.call_counts[module_name] += 1

        return result

    def get_report(self, sort_by: str = "total") -> str:
        """Generate profiling report.

        Args:
            sort_by: Sort key - "total", "avg", "count", or "name"

        Returns:
            Formatted profiling report string
        """
        if not self.enable_profiling:
            return "Profiling disabled"

        lines = []
        lines.append("=" * 80)
        lines.append("AUDIO ENGINE PERFORMANCE PROFILE")
        lines.append("=" * 80)
        lines.append("")

        # Calculate statistics
        module_stats = []
        for name, times in self.timings.items():
            count = self.call_counts[name]
            total = sum(times)
            avg = total / count if count > 0 else 0
            min_time = min(times) if times else 0
            max_time = max(times) if times else 0
            pct = (100 * total / self.total_time) if self.total_time > 0 else 0

            module_stats.append(
                {
                    "name": name,
                    "count": count,
                    "total": total,
                    "avg": avg,
                    "min": min_time,
                    "max": max_time,
                    "pct": pct,
                }
            )

        # Sort
        if sort_by == "total":
            module_stats.sort(key=lambda x: x["total"], reverse=True)
        elif sort_by == "avg":
            module_stats.sort(key=lambda x: x["avg"], reverse=True)
        elif sort_by == "count":
            module_stats.sort(key=lambda x: x["count"], reverse=True)
        else:  # name
            module_stats.sort(key=lambda x: x["name"])

        # Format table
        lines.append(
            f"{'Module':<30} {'Calls':>6} {'Avg':>8} {'Min':>8} {'Max':>8} {'Total':>8} {'%':>6}"
        )
        lines.append("-" * 80)

        for stat in module_stats:
            lines.append(
                f"{stat['name']:<30} "
                f"{stat['count']:>6} "
                f"{stat['avg']:>7.2f}ms "
                f"{stat['min']:>7.2f}ms "
                f"{stat['max']:>7.2f}ms "
                f"{stat['total']:>7.2f}ms "
                f"{stat['pct']:>5.1f}%"
            )

        lines.append("-" * 80)
        lines.append(f"Total render time: {self.total_time:.2f}ms")
        lines.append("")

        # Budget analysis
        sample_rate = 44100  # Default
        buffer_size = self._num_samples
        budget_ms = (buffer_size / sample_rate) * 1000
        headroom_pct = 100 * (1 - self.total_time / budget_ms)

        lines.append(f"Buffer size: {buffer_size} samples")
        lines.append(f"Time budget: {budget_ms:.2f}ms")
        lines.append(f"CPU headroom: {headroom_pct:.1f}%")

        if headroom_pct > 50:
            lines.append("Status: ✅ EXCELLENT - Plenty of headroom")
        elif headroom_pct > 20:
            lines.append("Status: ✓  GOOD - Adequate headroom")
        elif headroom_pct > 0:
            lines.append("Status: ⚠️  MARGINAL - Low headroom")
        else:
            lines.append("Status: ❌ OVERBUDGET - Buffer underruns likely")

        lines.append("=" * 80)

        return "\n".join(lines)

    def get_json_stats(self) -> dict[str, Any]:
        """Get profiling statistics as JSON-serializable dictionary.

        Returns:
            Dictionary containing profiling data
        """
        stats = []
        for name, times in self.timings.items():
            count = self.call_counts[name]
            total = sum(times)
            avg = total / count if count > 0 else 0
            min_time = min(times) if times else 0
            max_time = max(times) if times else 0
            pct = (100 * total / self.total_time) if self.total_time > 0 else 0

            stats.append(
                {
                    "module": name,
                    "call_count": count,
                    "total_ms": total,
                    "avg_ms": avg,
                    "min_ms": min_time,
                    "max_ms": max_time,
                    "percent": pct,
                }
            )

        return {
            "total_time_ms": self.total_time,
            "module_stats": stats,
            "buffer_size": self._num_samples,
        }

    def get_top_bottlenecks(self, n: int = 5) -> list[tuple[str, float]]:
        """Get top N modules by total time.

        Args:
            n: Number of top modules to return

        Returns:
            List of (module_name, total_time_ms) tuples
        """
        module_totals = [(name, sum(times)) for name, times in self.timings.items()]
        module_totals.sort(key=lambda x: x[1], reverse=True)
        return module_totals[:n]

    def clear(self) -> None:
        """Clear profiling statistics."""
        self.timings.clear()
        self.call_counts.clear()
        self.total_time = 0.0
        self._render_start_time = 0.0


class ProfilingSession:
    """Multi-cycle profiling session for long-term analysis.

    Aggregates profiling data across multiple render cycles to identify
    consistent patterns and trends.

    Usage:
        session = ProfilingSession()

        for _ in range(100):
            context = ProfilingRenderContext(512)
            # ... render ...
            session.add_cycle(context)

        print(session.get_summary())
    """

    def __init__(self):
        """Initialize profiling session."""
        self.cycles: list[dict[str, Any]] = []
        self.total_cycles = 0

    def add_cycle(self, context: ProfilingRenderContext) -> None:
        """Add a render cycle's profiling data.

        Args:
            context: ProfilingRenderContext from a completed render
        """
        self.cycles.append(context.get_json_stats())
        self.total_cycles += 1

    def get_summary(self) -> str:
        """Generate summary report across all cycles.

        Returns:
            Formatted summary report
        """
        if not self.cycles:
            return "No profiling data collected"

        lines = []
        lines.append("=" * 80)
        lines.append(f"PROFILING SESSION SUMMARY ({self.total_cycles} cycles)")
        lines.append("=" * 80)
        lines.append("")

        # Aggregate statistics
        module_totals: dict[str, list[float]] = defaultdict(list)
        total_times = []

        for cycle in self.cycles:
            total_times.append(cycle["total_time_ms"])
            for module_stat in cycle["module_stats"]:
                module_totals[module_stat["module"]].append(module_stat["total_ms"])

        # Calculate aggregates
        stats = []
        for module_name, times in module_totals.items():
            stats.append(
                {
                    "name": module_name,
                    "avg": np.mean(times),
                    "min": np.min(times),
                    "max": np.max(times),
                    "std": np.std(times),
                    "total": np.sum(times),
                }
            )

        stats.sort(key=lambda x: x["total"], reverse=True)

        # Format table
        lines.append(
            f"{'Module':<30} {'Avg':>8} {'Min':>8} {'Max':>8} {'StdDev':>8} {'Total':>10}"
        )
        lines.append("-" * 80)

        for stat in stats:
            lines.append(
                f"{stat['name']:<30} "
                f"{stat['avg']:>7.2f}ms "
                f"{stat['min']:>7.2f}ms "
                f"{stat['max']:>7.2f}ms "
                f"{stat['std']:>7.2f}ms "
                f"{stat['total']:>9.2f}ms"
            )

        lines.append("-" * 80)
        lines.append(
            f"Average render time: {np.mean(total_times):.2f}ms "
            f"(±{np.std(total_times):.2f}ms)"
        )
        lines.append(f"Min render time: {np.min(total_times):.2f}ms")
        lines.append(f"Max render time: {np.max(total_times):.2f}ms")
        lines.append("=" * 80)

        return "\n".join(lines)

    def export_csv(self, filename: str) -> None:
        """Export profiling data to CSV file.

        Args:
            filename: Output filename
        """
        import csv

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["cycle", "module", "total_ms", "avg_ms", "call_count", "percent"]
            )

            for cycle_idx, cycle in enumerate(self.cycles):
                for module_stat in cycle["module_stats"]:
                    writer.writerow(
                        [
                            cycle_idx,
                            module_stat["module"],
                            module_stat["total_ms"],
                            module_stat["avg_ms"],
                            module_stat["call_count"],
                            module_stat["percent"],
                        ]
                    )

    def clear(self) -> None:
        """Clear session data."""
        self.cycles.clear()
        self.total_cycles = 0
