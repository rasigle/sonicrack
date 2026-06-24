# scripts/benchmarks/benchmark_output.py

from __future__ import annotations

from collections.abc import Iterable, Mapping


def print_benchmark_table(
    *,
    title: str,
    rows: Iterable[Mapping[str, object]],
    include_allocations: bool,
    include_component: bool = False,
) -> None:
    headers = []

    if include_component:
        headers.append("Component")

    headers.extend(["Buffer Size", "Min (us)", "Mean (us)", "Max (us)"])

    if include_allocations:
        headers.extend(["Current Bytes", "Peak Bytes"])

    formatted_rows: list[list[str]] = []

    for row in rows:
        values: list[str] = []

        if include_component:
            values.append(str(row["component"]))

        values.extend(
            [
                str(row["buffer_size"]),
                f"{float(row['min_us']):.3f}",
                f"{float(row['mean_us']):.3f}",
                f"{float(row['max_us']):.3f}",
            ]
        )

        if include_allocations:
            values.extend(
                [
                    f"{int(row['current_bytes']):,}",
                    f"{int(row['peak_bytes']):,}",
                ]
            )

        formatted_rows.append(values)

    widths = [
        max(len(header), *(len(row[index]) for row in formatted_rows))
        for index, header in enumerate(headers)
    ]

    numeric_columns = set(range(len(headers)))
    if include_component:
        numeric_columns.remove(0)

    def format_row(values: list[str]) -> str:
        cells = []
        for index, value in enumerate(values):
            if index in numeric_columns:
                cells.append(value.rjust(widths[index]))
            else:
                cells.append(value.ljust(widths[index]))
        return "  ".join(cells)

    print(title)
    print("=" * len(title))
    print(format_row(headers))
    print(format_row(["-" * width for width in widths]))

    for row in formatted_rows:
        print(format_row(row))
