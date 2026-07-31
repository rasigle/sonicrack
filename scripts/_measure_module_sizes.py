"""Measure GUI module fixed sizes vs content size hints."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication([])

# Import via GUI entry to avoid circular import with registry.
from sonicrack.gui.main_window import ModularSynthWindow  # noqa: E402, F401
from sonicrack.patching.registry import initialize_module_registry  # noqa: E402

# Comfort padding below content so bottom margin isn't clipped.
CONTENT_PAD = 12
# Shrink only when spare exceeds this (keeps intentional breathing room).
SPARE_SHRINK_THRESHOLD = 28


def round_up_5(value: int) -> int:
    return ((value + 4) // 5) * 5


def main() -> None:
    registry = initialize_module_registry()
    print(
        f"{'Name':24} {'W':4} {'H':4} {'IdealH':6} {'dH':5} "
        f"{'ChildW':6} {'LayoutH':7} {'Action'}"
    )
    print("-" * 80)

    actions: list[tuple[str, str, int, int]] = []

    for name in sorted(registry.list_modules()):
        cls = registry.get(name)
        if cls is None:
            continue
        try:
            module = cls()
        except Exception as exc:  # noqa: BLE001
            print(f"{name:24} ERROR {exc}")
            continue

        width = module.module_width
        height = module.module_height
        title_h = module._title_bar_height()

        controls = getattr(module, "controls_widget", None)
        if controls is None:
            print(
                f"{name:24} {width:4} {height:4} {'—':>6} {'—':>5} {'—':>6} "
                f"{'—':>7} NO_CONTROLS"
            )
            continue

        controls.setFixedWidth(width)
        layout = controls.layout()
        if layout is None:
            print(
                f"{name:24} {width:4} {height:4} "
                f"{'—':>6} {'—':>5} {'—':>6} {'—':>7} NO_LAYOUT"
            )
            continue

        layout.activate()
        # Prefer total minimum height under the fixed width constraint.
        total_min = layout.totalMinimumSize()
        total_hint = layout.totalSizeHint()
        layout_h = max(total_min.height(), total_hint.height())
        children = controls.childrenRect()
        child_w = children.x() + children.width() if children.isValid() else 0

        ideal_h = round_up_5(title_h + layout_h + CONTENT_PAD)
        delta = ideal_h - height

        if delta > 4:
            action = f"GROW +{delta}"
            actions.append((name, "grow", height, ideal_h))
        elif delta < -SPARE_SHRINK_THRESHOLD:
            action = f"SHRINK {delta}"
            actions.append((name, "shrink", height, ideal_h))
        else:
            action = "OK"

        print(
            f"{name:24} {width:4} {height:4} {ideal_h:6} {delta:5} "
            f"{child_w:6} {layout_h:7} {action}"
        )

    if actions:
        print("\nSuggested height updates:")
        for name, kind, old, new in actions:
            print(f"  {kind:6} {name:24} {old} -> {new}")
    else:
        print("\nAll modules within thresholds.")


if __name__ == "__main__":
    main()
