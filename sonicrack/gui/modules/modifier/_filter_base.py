"""Shared helpers and base class for filter GUI modules.

Keeps Filter, Resonant Filter, and Acid Filter as separate product modules while
centralizing duplicated port setup, type mapping, and filter-specific wiring.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, overload

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel

from sonicrack.gui.widgets.module_widget import ModuleWidget

# Re-export runtime helpers used by filter modules.
from sonicrack.runtime.helpers import (  # noqa: F401
    ramp_if_changed,
    read_optional_port,
)

ButterworthFilterType = Literal["low", "high", "band"]
ResonantFilterType = Literal["low", "high", "band", "notch"]

FILTER_TYPE_UI_TO_ENGINE: dict[str, str] = {
    "Low-pass": "low",
    "High-pass": "high",
    "Band-pass": "band",
    "Notch": "notch",
    "low": "low",
    "high": "high",
    "band": "band",
    "notch": "notch",
}

BUTTERWORTH_TYPE_ITEMS: tuple[str, ...] = ("Low-pass", "High-pass", "Band-pass")
RESONANT_TYPE_ITEMS: tuple[str, ...] = (
    "Low-pass",
    "High-pass",
    "Band-pass",
    "Notch",
)


@overload
def normalize_filter_type(
    text: str,
    *,
    allow_notch: Literal[False] = False,
) -> ButterworthFilterType: ...


@overload
def normalize_filter_type(
    text: str,
    *,
    allow_notch: Literal[True],
) -> ResonantFilterType: ...


def normalize_filter_type(
    text: str,
    *,
    allow_notch: bool = False,
) -> ButterworthFilterType | ResonantFilterType:
    """Map UI or engine filter type text to the engine filter type string."""
    mapped = FILTER_TYPE_UI_TO_ENGINE.get(text, "low")
    if mapped == "notch" and not allow_notch:
        return "low"
    if allow_notch:
        if mapped in ("low", "high", "band", "notch"):
            return mapped  # type: ignore[return-value]
        return "low"
    if mapped in ("low", "high", "band"):
        return mapped  # type: ignore[return-value]
    return "low"


def create_filter_type_combo(
    items: Sequence[str],
    *,
    on_changed: Callable[[str], None] | None = None,
    centered: bool = False,
) -> tuple[QHBoxLayout, QComboBox]:
    """Build a Type: combo row used by multi-mode filter modules."""
    layout = QHBoxLayout()
    if centered:
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(QLabel("Type:"))
    combo = QComboBox()
    combo.addItems(list(items))
    if on_changed is not None:
        combo.currentTextChanged.connect(on_changed)
    layout.addWidget(combo)
    return layout, combo


class FilterModuleBase(ModuleWidget):
    """Common filter wiring: audio ports, sample-rate rebuild, type param."""

    # Subclasses narrow this to their concrete runtime-param tuple type.
    _runtime_filter_params: Any = None

    def _setup_filter_ports(
        self,
        *,
        cutoff_cv: bool = False,
        env_cv: bool = False,
        accent_cv: bool = False,
    ) -> None:
        """Add the standard audio I/O ports and optional CV inputs."""
        self.in_port = self.add_input("In")
        if cutoff_cv:
            self.cutoff_cv_port = self.add_input("Cutoff CV")
        if env_cv:
            self.env_cv_port = self.add_input("Env CV")
        if accent_cv:
            self.accent_cv_port = self.add_input("Accent CV")
        self.out_port = self.add_output("Out")

    def _on_global_sample_rate_changed(self, new_sample_rate: int) -> None:
        """Rebuild the engine component and clear cached runtime filter params."""
        del new_sample_rate
        self.component = self.create_engine_component()
        if hasattr(self, "_runtime_filter_params"):
            self._runtime_filter_params = None

    def get_required_inputs(self) -> list[str]:
        """Filter modules always require an audio input."""
        return ["In"]

    def _register_filter_type_parameter(self, combo: QComboBox) -> None:
        """Register a filter type combo for preset get/set."""
        self.register_parameter(
            "filter_type",
            combo,
            getter="currentText",
            setter="setCurrentText",
        )
