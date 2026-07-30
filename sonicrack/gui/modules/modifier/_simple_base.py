"""Base class for simple single-control In→Out modifier modules."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QVBoxLayout

from sonicrack.gui.widgets.module_widget import ModuleWidget


class SimpleModifierBase(ModuleWidget):
    """Shared wiring for Volume / Pan / Clipper-style simple modifiers.

    Provides audio I/O ports, required-input defaults, and a single centered
    parameter knob control shell.
    """

    def _setup_audio_io(self, *, out_component: Any | None = None) -> None:
        """Add standard In / Out ports."""
        self.in_port = self.add_input("In")
        if out_component is not None:
            self.out_port = self.add_output("Out", component=out_component)
        else:
            self.out_port = self.add_output("Out")

    def get_required_inputs(self) -> list[str]:
        """Simple modifiers require the audio input."""
        return ["In"]

    def _build_single_knob_controls(
        self,
        knob: Any,
        param_name: str,
        *,
        on_change: Callable[[float], None] | None = None,
        spacing: int = 10,
    ) -> QVBoxLayout:
        """Create controls with one centered parameter knob and register it."""
        layout = self._begin_controls(spacing=spacing)
        self.bind_parameter_knob(knob, param_name, on_change=on_change)
        layout.addWidget(knob, alignment=Qt.AlignmentFlag.AlignCenter)
        self._finish_controls(layout)
        self.register_parameter(param_name, knob)
        return layout
