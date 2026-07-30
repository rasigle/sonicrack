from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from PyQt6.QtGui import QColor
from soniclab import Panner

from sonicrack.gui.modules.modifier._simple_base import SimpleModifierBase
from sonicrack.gui.widgets import Knob
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import float_parameter, read_samples
from sonicrack.runtime.specs import RuntimeParameters

if TYPE_CHECKING:
    from sonicrack.patching.port import Port

logger = logging.getLogger(__name__)


@register_module()
class SimplePannerModule(SimpleModifierBase):
    """Simple panner module without modulation input."""

    runtime_kind = "panner"

    metadata = ModuleMetadata(
        title="Panner",
        category=ModuleCategory.MODIFIER,
        description="Simple stereo panner without modulation input",
    )

    def __init__(self):
        """Initialize simple panner module."""
        super().__init__(
            width=140,
            height=175,
            color=QColor(160, 60, 160),
        )

        # Create engine component FIRST (before ports)
        self.component = self.create_engine_component()

        # Add ports with component reference
        self.in_port: Port = self.add_input("In")
        self.out_port: Port = self.add_output("Out", component=self.component)

        self.pan_knob = Knob(
            label="Pan", min_value=-1.0, max_value=1.0, default_value=0.0
        )
        self._build_single_knob_controls(
            self.pan_knob,
            "position",
            on_change=self._on_pan_changed,
        )

    def _on_pan_changed(self, pan_value: float) -> None:
        """Handle pan knob changes by updating pan component."""
        if self.component:
            self.component.position = pan_value
            logger.debug(f"Panner: position value set to {pan_value:.3f}")

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Create the panner component."""
        del input_components, modulation_components
        return Panner(0.0)

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        """Pan the connected input for one render cycle."""
        if self._require_input_or_silence(num_samples):
            return

        samples = read_samples(self.in_port, num_samples)

        position = float_parameter(parameters, "position", self.pan_knob.get_value)
        self.component.position = position

        left, right = self.component.pan_vectorized(samples)
        self.out_port.write(np.column_stack((left, right)))
