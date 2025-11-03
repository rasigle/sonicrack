"""Noise generator module with multiple noise types."""

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QComboBox

from engine.noise import NoiseGenerator
from src.gui.audio_module_interface import ModuleCategory
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget
from src.gui.module_registry import register_module


@register_module()
class NoiseModule(ModuleWidget):
    """Noise generator module with multiple noise types.

    Provides access to all 7 noise types:
    - White: Flat frequency spectrum
    - Pink: 1/f spectrum (equal energy per octave)
    - Brown: 1/f² spectrum (warmer, deeper)
    - Blue: f spectrum (brighter, more high-end)
    - Grey: Psychoacoustic flat (equal loudness)
    - Velvet: Sparse random impulses
    - Sample & Hold: Stepped random values
    """

    def __init__(self):
        """Initialize noise module."""
        super().__init__(
            width=200,
            height=180,
            color=QColor(150, 100, 150),
        )

        # Add output port
        self.out_port = self.add_output_port("Out")

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Noise type selector
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(
            ["White", "Pink", "Brown", "Blue", "Grey", "Velvet", "Sample & Hold"]
        )
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        type_layout.addWidget(self.type_combo)
        layout.addLayout(type_layout)

        # Amplitude control
        self.amp_knob = Knob("Amplitude", 0.0, 1.0, 0.5)
        self.amp_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("amplitude", self.amp_knob.get_value())
        )
        layout.addWidget(self.amp_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter(
            "noise_type", self.type_combo, getter="currentText", setter="setCurrentText"
        )
        self.register_parameter("amplitude", self.amp_knob)

        self.component = self.create_component()

    # AudioModuleInterface implementation
    @property
    def module_title(self) -> str:
        """Return the module title."""
        return "Noise"

    @property
    def module_description(self) -> str:
        """Return module description."""
        return (
            "Multi-type noise generator (White, Pink, Brown, Blue, Grey, Velvet, S&H)"
        )

    @property
    def module_category(self) -> ModuleCategory:
        """Return SOURCE since noise generators generate audio."""
        return ModuleCategory.SOURCE

    def _on_type_changed(self, noise_type: str):
        """Handle noise type change."""
        self.component = self.create_component()
        self.parameter_changed.emit("noise_type", noise_type)

    def create_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Create the noise component."""
        noise_type = self.type_combo.currentText()
        amp = self.amp_knob.get_value()

        # Create noise generator with current settings
        return NoiseGenerator(noise_type=noise_type, amplitude=amp)
