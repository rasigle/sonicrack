"""Noise generator module with multiple noise types."""

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel

from src.constants import DEFAULT_GAIN_DB
from src.engine.noise import NoiseGenerator
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.module_registry import register_module
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget


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

    metadata = ModuleMetadata(
        title="Noise",
        category=ModuleCategory.SOURCE,
        description="Multi-type noise generator (White, Pink, Brown, Blue, Grey, "
        "Velvet, Sample & Hold",
    )

    def __init__(self):
        """Initialize noise module."""
        super().__init__(
            width=200,
            height=180,
            color=QColor(150, 100, 150),
        )

        # Add output port
        self.out_port = self.add_output("Out")

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

        # Gain in dB control (alternative to amplitude)
        self.gain_knob = Knob("Gain (dB)", -60, 12, DEFAULT_GAIN_DB, logarithmic=False)
        self.gain_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("gain_db", self.gain_knob.get_value())
        )
        layout.addWidget(self.gain_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter(
            "noise_type", self.type_combo, getter="currentText", setter="setCurrentText"
        )
        self.register_parameter("gain_db", self.gain_knob)

        self.component = self.create_engine_component()

    # AudioModuleInterface implementation
    def _on_type_changed(self, noise_type: str):
        """Handle noise type change."""
        self.component = self.create_engine_component()
        self.parameter_changed.emit("noise_type", noise_type)

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Create the noise component."""
        noise_type = self.type_combo.currentText()
        gain_db = self.gain_knob.get_value()
        return NoiseGenerator(noise_type=noise_type, gain_db=gain_db)

    def process(self, num_samples: int = 1):
        """Generate noise and write to output port.

        Generates noise samples based on the selected type and writes them to the
        output port.

        Args:
            num_samples: Number of samples to generate (default: 1 for per-sample
                processing)

        Note:
            In the current architecture, this method is not actively called during
            playback. The audio engine directly calls get_samples() on the compiled
            AudioComponents. This method exists to satisfy the AudioModule interface.
        """
        if self.out_port.is_connected and self.component:
            samples = self.component.get_samples(num_samples)
            self.out_port.write(samples)
