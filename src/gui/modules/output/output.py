from typing import Any

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QColor

from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.module_registry import register_module
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget
from src.utils.math import db_to_linear


@register_module()
class OutputModule(ModuleWidget):
    """Output module (sink for audio) with professional dB volume control.

    Master Volume Control:
    - Use gain_db for professional audio control (recommended)
    - Range: -60 dB (very quiet) to +6 dB (boost)
    - Default: -3 dB (safe headroom for mixing)
    - Linear knob for dB values (dB is already logarithmic)

    Audio Settings:
    - Sample Rate: Common rates from 44.1 kHz to 192 kHz
    - Buffer Size: Common sizes from 64 to 2048 samples

    The knob displays dB values but internally converts to linear
    amplitude for the audio engine.
    """

    metadata = ModuleMetadata(
        title="Output",
        category=ModuleCategory.OUTPUT,
        description="Audio output with master volume and audio settings",
    )

    # Special signal for master volume changes (bypasses hot-swapping)
    master_volume_changed = QtCore.pyqtSignal(float)
    # Signals for audio settings changes
    sample_rate_changed = QtCore.pyqtSignal(int)
    buffer_size_changed = QtCore.pyqtSignal(int)

    def __init__(self):
        """Initialize output module."""
        super().__init__(
            width=180,
            height=220,
            color=QColor(200, 80, 80),
        )

        # Add input port
        self.in_port = self.add_input_port("In")

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Sample Rate Selection
        sample_rate_container = QtWidgets.QHBoxLayout()
        sample_rate_label = QtWidgets.QLabel("Sample Rate:")
        sample_rate_label.setStyleSheet("color: white; font-size: 10px;")
        self.sample_rate_combo = QtWidgets.QComboBox()
        # Typical sample rates (in Hz)
        sample_rates = ["44100", "48000", "88200", "96000", "176400", "192000"]
        self.sample_rate_combo.addItems(sample_rates)
        self.sample_rate_combo.setCurrentText("48000")  # Default: 48 kHz
        self.sample_rate_combo.currentTextChanged.connect(self._on_sample_rate_changed)
        sample_rate_container.addWidget(sample_rate_label)
        sample_rate_container.addWidget(self.sample_rate_combo)
        layout.addLayout(sample_rate_container)

        # Buffer Size Selection
        buffer_size_container = QtWidgets.QHBoxLayout()
        buffer_size_label = QtWidgets.QLabel("Buffer Size:")
        buffer_size_label.setStyleSheet("color: white; font-size: 10px;")
        self.buffer_size_combo = QtWidgets.QComboBox()
        # Typical buffer sizes (in samples)
        buffer_sizes = ["64", "128", "256", "512", "1024", "2048"]
        self.buffer_size_combo.addItems(buffer_sizes)
        self.buffer_size_combo.setCurrentText("512")  # Default: 512 samples
        self.buffer_size_combo.currentTextChanged.connect(self._on_buffer_size_changed)
        buffer_size_container.addWidget(buffer_size_label)
        buffer_size_container.addWidget(self.buffer_size_combo)
        layout.addLayout(buffer_size_container)

        # Master volume in dB (professional control)
        # Range: -60 dB to +6 dB, default: -3 dB (safe headroom)
        # Linear knob (dB is already logarithmic scale)
        self.volume_knob = Knob("Master (dB)", -60, 6, -3, logarithmic=False)
        self.volume_knob.value_changed.connect(self._on_volume_changed)
        layout.addWidget(self.volume_knob, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter("master_volume_db", self.volume_knob)

        self.input_component = None

    def _on_volume_changed(self):
        """Handle volume knob changes - convert dB to linear and emit."""
        db_value = self.volume_knob.get_value()
        self.master_volume_changed.emit(db_to_linear(db_value))

    def _on_sample_rate_changed(self, text: str):
        """Handle sample rate combo box changes."""
        try:
            sample_rate = int(text)
            self.sample_rate_changed.emit(sample_rate)
        except ValueError:
            pass  # Invalid value, ignore

    def _on_buffer_size_changed(self, text: str):
        """Handle buffer size combo box changes."""
        try:
            buffer_size = int(text)
            self.buffer_size_changed.emit(buffer_size)
        except ValueError:
            pass  # Invalid value, ignore

    # AudioModuleInterface implementation
    def get_required_inputs(self) -> list[str]:
        """Output requires the In port to be connected."""
        return ["In"]

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Output doesn't create a component, it returns the input component."""
        if input_components and len(input_components) > 0:
            return input_components[0]
        return self.input_component

    def get_master_volume(self) -> float:
        """Get the master volume level in linear scale.

        Returns:
            Linear amplitude (0.0 to 2.0+) converted from dB.
        """
        return db_to_linear(self.volume_knob.get_value())

    def get_sample_rate(self) -> int:
        """Get the selected sample rate in Hz.

        Returns:
            Sample rate in Hz (e.g., 48000)
        """
        return int(self.sample_rate_combo.currentText())

    def get_buffer_size(self) -> int:
        """Get the selected buffer size in samples.

        Returns:
            Buffer size in samples (e.g., 512)
        """
        return int(self.buffer_size_combo.currentText())

