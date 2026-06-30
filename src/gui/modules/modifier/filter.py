"""Filter module for the modular synthesizer GUI."""

from __future__ import annotations

from typing import Any, Literal

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)

from src.engine.filter import ButterworthFilter
from src.gui.audio_config import audio_config
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.runtime import RuntimeParameters
from src.gui.core.runtime_helpers import (
    float_parameter,
    read_samples,
    silence,
    str_parameter,
    write_output,
)
from src.gui.module_registry import register_module
from src.gui.widgets import HSlider, Knob
from src.gui.widgets.module_widget import ModuleWidget


@register_module()
class FilterModule(ModuleWidget):
    """Butterworth filter module with frequency, order, and type controls."""

    runtime_kind = "component_modifier"

    metadata = ModuleMetadata(
        title="Filter",
        category=ModuleCategory.MODIFIER,
        description="Butterworth filter (low-pass, high-pass, band-pass)",
    )

    def __init__(self):
        """Initialize filter module."""
        super().__init__(
            width=220,
            height=285,
            color=QColor(100, 180, 140),  # Greenish color for filter
        )

        # Add ports
        self.in_port = self.add_input("In")
        self.out_port = self.add_output("Out")

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Filter type selector
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Low-pass", "High-pass", "Band-pass"])
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        type_layout.addWidget(self.type_combo)
        layout.addLayout(type_layout)

        # Cutoff frequency knobs (horizontal layout for low and high cutoff)
        freq_container_layout = QHBoxLayout()

        # Low cutoff knob (always visible)
        low_freq_layout = QVBoxLayout()
        self.low_freq_label = QLabel("Low Cutoff (Hz)")
        self.low_freq_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cutoff_knob = Knob(
            label="",
            min_value=20,
            max_value=20000,
            default_value=1000,
            logarithmic=True,  # Logarithmic scale for frequency
        )
        self.cutoff_knob.value_changed.connect(self._on_cutoff_changed)
        self.cutoff_value_label = QLabel("1000 Hz")
        self.cutoff_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        low_freq_layout.addWidget(self.low_freq_label)
        low_freq_layout.addWidget(self.cutoff_knob)
        low_freq_layout.addWidget(self.cutoff_value_label)
        freq_container_layout.addLayout(low_freq_layout)

        # High cutoff knob (for band-pass, initially hidden)
        self.high_freq_layout = QVBoxLayout()
        self.high_freq_label = QLabel("High Cutoff (Hz)")
        self.high_freq_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.high_cutoff_knob = Knob(
            label="",
            min_value=20,
            max_value=20000,
            default_value=2000,
            logarithmic=True,
        )
        self.high_cutoff_knob.value_changed.connect(self._on_high_cutoff_changed)
        self.high_cutoff_value_label = QLabel("2000 Hz")
        self.high_cutoff_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.high_freq_layout.addWidget(self.high_freq_label)
        self.high_freq_layout.addWidget(self.high_cutoff_knob)
        self.high_freq_layout.addWidget(self.high_cutoff_value_label)
        freq_container_layout.addLayout(self.high_freq_layout)

        layout.addLayout(freq_container_layout)

        # Hide high cutoff initially (only for band-pass)
        self._set_high_cutoff_visible(False)

        # Filter order slider
        order_layout = QVBoxLayout()
        order_label = QLabel("Order")
        order_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.order_slider = HSlider(
            label="",
            min_value=1,
            max_value=10,
            default_value=4,
        )
        self.order_slider.value_changed.connect(self._on_order_changed)
        self.order_value_label = QLabel("4")
        self.order_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        order_layout.addWidget(order_label)
        order_layout.addWidget(self.order_slider)
        order_layout.addWidget(self.order_value_label)
        layout.addLayout(order_layout)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters
        self.register_parameter("cutoff", self.cutoff_knob)
        self.register_parameter("high_cutoff", self.high_cutoff_knob)
        self.register_parameter("order", self.order_slider)
        self.register_parameter(
            "filter_type",
            self.type_combo,
            getter="currentText",
            setter="setCurrentText",
        )

        self.component = self.create_engine_component()
        self._runtime_filter_params: (
            tuple[float | tuple[float, float], int, Literal["low", "high", "band"]]
            | None
        ) = None
        self._sample_rate_listener = self._on_global_sample_rate_changed
        audio_config.add_sample_rate_listener(self._sample_rate_listener)
        self.destroyed.connect(self._cleanup_audio_config_listeners)

    def _set_high_cutoff_visible(self, visible: bool):
        """Show or hide the high cutoff controls."""
        for i in range(self.high_freq_layout.count()):
            widget = self.high_freq_layout.itemAt(i).widget()
            if widget:
                widget.setVisible(visible)

        # Update low cutoff label based on mode
        if visible:
            # Band-pass mode: show "Low Cutoff"
            self.low_freq_label.setText("Low Cutoff (Hz)")
        else:
            # Single cutoff mode: show "Cutoff"
            self.low_freq_label.setText("Cutoff (Hz)")

    def _on_type_changed(self, text: str):
        """Handle filter type change."""
        # Map display name to internal name
        type_map = {
            "Low-pass": "low",
            "High-pass": "high",
            "Band-pass": "band",
        }
        filter_type = type_map.get(text, "low")

        # Show/hide high cutoff for band-pass
        is_bandpass = filter_type == "band"
        self._set_high_cutoff_visible(is_bandpass)

        # Update component and emit signal
        self.component = self.create_engine_component()
        self.parameter_changed.emit("filter_type", filter_type)

    def _on_cutoff_changed(self):
        """Handle cutoff frequency change."""
        value = self.cutoff_knob.get_value()
        self.cutoff_value_label.setText(f"{int(value)} Hz")
        self.component = self.create_engine_component()
        self.parameter_changed.emit("cutoff", value)

    def _on_high_cutoff_changed(self):
        """Handle high cutoff frequency change (band-pass only)."""
        value = self.high_cutoff_knob.get_value()
        self.high_cutoff_value_label.setText(f"{int(value)} Hz")
        self.component = self.create_engine_component()
        self.parameter_changed.emit("high_cutoff", value)

    def _on_order_changed(self, value: float):
        """Handle filter order change."""
        # Convert to int since filter order must be an integer
        order_int = int(value)
        self.order_value_label.setText(str(order_int))
        self.component = self.create_engine_component()
        self.parameter_changed.emit("order", order_int)

    def _on_global_sample_rate_changed(self, new_sample_rate: int):
        """Redesign filter coefficients when the global sample rate changes."""
        _ = new_sample_rate
        self.component = self.create_engine_component()

    def _cleanup_audio_config_listeners(self, *_args):
        """Remove registered global listeners during Qt object teardown."""
        audio_config.remove_sample_rate_listener(self._sample_rate_listener)

    def _get_filter_type(self) -> Literal["low", "high", "band"]:
        """Map UI filter type text to engine filter type."""
        return self._normalize_filter_type(self.type_combo.currentText())

    @staticmethod
    def _normalize_filter_type(text: str) -> Literal["low", "high", "band"]:
        """Map UI or engine filter type text to the engine filter type."""
        type_map: dict[str, Literal["low", "high", "band"]] = {
            "Low-pass": "low",
            "High-pass": "high",
            "Band-pass": "band",
            "low": "low",
            "high": "high",
            "band": "band",
        }
        return type_map.get(text, "low")

    def _get_cutoff_param(self) -> float | tuple[float, float]:
        """Get the current cutoff parameter in engine format."""
        cutoff = self.cutoff_knob.get_value()
        high_cutoff = self.high_cutoff_knob.get_value()
        filter_type = self._get_filter_type()

        return self._build_cutoff_param(cutoff, high_cutoff, filter_type)

    def _build_cutoff_param(
        self,
        cutoff: float,
        high_cutoff: float,
        filter_type: Literal["low", "high", "band"],
    ) -> float | tuple[float, float]:
        """Build a valid engine cutoff parameter for the selected filter type."""
        if filter_type == "band":
            low = min(cutoff, high_cutoff)
            high = max(cutoff, high_cutoff)

            if low == high:
                if high < self.high_cutoff_knob.max_value:
                    high += 1.0
                else:
                    low = max(self.cutoff_knob.min_value, low - 1.0)

            return low, high

        return cutoff

    def get_required_inputs(self) -> list[str]:
        """Return list of required input port names.

        Filter module requires one audio input to process.

        Returns:
            List containing the name of the required input port
        """
        return ["In"]

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ) -> ButterworthFilter:
        """Create the filter modifier component.

        The patch compiler wraps modifier components in a ``Chain`` itself, so this
        method must only return the modifier and not a composed signal path.
        """
        del input_components, modulation_components

        return ButterworthFilter(
            cutoff=self._get_cutoff_param(),
            order=int(self.order_slider.get_value()),
            filter_type=self._get_filter_type(),
            sample_rate=audio_config.sample_rate,
        )

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        """Filter the connected input for one render cycle."""
        if not self.in_port.is_connected:
            write_output(self.out_port, silence(num_samples), num_samples)
            return

        filter_type = self._normalize_filter_type(
            str_parameter(parameters, "filter_type", self.type_combo.currentText)
        )
        cutoff = self._build_cutoff_param(
            float_parameter(parameters, "cutoff", self.cutoff_knob.get_value),
            float_parameter(parameters, "high_cutoff", self.high_cutoff_knob.get_value),
            filter_type,
        )
        filter_params = (
            cutoff,
            int(float_parameter(parameters, "order", self.order_slider.get_value)),
            filter_type,
        )

        if self.component is None or filter_params != self._runtime_filter_params:
            self.component = ButterworthFilter(
                cutoff=filter_params[0],
                order=filter_params[1],
                filter_type=filter_params[2],
                sample_rate=audio_config.sample_rate,
            )
            self._runtime_filter_params = filter_params

        write_output(
            self.out_port,
            self.component(read_samples(self.in_port, num_samples)),
            num_samples,
        )
