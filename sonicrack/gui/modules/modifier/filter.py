"""Filter module for the modular synthesizer GUI."""

from __future__ import annotations

from typing import Any, Literal

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout
from soniclab.dsp.filters.butterworth import ButterworthFilter

from sonicrack.config.audio_config import audio_config
from sonicrack.gui.modules.modifier._filter_base import (
    BUTTERWORTH_TYPE_ITEMS,
    FilterModuleBase,
    bind_parameter_knob,
    create_filter_type_combo,
    normalize_filter_type,
)
from sonicrack.gui.widgets import HSlider, Knob
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import float_parameter, read_samples, str_parameter
from sonicrack.runtime.specs import RuntimeParameters


@register_module()
class FilterModule(FilterModuleBase):
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
            height=310,
            color=QColor(100, 180, 140),  # Greenish color for filter
        )

        self._setup_filter_ports()
        layout = self._begin_controls()

        type_layout, self.type_combo = create_filter_type_combo(
            BUTTERWORTH_TYPE_ITEMS,
            on_changed=self._on_type_changed,
        )
        layout.addLayout(type_layout)

        # Cutoff frequency knobs (horizontal layout for low and high cutoff)
        freq_container_layout = QHBoxLayout()

        # Low cutoff knob (always visible)
        low_freq_layout = QVBoxLayout()
        self.low_freq_label = QLabel("Cutoff (Hz)")
        self.low_freq_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cutoff_knob = Knob(
            label="",
            min_value=20,
            max_value=20000,
            default_value=1000,
            logarithmic=True,
        )
        self.cutoff_value_label = QLabel("1000 Hz")
        self.cutoff_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bind_parameter_knob(
            self,
            self.cutoff_knob,
            "cutoff",
            value_label=self.cutoff_value_label,
            format_value=lambda v: f"{int(v)} Hz",
            on_change=lambda _v: self._rebuild_component(),
        )
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
        self.high_cutoff_value_label = QLabel("2000 Hz")
        self.high_cutoff_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bind_parameter_knob(
            self,
            self.high_cutoff_knob,
            "high_cutoff",
            value_label=self.high_cutoff_value_label,
            format_value=lambda v: f"{int(v)} Hz",
            on_change=lambda _v: self._rebuild_component(),
        )
        self.high_freq_layout.addWidget(self.high_freq_label)
        self.high_freq_layout.addWidget(self.high_cutoff_knob)
        self.high_freq_layout.addWidget(self.high_cutoff_value_label)
        freq_container_layout.addLayout(self.high_freq_layout)

        layout.addLayout(freq_container_layout)
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
        self.order_value_label = QLabel("4")
        self.order_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.order_slider.value_changed.connect(self._on_order_changed)
        order_layout.addWidget(order_label)
        order_layout.addWidget(self.order_slider)
        order_layout.addWidget(self.order_value_label)
        layout.addLayout(order_layout)

        self._finish_controls(layout)

        self.register_parameter("cutoff", self.cutoff_knob)
        self.register_parameter("high_cutoff", self.high_cutoff_knob)
        self.register_parameter("order", self.order_slider)
        self._register_filter_type_parameter(self.type_combo)

        self.component = self.create_engine_component()
        self._runtime_filter_params: (
            tuple[float | tuple[float, float], int, Literal["low", "high", "band"]]
            | None
        ) = None
        self._install_sample_rate_listener()

    def _rebuild_component(self) -> None:
        """Recreate the engine component from current UI parameters."""
        self.component = self.create_engine_component()

    def _set_high_cutoff_visible(self, visible: bool) -> None:
        """Show or hide the high cutoff controls."""
        for i in range(self.high_freq_layout.count()):
            widget = self.high_freq_layout.itemAt(i).widget()
            if widget:
                widget.setVisible(visible)

        if visible:
            self.low_freq_label.setText("Low Cutoff (Hz)")
        else:
            self.low_freq_label.setText("Cutoff (Hz)")

    def _on_type_changed(self, text: str) -> None:
        """Handle filter type change."""
        filter_type = normalize_filter_type(text, allow_notch=False)
        self._set_high_cutoff_visible(filter_type == "band")
        self._rebuild_component()
        self.parameter_changed.emit("filter_type", filter_type)

    def _on_order_changed(self, value: float) -> None:
        """Handle filter order change."""
        order_int = int(value)
        self.order_value_label.setText(str(order_int))
        self._rebuild_component()
        self.parameter_changed.emit("order", order_int)

    def _get_filter_type(self) -> Literal["low", "high", "band"]:
        """Map UI filter type text to engine filter type."""
        return normalize_filter_type(  # type: ignore[return-value]
            self.type_combo.currentText(), allow_notch=False
        )

    def _get_cutoff_param(self) -> float | tuple[float, float]:
        """Get the current cutoff parameter in engine format."""
        return self._build_cutoff_param(
            self.cutoff_knob.get_value(),
            self.high_cutoff_knob.get_value(),
            self._get_filter_type(),
        )

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
            self._write_silence(num_samples)
            return

        filter_type = normalize_filter_type(  # type: ignore[assignment]
            str_parameter(parameters, "filter_type", self.type_combo.currentText),
            allow_notch=False,
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

        self.out_port.write(self.component(read_samples(self.in_port, num_samples)))
