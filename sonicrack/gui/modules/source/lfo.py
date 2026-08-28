from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout
from soniclab.dsp.modulators import LFO, LFOShape

from sonicrack.config.audio_config import audio_config
from sonicrack.constants import (
    DEFAULT_PW_PERCENTAGE_VALUE,
    MAX_PW_PERCENTAGE_VALUE,
    MIN_PW_PERCENTAGE_VALUE,
)
from sonicrack.gui.widgets import (
    Knob,
    medium_knob_style,
    metal_knob_style,
    small_knob_style,
)
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.port import PortSignal
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import (
    float_parameter,
    gate_transition_indices,
    read_samples,
    str_parameter,
)
from sonicrack.runtime.specs import RuntimeParameters

if TYPE_CHECKING:
    from sonicrack.patching.port import Port


LFO_MIN_RATE = 0.01
LFO_MAX_RATE = 20.0
LFO_DEFAULT_RATE = 1.0
LFO_DEFAULT_DEPTH = 1.0
LFO_DEFAULT_OFFSET = 0.0

# Output shapes backed by soniclab.LFO (order matches self.ports / self.lfos).
_LFO_OUTPUT_SHAPES: tuple[tuple[str, LFOShape], ...] = (
    ("Sine", LFOShape.SINE),
    ("Triangle", LFOShape.TRIANGLE),
    ("Square", LFOShape.SQUARE),
    ("Random", LFOShape.SAMPLE_AND_HOLD),
)


@register_module()
class LFOModule(ModuleWidget):
    """LFO module wrapping :class:`soniclab.dsp.modulators.LFO`.

    Multi-output (sine / triangle / square / sample-and-hold) so each shape can
    be patched independently while sharing rate, depth, polarity, offset, and
    clock reset.
    """

    runtime_kind = "lfo"

    metadata = ModuleMetadata(
        title="LFO",
        category=ModuleCategory.SOURCE,
        description=(
            "Low-frequency modulation source (soniclab LFO): "
            "sine, triangle, square, sample-and-hold"
        ),
    )

    def __init__(self) -> None:
        super().__init__(
            width=200,
            height=230,
            color=QColor(100, 140, 200),
        )

        self.clock_input: Port = self.add_input("Clock", signal=PortSignal.TRIGGER)

        sample_rate = float(audio_config.sample_rate)
        pulsewidth = DEFAULT_PW_PERCENTAGE_VALUE / 100.0

        self.lfos: list[LFO] = []
        self.ports: list[Port] = []
        for port_name, shape in _LFO_OUTPUT_SHAPES:
            lfo = LFO(
                rate_hz=LFO_DEFAULT_RATE,
                depth=LFO_DEFAULT_DEPTH,
                offset=LFO_DEFAULT_OFFSET,
                shape=shape,
                bipolar=True,
                pulse_width=pulsewidth,
                sample_rate=sample_rate,
            )
            self.lfos.append(lfo)
            port = self.add_output(
                port_name,
                component=lfo,
                signal=PortSignal.CONTROL_CV,
            )
            self.ports.append(port)

        # Named outputs used by tests and patch call sites.
        self.sine_port: Port = self.ports[0]
        self.triangle_port: Port = self.ports[1]
        self.square_port: Port = self.ports[2]
        self.random_port: Port = self.ports[3]

        # Convenience aliases used by tests and call sites.
        self._sine_lfo = self.lfos[0]
        self._triangle_lfo = self.lfos[1]
        self._square_lfo = self.lfos[2]
        self._random_lfo = self.lfos[3]

        self._previous_clock = 0.0

        primary_style = metal_knob_style()
        depth_style = medium_knob_style()
        secondary_style = small_knob_style()

        layout = self._begin_controls(spacing=4)

        # --- Primary: Rate + Depth (always used) ---
        primary = QHBoxLayout()
        primary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        primary.setSpacing(2)

        self.freq_knob = Knob(
            label="Rate",
            description="LFO rate in Hz",
            min_value=LFO_MIN_RATE,
            max_value=LFO_MAX_RATE,
            default_value=LFO_DEFAULT_RATE,
            logarithmic=True,
            style=primary_style,
        )
        self.freq_knob.value_changed.connect(self._on_rate_changed)
        primary.addWidget(self.freq_knob)

        self.amount_knob = Knob(
            label="Depth",
            description="Modulation depth (output scale)",
            min_value=0.0,
            max_value=1.0,
            default_value=LFO_DEFAULT_DEPTH,
            style=depth_style,
        )
        self.amount_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("amount", self.amount_knob.get_value())
        )
        primary.addWidget(self.amount_knob)
        layout.addLayout(primary)

        # --- Secondary: Offset (all outs) + Width (square) ---
        secondary = QHBoxLayout()
        secondary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        secondary.setSpacing(2)

        self.offset_knob = Knob(
            label="Offset",
            description="DC offset after depth scaling",
            min_value=-1.0,
            max_value=1.0,
            default_value=LFO_DEFAULT_OFFSET,
            style=secondary_style,
        )
        self.offset_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("offset", self.offset_knob.get_value())
        )
        secondary.addWidget(self.offset_knob)

        self.pulsewidth_knob = Knob(
            label="Width",
            description="Square-wave pulse width (duty cycle)",
            min_value=MIN_PW_PERCENTAGE_VALUE / 100,
            max_value=MAX_PW_PERCENTAGE_VALUE / 100,
            default_value=DEFAULT_PW_PERCENTAGE_VALUE / 100,
            style=secondary_style,
        )
        self.pulsewidth_knob.value_changed.connect(self._on_width_changed)
        secondary.addWidget(self.pulsewidth_knob)
        layout.addLayout(secondary)

        # Push content slightly so knobs sit closer to the port column.
        layout.addStretch(1)

        self._finish_controls(layout)

        # Setup option: right-click module header → Range (bipolar/unipolar)
        self.polarity_param = self.register_menu_choice(
            "polarity",
            "Range",
            ["Bipolar", "Unipolar"],
            "Bipolar",
            tooltip="Bipolar: −1…+1 before depth\nUnipolar: 0…1 before depth",
        )

        # Parameter names keep patch compatibility (frequency / amount / pulsewidth).
        self.register_parameter("frequency", self.freq_knob)
        self.register_parameter("amount", self.amount_knob)
        self.register_parameter("offset", self.offset_knob)
        self.register_parameter("pulsewidth", self.pulsewidth_knob)

        self._install_sample_rate_listener()

    def _on_global_sample_rate_changed(self, new_sample_rate: int) -> None:
        """Propagate global sample-rate changes to every LFO instance."""
        for lfo in self.lfos:
            lfo.sample_rate = float(new_sample_rate)

    def _on_rate_changed(self) -> None:
        self.parameter_changed.emit("frequency", self.freq_knob.get_value())

    def _on_width_changed(self) -> None:
        pulsewidth = self.pulsewidth_knob.get_value()
        for lfo in self.lfos:
            lfo.pulse_width = pulsewidth
        self.parameter_changed.emit("pulsewidth", pulsewidth)

    def _apply_runtime_parameters(self, parameters: RuntimeParameters) -> None:
        rate = float_parameter(parameters, "frequency", self.freq_knob.get_value)
        depth = float_parameter(parameters, "amount", self.amount_knob.get_value)
        offset = float_parameter(parameters, "offset", self.offset_knob.get_value)
        pulsewidth = float_parameter(
            parameters, "pulsewidth", self.pulsewidth_knob.get_value
        )
        polarity = str_parameter(
            parameters, "polarity", self.polarity_param.get_value
        ).lower()
        bipolar = not polarity.startswith("uni")

        for lfo in self.lfos:
            lfo.rate_hz = max(LFO_MIN_RATE, float(rate))
            lfo.depth = max(0.0, float(depth))
            lfo.offset = float(offset)
            lfo.bipolar = bipolar
            lfo.pulse_width = float(pulsewidth)
            lfo.bpm = None

    @staticmethod
    def _render_with_clock_resets(
        lfo: LFO,
        num_samples: int,
        note_on_indices: np.ndarray,
    ) -> np.ndarray:
        """Render ``num_samples``, calling :meth:`LFO.trigger_note_on` on rises."""
        if note_on_indices.size == 0:
            return np.asarray(lfo.get_samples(num_samples), dtype=np.float32)

        output = np.empty(num_samples, dtype=np.float32)
        start = 0
        for index in note_on_indices:
            idx = int(index)
            if idx > start:
                output[start:idx] = lfo.get_samples(idx - start)
            lfo.trigger_note_on()
            start = idx
        if start < num_samples:
            output[start:] = lfo.get_samples(num_samples - start)
        return output

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        """Render each LFO output for the current engine cycle."""
        self._apply_runtime_parameters(parameters)

        clock_signal = (
            read_samples(self.clock_input, num_samples)
            if self.clock_input.is_connected
            else None
        )

        if clock_signal is not None:
            note_ons, _note_offs, final_clock = gate_transition_indices(
                clock_signal, self._previous_clock
            )
            self._previous_clock = final_clock
        else:
            note_ons = np.empty(0, dtype=np.intp)
            self._previous_clock = 0.0

        any_connected = any(port.is_connected for port in self.ports)
        # When nothing is patched (tests / offline), still advance every shape
        # so phases stay locked and outputs are available for inspection.
        force_all = not any_connected

        for port, lfo in zip(self.ports, self.lfos, strict=True):
            samples = self._render_with_clock_resets(lfo, num_samples, note_ons)
            if force_all or port.is_connected:
                port.write(samples)

    def get_cv_output_range(self) -> tuple[float, float]:
        """LFO output range depends on polarity mode.

        Returns the unit range before offset for CV scaling helpers:
        ``(-1.0, 1.0)`` bipolar or ``(0.0, 1.0)`` unipolar.
        """
        polarity = self.polarity_param.get_value().lower()
        if polarity.startswith("uni"):
            return 0.0, 1.0
        return -1.0, 1.0
