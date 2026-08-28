"""Four-channel mixer with gain, pan, mute, and peak meters."""

from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout
from soniclab import Chain, Volume, WaveAdder

from sonicrack.gui.widgets import Knob, LevelMeter
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import (
    as_mono,
    bool_parameter,
    constant_power_pan,
    float_parameter,
    read_samples,
    silence,
)
from sonicrack.runtime.specs import RuntimeParameters

logger = logging.getLogger(__name__)

DEFAULT_CHANNEL_VOLUME = 0.7
CHANNEL_COUNT = 4


@register_module()
class MixerModule(ModuleWidget):
    """Four-channel stereo mixer with per-channel gain, pan, mute, and meters."""

    runtime_kind = "mixer"

    metadata = ModuleMetadata(
        title="Mixer",
        category=ModuleCategory.MIXER,
        description="4-channel mixer with pan, mute, and peak meters",
    )

    def __init__(self) -> None:
        super().__init__(width=280, height=340, color=QColor(90, 150, 105))

        self.in1_port = self.add_input("In 1")
        self.in2_port = self.add_input("In 2")
        self.in3_port = self.add_input("In 3")
        self.in4_port = self.add_input("In 4")
        self.out_port = self.add_output("Out")

        layout = self._begin_controls(spacing=4)
        channels = QHBoxLayout()
        channels.setSpacing(4)

        self.gain_knobs: list[Knob] = []
        self.pan_knobs: list[Knob] = []
        self.mute_buttons: list[QPushButton] = []
        self.meters: list[LevelMeter] = []
        self._peaks = [0.0] * CHANNEL_COUNT
        self._volume_components = [
            Volume(amplitude=DEFAULT_CHANNEL_VOLUME) for _ in range(CHANNEL_COUNT)
        ]

        for index in range(CHANNEL_COUNT):
            column = QVBoxLayout()
            column.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            column.setSpacing(3)

            header = QLabel(f"{index + 1}")
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            column.addWidget(header)

            mute = QPushButton("M")
            mute.setCheckable(True)
            mute.setFixedWidth(28)
            mute.setToolTip(f"Mute channel {index + 1}")
            mute.toggled.connect(
                lambda checked, i=index: self.parameter_changed.emit(
                    f"mute{i + 1}", checked
                )
            )
            column.addWidget(mute, alignment=Qt.AlignmentFlag.AlignHCenter)
            self.mute_buttons.append(mute)

            meter = LevelMeter(width=10, height=44)
            column.addWidget(meter, alignment=Qt.AlignmentFlag.AlignHCenter)
            self.meters.append(meter)

            gain = Knob(
                label="Gain",
                description=f"Channel {index + 1} gain",
                min_value=0.0,
                max_value=1.0,
                default_value=DEFAULT_CHANNEL_VOLUME,
            )
            gain.value_changed.connect(lambda _value, i=index: self._on_gain_changed(i))
            column.addWidget(gain)
            self.gain_knobs.append(gain)

            pan = Knob(
                label="Pan",
                description=f"Channel {index + 1} pan",
                min_value=-1.0,
                max_value=1.0,
                default_value=0.0,
            )
            pan.value_changed.connect(
                lambda _value, i=index, knob=pan: self.parameter_changed.emit(
                    f"pan{i + 1}", knob.get_value()
                )
            )
            column.addWidget(pan)
            self.pan_knobs.append(pan)
            channels.addLayout(column)

        layout.addLayout(channels)
        self._finish_controls(layout)

        self.ch1_gain_knob = self.gain_knobs[0]
        self.ch2_gain_knob = self.gain_knobs[1]
        self.ch3_gain_knob = self.gain_knobs[2]
        self.ch4_gain_knob = self.gain_knobs[3]

        for index in range(CHANNEL_COUNT):
            self.register_parameter(f"gain{index + 1}", self.gain_knobs[index])
            self.register_parameter(f"pan{index + 1}", self.pan_knobs[index])
            self.register_parameter(
                f"mute{index + 1}",
                self.mute_buttons[index],
                getter="isChecked",
                setter="setChecked",
            )

        self._meter_timer = QTimer(self)
        self._meter_timer.setTimerType(Qt.TimerType.CoarseTimer)
        self._meter_timer.setInterval(40)
        self._meter_timer.timeout.connect(self._refresh_meters)
        self._meter_timer.start()

        logger.debug("Mixer initialized with 4 channels")

    def _on_gain_changed(self, channel_index: int) -> None:
        new_gain = self.gain_knobs[channel_index].get_value()
        self._volume_components[channel_index].amplitude = new_gain
        self.parameter_changed.emit(f"gain{channel_index + 1}", new_gain)

    def create_engine_component(
        self,
        input_components=None,
        modulation_components=None,
    ):
        """Sum connected inputs with per-channel gain (legacy compiler path)."""
        del modulation_components
        if not input_components:
            return None

        processed_inputs = []
        for channel_idx, input_component in enumerate(input_components):
            if input_component is None:
                continue
            gain = DEFAULT_CHANNEL_VOLUME
            if channel_idx < len(self.gain_knobs):
                gain = self.gain_knobs[channel_idx].get_value()
            processed_inputs.append(Chain(input_component, Volume(amplitude=gain)))

        if not processed_inputs:
            return None
        return WaveAdder(*processed_inputs, mix_mode="sum")

    def _refresh_meters(self) -> None:
        for meter, peak in zip(self.meters, self._peaks, strict=True):
            meter.set_level(peak)
            meter.decay()
            # Let the DSP-held peak fall so a silent channel decays.
        self._peaks = [p * 0.7 for p in self._peaks]

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        mixed_left = None
        mixed_right = None
        mixed_mono = None
        any_panned = False
        input_ports = [self.in1_port, self.in2_port, self.in3_port, self.in4_port]

        for channel_idx, port in enumerate(input_ports):
            if not port.is_connected:
                self._peaks[channel_idx] = 0.0
                continue
            muted = bool_parameter(
                parameters,
                f"mute{channel_idx + 1}",
                self.mute_buttons[channel_idx].isChecked,
            )
            if muted:
                self._peaks[channel_idx] = 0.0
                continue

            gain = float_parameter(
                parameters,
                f"gain{channel_idx + 1}",
                self.gain_knobs[channel_idx].get_value,
            )
            pan = float_parameter(
                parameters,
                f"pan{channel_idx + 1}",
                self.pan_knobs[channel_idx].get_value,
            )
            self._volume_components[channel_idx].amplitude = gain
            signal = as_mono(read_samples(port, num_samples))
            gained = np.asarray(
                self._volume_components[channel_idx](signal), dtype=np.float32
            )
            peak = float(np.max(np.abs(gained))) if gained.size else 0.0
            self._peaks[channel_idx] = max(self._peaks[channel_idx], min(1.0, peak))

            mixed_mono = gained if mixed_mono is None else mixed_mono + gained
            if abs(pan) > 1e-3:
                any_panned = True
            left_gain, right_gain = constant_power_pan(pan)
            left = gained * np.float32(left_gain)
            right = gained * np.float32(right_gain)
            mixed_left = left if mixed_left is None else mixed_left + left
            mixed_right = right if mixed_right is None else mixed_right + right

        if mixed_mono is None:
            self.out_port.write(silence(num_samples))
            return
        if any_panned and mixed_left is not None and mixed_right is not None:
            self.out_port.write(np.column_stack((mixed_left, mixed_right)))
            return
        self.out_port.write(mixed_mono)
