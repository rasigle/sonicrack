"""Output module UI with sample rate and buffer size controls."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout
from soniclab.audio_io import AudioOutput
from soniclab.dsp.modifiers.amplitude import Volume

from sonicrack.config.audio_config import audio_config
from sonicrack.gui.widgets import Knob, LevelMeter
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module

if TYPE_CHECKING:
    from sonicrack.patching.port import Port
    from sonicrack.runtime.engine import AudioEngine

logger = logging.getLogger(__name__)


@register_module()
class OutputModule(ModuleWidget):
    """Audio output sink with configurable sample rate and buffer size.

    Output is a passive runtime sink. Its audio callback requests rendered input
    buffers from ``AudioEngine.render_ports()`` and never pulls upstream modules
    directly through ``Port.read()``. That keeps playback, visualizers, and
    future writer sinks on one shared render path.
    """

    runtime_kind = "passive_sink"

    metadata = ModuleMetadata(
        title="Output",
        category=ModuleCategory.OUTPUT,
        description="Audio output",
    )

    def __init__(self):
        """Initialize output module."""
        super().__init__(
            width=230,
            height=200,
            color=QColor(200, 80, 80),
        )

        # Add input ports
        self.inp_port_l: Port = self.add_input("Left/Mono")
        self.inp_port_r: Port = self.add_input("Right")

        # Create audio output handler using global settings
        self.audio_output = AudioOutput(
            sample_rate=audio_config.sample_rate,
            buffer_size=audio_config.buffer_size,
            audio_callback=self._generate_samples,  # Callback pulls samples
        )
        self.audio_engine: AudioEngine | None = None

        # Register for global audio setting changes
        self._install_sample_rate_listener()
        audio_config.add_buffer_size_listener(self._on_global_buffer_size_changed)

        # Build UI
        layout = self._begin_controls()

        # Master gain control
        self.gain_db = 0.0  # Initialize at 0 dB

        # Create Volume component for smooth, click-free gain control
        self.volume_component = Volume(
            gain_db=0.0,
            sample_rate=audio_config.sample_rate,
            smoothing_time_ms=10.0,  # 10ms smoothing prevents clicks
        )

        self.master_gain_knob = Knob(
            label="Master",
            description="Controls the master output gain",
            min_value=-80.0,
            max_value=12.0,
            default_value=0.0,
            callback=self._on_gain_changed,
        )

        meter_col = QVBoxLayout()
        meter_col.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        meter_labels = QHBoxLayout()
        meter_labels.setSpacing(8)
        left_lbl = QLabel("L")
        left_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_lbl = QLabel("R")
        right_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        meter_labels.addWidget(left_lbl)
        meter_labels.addWidget(right_lbl)
        meter_col.addLayout(meter_labels)
        meters = QHBoxLayout()
        meters.setSpacing(8)
        self.left_meter = LevelMeter(width=12, height=52)
        self.right_meter = LevelMeter(width=12, height=52)
        meters.addWidget(self.left_meter)
        meters.addWidget(self.right_meter)
        meter_col.addLayout(meters)

        row = QHBoxLayout()
        row.addWidget(self.master_gain_knob)
        row.addLayout(meter_col)
        layout.addLayout(row)

        # Status label only; sample/buffer configured globally
        self.status_label = QtWidgets.QLabel("Stopped")
        self.status_label.setStyleSheet(
            "color: #888; font-size: 10px; font-style: italic;"
        )
        layout.addWidget(self.status_label)
        layout.addStretch()

        self._peak_l = 0.0
        self._peak_r = 0.0
        self._meter_timer = QTimer(self)
        self._meter_timer.setTimerType(Qt.TimerType.CoarseTimer)
        self._meter_timer.setInterval(40)
        self._meter_timer.timeout.connect(self._refresh_meters)
        self._meter_timer.start()

        self._finish_controls(layout)

        logger.debug("OutputModule initialized")

    def _on_global_sample_rate_changed(self, sample_rate: int):
        """Sync audio output with global sample rate changes."""
        logger.info(f"Output module updating sample rate to {sample_rate} Hz")
        self.audio_output.set_sample_rate(sample_rate)
        # Update Volume component sample rate for correct smoothing duration
        self.volume_component.sample_rate = sample_rate

    def _on_global_buffer_size_changed(self, buffer_size: int):
        """Sync audio output with global buffer size changes."""
        logger.info(f"Output module updating buffer size to {buffer_size} samples")
        self.audio_output.set_buffer_size(buffer_size)

    def _on_gain_changed(self, value: float):
        """Update master gain when knob changes.

        Uses Volume component which provides smooth gain transitions
        to prevent clicking/popping artifacts.
        """
        self.gain_db = value
        self.volume_component.gain_db = value
        logger.debug(f"Master gain changed to {value:.1f} dB")

    def get_output_component(self, port_name: str):
        """This module has no output component (it's a sink)."""
        return None

    def set_active(self, active: bool) -> None:
        """Toggle output playback with the module power state."""
        was_active = self.is_active
        super().set_active(active)
        if was_active and not active:
            self.stop_playback()

    def start_playback(self) -> bool:
        """Start audio playback through the engine-owned render graph.

        Also stops playback if there are no connections.

        Returns:
            True if playback is running or was started, False when this module
            intentionally stays stopped.
        """
        logger.info("OutputModule.start_playback() called")
        logger.info(
            "  input connections: left=%s right=%s",
            self.inp_port_l.is_connected,
            self.inp_port_r.is_connected,
        )

        if not self.is_active:
            logger.info("Output module is inactive; playback will not start")
            if self.audio_output.is_playing:
                self.stop_playback()
            self.status_label.setText("Off")
            self.status_label.setStyleSheet(
                "color: #888; font-size: 10px; font-style: italic;"
            )
            return False

        # Check if we have any connections
        if not self.inp_port_l.is_connected and not self.inp_port_r.is_connected:
            logger.warning("Output module has no connections")

            # Stop playback if it's running
            if self.audio_output.is_playing:
                logger.info("Stopping playback - no connections")
                self.stop_playback()

            self.status_label.setText("No input")
            self.status_label.setStyleSheet(
                "color: #f80; font-size: 10px; font-style: italic;"
            )
            return False

        # We have connections - start playback if not already playing
        if not self.audio_output.is_playing:
            logger.info("Starting playback using engine-owned render graph")
            self.audio_output.start_playback()
            self.status_label.setText("Playing")
            self.status_label.setStyleSheet(
                "color: #4f4; font-size: 10px; font-weight: bold;"
            )

        return True

    def stop_playback(self, graceful: bool = True):
        """Stop audio playback.

        Args:
            graceful: If True, perform the audio output fade-out before stopping.
        """
        if self.audio_output.is_playing:
            self.audio_output.stop_playback(graceful=graceful)
            self.status_label.setText("Stopped")
            self.status_label.setStyleSheet(
                "color: #888; font-size: 10px; font-style: italic;"
            )

    def shutdown(self, graceful: bool = True):
        """Release audio resources before application shutdown."""
        logger.debug("OutputModule shutdown requested")
        if hasattr(self, "audio_output") and self.audio_output:
            self.audio_output.cleanup(graceful=graceful)

        audio_config.remove_sample_rate_listener(self._on_global_sample_rate_changed)
        audio_config.remove_buffer_size_listener(self._on_global_buffer_size_changed)

        self.status_label.setText("Stopped")
        self.status_label.setStyleSheet(
            "color: #888; font-size: 10px; font-style: italic;"
        )

    def closeEvent(self, event):
        """Handle module close/deletion - ensure audio stops first."""
        logger.debug("OutputModule closing - stopping audio")

        # Stop audio output before Qt deletes the module
        try:
            self.shutdown(graceful=True)
        except Exception as e:
            logger.debug(f"Error stopping audio on close: {e}")
        super().closeEvent(event)

    @staticmethod
    def _copy_samples_to_channel(
        samples,
        out: np.ndarray,
        channel: int,
        num_frames: int,
    ) -> bool:
        if samples is None:
            return False

        x = np.asarray(samples)
        if x.size == 0:
            return False

        if x.ndim == 1:
            n = min(x.shape[0], num_frames)
            out[:n, channel] = x[:n]
            return n > 0

        if x.ndim == 2 and x.shape[1] == 2:
            n = min(x.shape[0], num_frames)
            out[:n, channel] = x[:n, channel]
            return n > 0

        if x.ndim == 2 and x.shape[1] == 1:
            n = min(x.shape[0], num_frames)
            out[:n, channel] = x[:n, 0]
            return n > 0

        if x.ndim == 2 and x.shape[0] == 2:
            n = min(x.shape[1], num_frames)
            out[:n, channel] = x[channel, :n]
            return n > 0

        x = x.ravel()
        n = min(x.shape[0], num_frames)
        out[:n, channel] = x[:n]
        return n > 0

    @staticmethod
    def _copy_samples_to_stereo(
        samples,
        out: np.ndarray,
        num_frames: int,
    ) -> bool:
        if samples is None:
            return False

        x = np.asarray(samples)
        if x.size == 0:
            return False

        if x.ndim == 0:
            out[:1, 0] = x
            out[:1, 1] = x
            return True

        if x.ndim == 1:
            n = min(x.shape[0], num_frames)
            out[:n, 0] = x[:n]
            out[:n, 1] = x[:n]
            return n > 0

        if x.ndim == 2 and x.shape[1] == 2:
            n = min(x.shape[0], num_frames)
            out[:n, :2] = x[:n, :2]
            return n > 0

        if x.ndim == 2 and x.shape[1] == 1:
            n = min(x.shape[0], num_frames)
            mono = x[:n, 0]
            out[:n, 0] = mono
            out[:n, 1] = mono
            return n > 0

        if x.ndim == 2 and x.shape[0] == 2:
            n = min(x.shape[1], num_frames)
            out[:n, 0] = x[0, :n]
            out[:n, 1] = x[1, :n]
            return n > 0

        x = x.ravel()
        n = min(x.shape[0], num_frames)
        out[:n, 0] = x[:n]
        out[:n, 1] = x[:n]
        return n > 0

    def render_offline(self, num_frames: int) -> np.ndarray:
        """Render one stereo block for File → Export Audio (no live callback)."""
        return self._generate_samples(num_frames)

    def _generate_samples(self, num_frames: int) -> np.ndarray:
        """Generate stereo audio samples through the engine-owned graph.

        Routing behavior:
        - Only L connected: mono L is duplicated; stereo L passes through
        - Only R connected: mono R is duplicated; stereo R passes through
        - Both L+R connected: L to left channel, R to right channel
        - Neither connected: Return silence

        Args:
            num_frames: Number of samples requested by audio callback

        Returns:
            Stereo audio samples as (N, 2) numpy array
        """
        out = np.zeros((num_frames, 2), dtype=np.float32)

        if not self.is_active:
            return out

        l_connected = self.inp_port_l.is_connected
        r_connected = self.inp_port_r.is_connected

        if not l_connected and not r_connected:
            return out

        engine = self.audio_engine
        if engine is None:
            logger.error("Output module has no AudioEngine; returning silence")
            return out

        if l_connected and r_connected:
            left, right = engine.render_ports(
                [self.inp_port_l, self.inp_port_r],
                num_frames,
            )
            self._copy_samples_to_channel(left, out, 0, num_frames)
            self._copy_samples_to_channel(right, out, 1, num_frames)
        elif l_connected:
            left = engine.render_ports([self.inp_port_l], num_frames)[0]
            self._copy_samples_to_stereo(left, out, num_frames)
        else:
            right = engine.render_ports([self.inp_port_r], num_frames)[0]
            self._copy_samples_to_stereo(right, out, num_frames)

        # Apply smooth, click-free gain using Volume component
        # The Volume component handles gain smoothing internally to prevent clicks
        out = self.volume_component(out)
        self._capture_peaks(out)
        return out

    def _capture_peaks(self, stereo: np.ndarray) -> None:
        """Store peak levels for the UI meter timer (audio-thread safe floats)."""
        if stereo.size == 0:
            return
        block = np.asarray(stereo)
        if block.ndim == 1:
            peak = float(np.max(np.abs(block)))
            self._peak_l = max(self._peak_l, peak)
            self._peak_r = max(self._peak_r, peak)
            return
        self._peak_l = max(self._peak_l, float(np.max(np.abs(block[:, 0]))))
        if block.shape[1] > 1:
            self._peak_r = max(self._peak_r, float(np.max(np.abs(block[:, 1]))))
        else:
            self._peak_r = self._peak_l

    def _refresh_meters(self) -> None:
        self.left_meter.set_level(min(1.0, self._peak_l))
        self.right_meter.set_level(min(1.0, self._peak_r))
        self.left_meter.decay()
        self.right_meter.decay()
        self._peak_l *= 0.7
        self._peak_r *= 0.7
