"""Output module UI with sample rate and buffer size controls."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtGui import QColor

from src.engine.io.audio_output import AudioOutput
from src.gui.audio_config import audio_config
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.module_registry import register_module
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget

if TYPE_CHECKING:
    from src.gui.core.port import Port

logger = logging.getLogger(__name__)


@register_module()
class OutputModule(ModuleWidget):
    """Audio output module with configurable sample rate and buffer size."""

    metadata = ModuleMetadata(
        title="Output",
        category=ModuleCategory.OUTPUT,
        description="Audio output",
    )

    def __init__(self):
        """Initialize output module."""
        super().__init__(
            width=220,
            height=260,
            color=QColor(200, 80, 80),
        )

        # Add input port
        self.input_port: Port = self.add_input("In")

        # Create audio output handler using global settings
        self.audio_output = AudioOutput(
            sample_rate=audio_config.sample_rate,
            buffer_size=audio_config.buffer_size,
            audio_callback=None,
        )

        # Register for global audio setting changes
        audio_config.add_sample_rate_listener(self._on_global_sample_rate_changed)
        audio_config.add_buffer_size_listener(self._on_global_buffer_size_changed)

        # Build UI
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Master gain control
        self.gain_db = 0.0  # Initialize at 0 dB
        self.gain_knob = Knob(
            label="Master Gain",
            min_value=-80.0,  # Effectively -infinity
            max_value=12.0,
            default_value=0.0,
            callback=self._on_gain_changed,
        )
        layout.addWidget(self.gain_knob)  # Add knob to layout

        # Status label only; sample/buffer configured globally
        self.status_label = QtWidgets.QLabel("Stopped")
        self.status_label.setStyleSheet(
            "color: #888; font-size: 10px; font-style: italic;"
        )
        layout.addWidget(self.status_label)
        layout.addStretch()

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Input component reference
        self._input_component = None

        # Reference to audio engine (set by main window)
        self.audio_engine = None

        logger.debug("OutputModule initialized")

    def _on_global_sample_rate_changed(self, sample_rate: int):
        """Sync audio output with global sample rate changes."""
        logger.info(f"Output module updating sample rate to {sample_rate} Hz")
        self.audio_output.set_sample_rate(sample_rate)

    def _on_global_buffer_size_changed(self, buffer_size: int):
        """Sync audio output with global buffer size changes."""
        logger.info(f"Output module updating buffer size to {buffer_size} samples")
        self.audio_output.set_buffer_size(buffer_size)

    def _on_gain_changed(self, value: float):
        """Update master gain when knob changes."""
        self.gain_db = value
        logger.debug(f"Master gain changed to {value:.1f} dB")

    def _generate_audio(self, num_samples: int) -> np.ndarray | None:
        """Generate audio samples from connected input using process-based approach.

        This uses the audio_engine to coordinate module processing:
        1. Audio engine processes all modules in topological order
        2. Each module processes its inputs and writes to its output ports
        3. Output module reads from its input port

        Args:
            num_samples: Number of samples to generate

        Returns:
            Stereo audio array or None
        """
        if not self.input_port.is_connected:
            return None

        return np.asarray(self.input_port.read())

    def get_output_component(self, port_name: str):
        """This module has no output component (it's a sink)."""
        return None


    def start_playback(self):
        """Start audio playback using process-based architecture.

        Also stops playback if there are no connections.
        """
        # Debug: Check connection status
        logger.info(f"OutputModule.start_playback() called")
        logger.info(f"  input_port.is_connected = {self.input_port.is_connected}")
        logger.info(f"  input_port.connected_to = {self.input_port.connected_to}")
        logger.info(f"  Number of connections: {len(self.input_port.connected_to)}")

        # Check if we have any connections
        if not self.input_port.is_connected:
            logger.warning("Output module has no connections")

            # Stop playback if it's running
            if self.audio_output.is_playing:
                logger.info("Stopping playback - no connections")
                self.stop_playback()

            self.status_label.setText("No input")
            self.status_label.setStyleSheet(
                "color: #f80; font-size: 10px; font-style: italic;"
            )
            return

        # We have connections - start playback if not already playing
        if not self.audio_output.is_playing:
            logger.info("Starting playback using PROCESS-BASED architecture")
            self.audio_output.start_playback()
            self.status_label.setText("Playing")
            self.status_label.setStyleSheet(
                "color: #4f4; font-size: 10px; font-weight: bold;"
            )

    def stop_playback(self):
        """Stop audio playback."""
        if self.audio_output.is_playing:
            self.audio_output.stop_playback()
            self.status_label.setText("Stopped")
            self.status_label.setStyleSheet(
                "color: #888; font-size: 10px; font-style: italic;"
            )

    def cleanup(self):
        """Clean up audio resources."""
        audio_config.remove_sample_rate_listener(self._on_global_sample_rate_changed)
        audio_config.remove_buffer_size_listener(self._on_global_buffer_size_changed)
        self.audio_output.cleanup()

    def process(self, num_samples: int):
        """Push audio from the input port into AudioOutput with proper shape."""
        if not self.input_port.is_connected:
            return

        samples = self.input_port.read()
        if samples is None:
            return

        samples = np.asarray(samples)
        if samples.size == 0:
            return

        # Apply master gain (convert dB to linear)
        if self.gain_db <= -80.0:
            # Treat -80 dB as silence (effectively -infinity)
            return
        elif self.gain_db != 0.0:
            linear_gain = 10.0 ** (self.gain_db / 20.0)
            samples = samples * linear_gain

        self.audio_output.write(_ensure_stereo(samples).astype(np.float32))


def _ensure_stereo(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 0:
        val = float(samples)
        samples = np.array([[val, val]])

    elif samples.ndim == 1:
        if samples.size == 1:
            val = float(samples[0])
            samples = np.array([[val, val]])
        else:
            samples = np.column_stack((samples, samples))

    elif samples.ndim == 2:
        rows, cols = samples.shape
        if cols == 2:
            pass
        elif rows == 2 and cols != 2:
            samples = samples.T
        elif cols == 1:
            samples = np.repeat(samples, 2, axis=1)
        elif rows == 1:
            samples = np.repeat(samples.T, 2, axis=1)
        else:
            mean_vals = samples.mean(axis=1)
            samples = np.column_stack((mean_vals, mean_vals))
    else:
        flat = samples.reshape(-1)
        if flat.size == 1:
            val = float(flat[0])
            samples = np.array([[val, val]])
        else:
            if flat.size % 2 != 0:
                flat = np.pad(flat, (0, 1), mode="constant")
            samples = flat.reshape(-1, 2)
    return samples
