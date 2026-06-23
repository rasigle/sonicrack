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
from src.utils.audio_utils import combine_lr_to_stereo, mono_to_stereo

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

        # Add input ports
        self.inp_port_l: Port = self.add_input("Left/Mono")
        self.inp_port_r: Port = self.add_input("Right")

        # Create audio output handler using global settings
        self.audio_output = AudioOutput(
            sample_rate=audio_config.sample_rate,
            buffer_size=audio_config.buffer_size,
            audio_callback=self._generate_samples,  # Callback pulls samples
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

    def get_output_component(self, port_name: str):
        """This module has no output component (it's a sink)."""
        return None

    def start_playback(self):
        """Start audio playback using process-based architecture.

        Also stops playback if there are no connections.
        """
        # Debug: Check connection status
        logger.info("OutputModule.start_playback() called")
        logger.info(f"  input_port.is_connected = {self.inp_port_l.is_connected}")
        logger.info(f"  input_port.connected_to = {self.inp_port_l.connected_to}")
        logger.info(f"  Number of connections: {len(self.inp_port_l.connected_to)}")

        # Check if we have any connections
        if not self.inp_port_l.is_connected:
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

    def _generate_samples(self, num_samples: int) -> np.ndarray:
        """Generate audio samples by pulling from input ports.

        This is called directly by the audio callback in the real-time audio thread.
        It pulls samples from connected input ports and converts them to stereo.

        Routing behavior:
        - Only L connected: L signal duplicated to both stereo channels
        - Only R connected: R signal duplicated to both stereo channels
        - Both L+R connected: L to left channel, R to right channel
        - Neither connected: Return silence

        Args:
            num_samples: Number of samples requested by audio callback

        Returns:
            Stereo audio samples as (N, 2) numpy array
        """
        # Invalidate all module caches at the start of each audio cycle
        # This ensures all modules regenerate their samples for this cycle
        if hasattr(self, "audio_engine") and self.audio_engine:
            self.audio_engine.invalidate_all_caches()

        # Check which ports are connected
        l_connected = self.inp_port_l.is_connected
        r_connected = self.inp_port_r.is_connected

        if not l_connected and not r_connected:
            # No input - return silence
            return np.zeros((num_samples, 2), dtype=np.float32)

        # Read from connected ports (with num_samples to trigger upstream generation)
        left_samples = self.inp_port_l.read(num_samples) if l_connected else None
        right_samples = self.inp_port_r.read(num_samples) if r_connected else None

        # Convert to numpy arrays and handle None/empty cases
        if left_samples is not None:
            left_samples = np.asarray(left_samples)
            if left_samples.size == 0:
                left_samples = None

        if right_samples is not None:
            right_samples = np.asarray(right_samples)
            if right_samples.size == 0:
                right_samples = None

        # If both are None/empty after conversion, return silence
        if left_samples is None and right_samples is None:
            return np.zeros((num_samples, 2), dtype=np.float32)

        # Build stereo output based on what's connected
        if left_samples is not None and right_samples is not None:
            # Both connected: use L for left channel, R for right channel
            stereo_samples = combine_lr_to_stereo(left_samples, right_samples)
        elif left_samples is not None:
            # Only L connected: duplicate to both channels
            stereo_samples = mono_to_stereo(left_samples)
        else:
            # Only R connected: duplicate to both channels
            stereo_samples = mono_to_stereo(right_samples)

        # Apply master gain (convert dB to linear)
        if self.gain_db <= -80.0:
            # Treat -80 dB as silence (effectively -infinity)
            return np.zeros((num_samples, 2), dtype=np.float32)
        elif self.gain_db != 0.0:
            linear_gain = 10.0 ** (self.gain_db / 20.0)
            stereo_samples = stereo_samples * linear_gain

        # Ensure correct length (pad or trim if needed)
        if stereo_samples.shape[0] != num_samples:
            result = np.zeros((num_samples, 2), dtype=np.float32)
            length = min(stereo_samples.shape[0], num_samples)
            result[:length] = stereo_samples[:length]
            return result

        return stereo_samples.astype(np.float32)

    def process(self, num_samples: int):
        """Process method for compatibility with QTimer-based architecture.

        NOTE: This method is no longer used when using callback-based pulling.
        The audio callback directly calls _generate_samples() instead.
        Kept for backward compatibility during transition.
        """
        pass  # No-op - audio callback drives everything now
