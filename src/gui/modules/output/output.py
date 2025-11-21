"""Output module UI with sample rate and buffer size controls."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtGui import QColor

from src.constants import DEFAULT_SAMPLE_RATE, DEFAULT_BUFFER_SIZE
from src.engine.io.audio_output import AudioOutput
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.module_registry import register_module
from src.gui.widgets.module_widget import ModuleWidget

if TYPE_CHECKING:
    from src.gui.core.port import Port

logger = logging.getLogger(__name__)

# Standard audio settings
SAMPLE_RATES = [22050, 44100, 48000, 88200, 96000]
BUFFER_SIZES = [128, 256, 512, 1024, 2048, 4096]


@register_module()
class OutputModule(ModuleWidget):
    """Audio output module with configurable sample rate and buffer size."""

    metadata = ModuleMetadata(
        title="Output",
        category=ModuleCategory.OUTPUT,
        description="Audio output with configurable sample rate and buffer size",
    )

    def __init__(self):
        """Initialize output module."""
        super().__init__(
            width=220,
            height=200,
            color=QColor(200, 80, 80),
        )

        # Add input port
        self.input_port: Port = self.add_input("In")

        # Create audio output handler
        self.audio_output = AudioOutput(
            sample_rate=DEFAULT_SAMPLE_RATE,
            buffer_size=DEFAULT_BUFFER_SIZE,
            audio_callback=self._generate_audio,
        )

        # Build UI
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Sample rate selection
        sr_layout = QtWidgets.QHBoxLayout()
        sr_label = QtWidgets.QLabel("Sample Rate:")
        sr_label.setStyleSheet("color: white; font-size: 11px;")
        self.sample_rate_combo = QtWidgets.QComboBox()

        # Add items first
        for sr in SAMPLE_RATES:
            self.sample_rate_combo.addItem(f"{sr} Hz", sr)

        # Set default
        default_idx = SAMPLE_RATES.index(DEFAULT_SAMPLE_RATE)
        self.sample_rate_combo.setCurrentIndex(default_idx)

        # Connect signal AFTER setting initial value to avoid triggering callback
        self.sample_rate_combo.currentIndexChanged.connect(self._on_sample_rate_changed)

        sr_layout.addWidget(sr_label)
        sr_layout.addWidget(self.sample_rate_combo)
        sr_layout.addStretch()
        layout.addLayout(sr_layout)

        # Buffer size selection
        buf_layout = QtWidgets.QHBoxLayout()
        buf_label = QtWidgets.QLabel("Buffer Size:")
        buf_label.setStyleSheet("color: white; font-size: 11px;")
        self.buffer_size_combo = QtWidgets.QComboBox()

        # Add items first
        for bs in BUFFER_SIZES:
            self.buffer_size_combo.addItem(f"{bs}", bs)

        # Set default
        default_idx = BUFFER_SIZES.index(DEFAULT_BUFFER_SIZE)
        self.buffer_size_combo.setCurrentIndex(default_idx)

        # Connect signal AFTER setting initial value to avoid triggering callback
        self.buffer_size_combo.currentIndexChanged.connect(self._on_buffer_size_changed)

        buf_layout.addWidget(buf_label)
        buf_layout.addWidget(self.buffer_size_combo)
        buf_layout.addStretch()
        layout.addLayout(buf_layout)

        # Status label
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

        logger.debug("OutputModule initialized")

    def _on_sample_rate_changed(self, index: int):
        """Handle sample rate selection change."""
        sample_rate = self.sample_rate_combo.itemData(index)
        logger.info(f"Sample rate changed to {sample_rate} Hz")
        self.audio_output.set_sample_rate(sample_rate)

    def _on_buffer_size_changed(self, index: int):
        """Handle buffer size selection change."""
        buffer_size = self.buffer_size_combo.itemData(index)
        logger.info(f"Buffer size changed to {buffer_size} samples")
        self.audio_output.set_buffer_size(buffer_size)

    def _generate_audio(self, num_samples: int) -> np.ndarray | None:
        """Generate audio samples from connected input.

        Args:
            num_samples: Number of samples to generate

        Returns:
            Stereo audio array or None
        """
        if self._input_component is None:
            return None

        try:
            samples = self._input_component.get_samples(num_samples)

            # Convert to stereo if needed
            if isinstance(samples, (list, tuple)) and len(samples) == 2:
                left, right = samples
                stereo = np.column_stack((left, right))
            elif isinstance(samples, np.ndarray):

                if len(samples.shape) == 1:
                    stereo = np.column_stack((samples, samples))
                else:
                    stereo = samples
            else:
                return None

            return stereo

        except Exception as e:
            logger.error(f"Error generating audio: {e}", exc_info=True)
            return None

    def get_output_component(self, port_name: str):
        """This module has no output component (it's a sink)."""
        return None

    def create_engine_component(
        self,
        input_components: list | None = None,
        modulation_components: dict | None = None,
    ):
        """Create the audio output engine component.

        Args:
            input_components: List of input audio components (should be 1)
            modulation_components: Not used for output module
        """
        if input_components is None or len(input_components) != 1:
            logger.warning(
                f"Output module expects 1 input, "
                f"got {len(input_components) if input_components else 0}"
            )
            return None

        self._input_component = input_components[0]
        logger.info(
            f"Output module connected to {type(self._input_component).__name__}"
        )

        # Don't return anything - this is a sink module
        return None

    def start_playback(self):
        """Start audio playback."""
        if not self.audio_output.is_playing:
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
        self.audio_output.cleanup()
