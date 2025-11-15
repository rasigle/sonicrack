"""Waveform display module for visualizing audio signals."""

import logging
from typing import Any

import numpy as np
from PyQt6 import QtCore
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPen, QColor, QPainterPath
from PyQt6.QtWidgets import QWidget, QLabel, QHBoxLayout

from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.module_registry import register_module
from src.gui.widgets.module_widget import ModuleWidget

logger = logging.getLogger(__name__)


@register_module()
class WaveformModule(ModuleWidget):
    """Waveform display module for real-time audio visualization.

    This is a pure visualization module - it only displays the signal and
    does NOT pass it through. It has no output port.

    **Important:** This module visualizes the FINAL OUTPUT after all processing.
    It connects to the audio engine's output signal to show what's actually
    being sent to the speakers.

    **Usage:**
    - Connect any module to the Waveform's input (optional - for validation)
    - The display always shows the final output signal
    - Place anywhere in your patch for monitoring
    """

    metadata = ModuleMetadata(
        title="Waveform",
        category=ModuleCategory.VISUALIZATION,
        description="Real-time waveform visualization (monitors final output)",
    )

    def __init__(self):
        """Initialize waveform display module."""
        super().__init__(
            width=420,
            height=260,
            color=QColor(80, 80, 120),
        )

        # Add input port (optional - for patch organization only)
        # The input connection doesn't affect what's displayed
        self.in_port = self.add_input_port("In")

        # NO OUTPUT PORT - this is a visualization-only module

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Create waveform display widget
        self.waveform_display = WaveformDisplay()
        self.waveform_display.setMinimumSize(400, 150)
        layout.addWidget(self.waveform_display)

        # Add min/max value display
        stats_layout = QHBoxLayout()

        self.min_label = QLabel("Min: 0.000")
        self.min_label.setStyleSheet("color: #ff6b6b; font-weight: bold;")
        stats_layout.addWidget(self.min_label)

        stats_layout.addStretch()

        self.max_label = QLabel("Max: 0.000")
        self.max_label.setStyleSheet("color: #4ecdc4; font-weight: bold;")
        stats_layout.addWidget(self.max_label)

        layout.addLayout(stats_layout)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Store reference to input component
        self.input_component = None

        # Audio engine connection
        self.audio_engine = None
        self._audio_engine_connected = False

        # Use timer to find and connect to audio engine
        self.connection_timer = QtCore.QTimer()
        self.connection_timer.timeout.connect(self._try_connect_audio_engine)
        self.connection_timer.setInterval(100)  # Try every 100ms
        self.connection_timer.start()

    def _try_connect_audio_engine(self):
        """Try to find and connect to the audio engine via scene/parent chain."""
        if self._audio_engine_connected:
            self.connection_timer.stop()
            return

        try:
            # Navigate: Module → Scene → View (PatchCanvas) → Window (MainWindow)
            scene = self.scene()
            if scene is None:
                return

            views = scene.views()
            if not views:
                return

            view = views[0]
            main_window = view.window()

            if not hasattr(main_window, 'audio_engine'):
                return

            # Found audio engine!
            self.audio_engine = main_window.audio_engine

            # Connect to signals
            self.audio_engine.samples_generated.connect(self._on_samples_generated)
            self.audio_engine.playback_stopped.connect(self._on_playback_stopped)

            self._audio_engine_connected = True
            self.connection_timer.stop()

            logger.info("✓ Waveform module connected to audio engine")

        except Exception as e:
            logger.debug(f"Waiting for audio engine: {e}")

    def _on_samples_generated(self, samples: np.ndarray):
        """Handle audio samples from the audio engine.

        This receives the FINAL output signal (after all processing).

        Args:
            samples: Final output samples (mono or stereo)
        """
        try:
            logger.debug(f"Waveform received {len(samples)} samples, shape={samples.shape}, "
                        f"min={np.min(samples):.3f}, max={np.max(samples):.3f}")

            # Convert stereo to mono for display
            if len(samples.shape) == 2:
                samples = np.mean(samples, axis=1)

            # Update display
            self.waveform_display.set_samples(samples)

            # Update min/max
            if len(samples) > 0:
                min_val = np.min(samples)
                max_val = np.max(samples)
                self.min_label.setText(f"Min: {min_val:+.3f}")
                self.max_label.setText(f"Max: {max_val:+.3f}")

        except Exception as e:
            logger.error(f"Error updating waveform: {e}", exc_info=True)

    def _on_playback_stopped(self):
        """Clear display when playback stops."""
        self.waveform_display.clear()
        self.min_label.setText("Min: 0.000")
        self.max_label.setText("Max: 0.000")

    # AudioModuleInterface implementation
    def get_required_inputs(self) -> list[str]:
        """Waveform display has no required inputs.

        The input port is optional - it doesn't affect what's displayed.
        The waveform always shows the final audio engine output.
        """
        return []  # No required inputs

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Create the engine component.

        Waveform is a pure visualization module with no audio processing.
        It stores the input component reference for validation but doesn't
        use it (the display shows the final audio engine output).

        Args:
            input_components: List of input audio components (optional)
            modulation_components: Dict of modulation components (not used)

        Returns:
            None (visualization modules don't produce audio output)
        """
        # Store reference for validation/debugging, but don't use it
        if input_components and len(input_components) > 0:
            self.input_component = input_components[0]
            logger.debug(
                f"Waveform display input connected: "
                f"{type(self.input_component).__name__}"
            )
        else:
            self.input_component = None
            logger.debug("Waveform display: no input connected (OK - monitors final output)")

        # Return None - this module has no audio output
        return None


    def cleanup(self):
        """Clean up resources when module is removed."""
        self.connection_timer.stop()

        # Disconnect from audio engine
        if self.audio_engine and self._audio_engine_connected:
            try:
                self.audio_engine.samples_generated.disconnect(self._on_samples_generated)
                self.audio_engine.playback_stopped.disconnect(self._on_playback_stopped)
            except Exception as e:
                logger.debug(f"Error disconnecting from audio engine: {e}")

        self.waveform_display.clear()
        self.waveform_display.clear()


class WaveformDisplay(QWidget):
    """Widget for displaying audio waveforms in real-time.

    Shows the time-domain representation of audio signals.
    """

    def __init__(self, parent: QWidget | None = None):
        """Initialize the waveform display.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        self.setMinimumSize(400, 150)
        self.samples: np.ndarray | None = None
        self.display_samples = 2048  # Number of samples to display
        self.is_stereo = False

        # Visual properties
        self.bg_color = QColor(20, 20, 25)
        self.grid_color = QColor(40, 40, 45)
        self.wave_color_mono = QColor(100, 200, 255)
        self.wave_color_left = QColor(100, 255, 100)  # Green for left
        self.wave_color_right = QColor(255, 100, 100)  # Red for right
        self.center_line_color = QColor(80, 80, 85)

    def set_samples(self, samples: np.ndarray):
        """Set the audio samples to display.

        Args:
            samples: Audio samples array (can be mono or stereo)
        """
        if samples is None or len(samples) == 0:
            self.samples = None
            self.is_stereo = False
            self.update()
            return

        # Check if stereo
        self.is_stereo = len(samples.shape) > 1 and samples.shape[1] == 2

        # Downsample if needed
        if len(samples) > self.display_samples:
            step = len(samples) // self.display_samples
            self.samples = samples[::step][: self.display_samples]
        else:
            self.samples = samples

        self.update()

    def clear(self):
        """Clear the display."""
        self.samples = None
        self.update()

    def paintEvent(self, event):
        """Paint the waveform."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw background
        painter.fillRect(self.rect(), self.bg_color)

        width = self.width()
        height = self.height()
        center_y = height / 2

        # Draw grid lines
        painter.setPen(QPen(self.grid_color, 1))
        for i in range(5):
            y = height * i / 4
            painter.drawLine(0, int(y), width, int(y))

        # Draw center line
        painter.setPen(QPen(self.center_line_color, 1))
        if self.is_stereo:
            # Draw center lines for stereo (split view)
            quarter_y = height / 4
            three_quarter_y = 3 * height / 4
            painter.drawLine(0, int(quarter_y), width, int(quarter_y))
            painter.drawLine(0, int(three_quarter_y), width, int(three_quarter_y))
            # Draw separator
            painter.setPen(QPen(self.grid_color, 2))
            painter.drawLine(0, int(center_y), width, int(center_y))
        else:
            painter.drawLine(0, int(center_y), width, int(center_y))

        # Draw waveform
        if self.samples is not None and len(self.samples) > 0:
            x_scale = width / len(self.samples)

            if self.is_stereo:
                # Draw stereo waveforms
                # Left channel (top half)
                path_left = QPainterPath()
                left_center = height / 4
                y_scale = (height / 4) * 0.9

                first_left = np.clip(self.samples[0, 0], -1, 1)
                path_left.moveTo(0, left_center - first_left * y_scale)

                for i in range(len(self.samples)):
                    x = i * x_scale
                    sample = np.clip(self.samples[i, 0], -1, 1)
                    y = left_center - sample * y_scale
                    path_left.lineTo(x, y)

                painter.setPen(QPen(self.wave_color_left, 2))
                painter.drawPath(path_left)

                # Right channel (bottom half)
                path_right = QPainterPath()
                right_center = 3 * height / 4

                first_right = np.clip(self.samples[0, 1], -1, 1)
                path_right.moveTo(0, right_center - first_right * y_scale)

                for i in range(len(self.samples)):
                    x = i * x_scale
                    sample = np.clip(self.samples[i, 1], -1, 1)
                    y = right_center - sample * y_scale
                    path_right.lineTo(x, y)

                painter.setPen(QPen(self.wave_color_right, 2))
                painter.drawPath(path_right)

                # Draw labels
                painter.setPen(QColor(150, 255, 150))
                painter.drawText(5, 15, "L")
                painter.setPen(QColor(255, 150, 150))
                painter.drawText(5, int(center_y) + 15, "R")
            else:
                # Draw mono waveform
                path = QPainterPath()
                y_scale = (height / 2) * 0.9

                first_sample = np.clip(self.samples[0], -1, 1)
                path.moveTo(0, center_y - first_sample * y_scale)

                for i, sample in enumerate(self.samples):
                    x = i * x_scale
                    sample = np.clip(sample, -1, 1)
                    y = center_y - sample * y_scale
                    path.lineTo(x, y)

                painter.setPen(QPen(self.wave_color_mono, 2))
                painter.drawPath(path)
        else:
            # Draw "No Signal" text
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Signal")
