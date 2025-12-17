"""Waveform display module for visualizing audio signals."""

import logging

import numpy as np
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

    This is a visualization module that displays the waveform of the input signal.
    Connect it anywhere in your signal chain to visualize the audio.

    **Usage:**
    - Connect audio signal to the "In" port
    - The display shows the waveform of the input signal
    - Great for monitoring signal levels and debugging
    """

    metadata = ModuleMetadata(
        title="Waveform",
        category=ModuleCategory.VISUALIZATION,
        description="Real-time waveform visualization",
    )

    def __init__(self):
        """Initialize waveform display module."""
        super().__init__(
            width=420,
            height=260,
            color=QColor(80, 80, 120),
        )

        # Add input and output ports - pass signal through!
        self.in_port = self.add_input("In")

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

    def _update_samples(self, samples: np.ndarray):
        """Update the display with new audio samples.

        Args:
            samples: Audio samples (mono or stereo) in a flexible set of formats:
                     - 1D array: (N,)
                     - 2D array sample-major: (N, 2)
                     - 2D array channel-major: (2, N)
                     - list or tuple of channel arrays: [left_arr, right_arr]
        """
        try:
            if samples is None:
                self.waveform_display.clear()
                self.min_label.setText("Min: 0.000")
                self.max_label.setText("Max: 0.000")
                return

            # Convert lists/tuples of channel-arrays -> (N, 2)
            if isinstance(samples, (list, tuple)) and len(samples) >= 1:
                # If each element is an array-like channel, stack them as columns
                if all(isinstance(ch, (np.ndarray, list, tuple)) for ch in samples):
                    try:
                        samples = np.stack([np.asarray(ch).ravel() for ch in samples], axis=1)
                    except Exception:
                        samples = np.asarray(samples)
                else:
                    samples = np.asarray(samples)

            samples = np.asarray(samples)

            # If channel-major (2, N) convert to (N, 2)
            if samples.ndim == 2 and samples.shape[0] == 2 and samples.shape[1] > 2:
                samples = samples.T

            # If many channels (>2), downmix to stereo/mono: average across channels
            if samples.ndim == 2 and samples.shape[1] > 2:
                # Average channels into stereo if exactly 2 groups? fallback to mono average
                samples = np.mean(samples, axis=1)

            # If stereo as (N,2) keep stereo; if 2D but shape (2,) or other small shapes, flatten
            if samples.ndim == 1:
                display_samples = samples.astype(np.float32)
                self.waveform_display.set_samples(display_samples)
                if display_samples.size > 0:
                    self.min_label.setText(f"Min: {np.min(display_samples):.3f}")
                    self.max_label.setText(f"Max: {np.max(display_samples):.3f}")
                else:
                    self.min_label.setText("Min: 0.000")
                    self.max_label.setText("Max: 0.000")
            elif samples.ndim == 2 and samples.shape[1] == 2:
                display_samples = samples.astype(np.float32)
                self.waveform_display.set_samples(display_samples)
                if display_samples.size > 0:
                    # compute min/max across both channels
                    min_val = float(np.min(display_samples))
                    max_val = float(np.max(display_samples))
                    self.min_label.setText(f"Min: {min_val:.3f}")
                    self.max_label.setText(f"Max: {max_val:.3f}")
                else:
                    self.min_label.setText("Min: 0.000")
                    self.max_label.setText("Max: 0.000")
            else:
                # Fallback: try flattening / treating as mono
                display_samples = samples.ravel().astype(np.float32)
                self.waveform_display.set_samples(display_samples)
                if display_samples.size > 0:
                    self.min_label.setText(f"Min: {np.min(display_samples):.3f}")
                    self.max_label.setText(f"Max: {np.max(display_samples):.3f}")
                else:
                    self.min_label.setText("Min: 0.000")
                    self.max_label.setText("Max: 0.000")

        except Exception as e:
            logger.warning(f"Error updating waveform: {e}")

    # AudioModuleInterface implementation
    def get_required_inputs(self) -> list[str]:
        """Waveform requires input to visualize.

        Returns:
            List of required input port names
        """
        return []  # Optional input - show "No Signal" if not connected

    def process(self, num_samples: int = 1):
        """Process audio data and update waveform display.

        Reads from the input port and updates the waveform display with
        the incoming audio signal.

        Args:
            num_samples: Number of samples to process
        """
        logger.debug(f"Waveform: process() called, in_port.is_connected={self.in_port.is_connected}")

        # Check if input is connected
        if not self.in_port.is_connected:
            # No input - clear display
            self.waveform_display.clear()
            self.min_label.setText("Min: 0.000")
            self.max_label.setText("Max: 0.000")
            return

        # Read input samples
        samples = self.in_port.read()
        logger.debug(f"Waveform: Read samples: {samples is not None}, type={type(samples) if samples is not None else None}")

        if samples is None:
            self.waveform_display.clear()
            return

        # Update display with new samples
        self._update_samples(samples)


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
        # Accept None or empty
        if samples is None or samples.size == 0:
            self.samples = None
            self.is_stereo = False
            self.update()
            return

        samples = np.asarray(samples)

        # If channel-major (2, N) convert to (N, 2)
        if samples.ndim == 2 and samples.shape[0] == 2 and samples.shape[1] > 2:
            samples = samples.T

        # Determine stereo vs mono: expect (N,2) for stereo
        if samples.ndim == 2 and samples.shape[1] == 2:
            self.is_stereo = True
        else:
            # If 2D but second dimension isn't 2, flatten to 1D
            if samples.ndim == 2:
                samples = samples.ravel()
            self.is_stereo = False

        # Downsample if needed (preserve columns for stereo)
        if not self.is_stereo:
            if samples.size > self.display_samples:
                step = samples.size // self.display_samples
                self.samples = samples[::step][: self.display_samples]
            else:
                self.samples = samples
        else:
            # stereo: samples shape is (N,2)
            if samples.shape[0] > self.display_samples:
                step = samples.shape[0] // self.display_samples
                self.samples = samples[::step][: self.display_samples, :]
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
            x_scale = width / max(1, len(self.samples))

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
