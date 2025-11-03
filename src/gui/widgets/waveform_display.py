"""Real-time waveform display widget."""

from typing import Optional
import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPen, QColor, QPainterPath


class WaveformDisplay(QWidget):
    """Widget for displaying audio waveforms in real-time.

    Shows the time-domain representation of audio signals.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the waveform display.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        self.setMinimumSize(400, 150)
        self.samples: Optional[np.ndarray] = None
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
