"""Real-time spectrum analyzer widget."""

from typing import Optional
import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush


class SpectrumAnalyzer(QWidget):
    """Widget for displaying audio spectrum in real-time.

    Shows the frequency-domain representation of audio signals using FFT.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the spectrum analyzer.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        self.setMinimumSize(400, 150)
        self.fft_data: Optional[np.ndarray] = None
        self.fft_bins = 256

        # Visual properties
        self.bg_color = QColor(20, 20, 25)
        self.grid_color = QColor(40, 40, 45)
        self.bar_color = QColor(255, 150, 0)
        self.peak_color = QColor(255, 0, 0)

    def set_samples(self, samples: np.ndarray):
        """Set the audio samples and compute FFT.

        Args:
            samples: Audio samples array
        """
        if samples is None or len(samples) == 0:
            self.fft_data = None
            self.update()
            return

        # Handle stereo by taking first channel
        if len(samples.shape) > 1 and samples.shape[1] == 2:
            samples = samples[:, 0]

        # Compute FFT
        n = min(len(samples), 4096)
        if n < 256:
            self.fft_data = None
            self.update()
            return

        # Apply window function
        window = np.hanning(n)
        windowed = samples[:n] * window

        # Compute FFT
        fft = np.fft.rfft(windowed)
        magnitude = np.abs(fft)

        # Convert to dB
        magnitude = np.maximum(magnitude, 1e-10)  # Avoid log(0)
        db = 20 * np.log10(magnitude)

        # Normalize to 0-1 range (assuming -80 dB to 0 dB)
        db = np.clip(db, -80, 0)
        normalized = (db + 80) / 80

        # Downsample to display bins
        bins_per_bucket = len(normalized) // self.fft_bins
        if bins_per_bucket > 0:
            buckets = []
            for i in range(self.fft_bins):
                start = i * bins_per_bucket
                end = start + bins_per_bucket
                if end <= len(normalized):
                    buckets.append(np.max(normalized[start:end]))
            self.fft_data = np.array(buckets)
        else:
            self.fft_data = normalized[: self.fft_bins]

        self.update()

    def clear(self):
        """Clear the display."""
        self.fft_data = None
        self.update()

    def paintEvent(self, event):
        """Paint the spectrum."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw background
        painter.fillRect(self.rect(), self.bg_color)

        width = self.width()
        height = self.height()

        # Draw grid lines (horizontal)
        painter.setPen(QPen(self.grid_color, 1))
        for i in range(5):
            y = height * i / 4
            painter.drawLine(0, int(y), width, int(y))

        # Draw frequency scale labels
        painter.setPen(QColor(100, 100, 100))
        freq_labels = ["20Hz", "200Hz", "2kHz", "20kHz"]
        for i, label in enumerate(freq_labels):
            x = width * i / (len(freq_labels) - 1)
            painter.drawText(int(x) - 20, height - 5, label)

        # Draw spectrum bars
        if self.fft_data is not None and len(self.fft_data) > 0:
            bar_width = width / len(self.fft_data)

            for i, magnitude in enumerate(self.fft_data):
                x = i * bar_width
                bar_height = magnitude * (height - 20)
                y = height - bar_height - 15

                # Color gradient based on level
                if magnitude > 0.9:
                    color = self.peak_color
                elif magnitude > 0.7:
                    color = QColor(255, 200, 0)
                else:
                    color = self.bar_color

                painter.fillRect(
                    int(x), int(y), max(1, int(bar_width) - 1), int(bar_height), color
                )
        else:
            # Draw "No Signal" text
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Signal")
