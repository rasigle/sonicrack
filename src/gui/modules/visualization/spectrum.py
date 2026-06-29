"""Spectrum analyzer module for visualizing audio frequency content."""

import logging
import threading

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from src.gui.audio_config import audio_config
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.module_registry import register_module
from src.gui.modules.visualization.visualizer_utils import (
    get_visualizer_samples,
    validate_samples,
)
from src.gui.widgets.module_widget import ModuleWidget

logger = logging.getLogger(__name__)


@register_module()
class SpectrumModule(ModuleWidget):
    """Spectrum analyzer module for real-time frequency visualization.

    This is a visualization module that displays the frequency spectrum
    of the input signal. Connect it after your audio processing chain
    to see the frequency content.

    **Usage:**
    - Connect audio signal to the "In" port
    - The display shows the frequency spectrum of the input signal
    - Great for analyzing frequency content and monitoring mix

    Runtime behavior:
    - Passive sink in the render graph; it does not process upstream modules.
    - When audio output is playing, it displays samples tapped from the shared
      render path.
    - Without active audio output, AudioEngine's monitor timer renders only the
      connected visualizer sink ports so sources still animate silently.
    """

    runtime_kind = "passive_sink"

    metadata = ModuleMetadata(
        title="Spectrum",
        category=ModuleCategory.VISUALIZATION,
        description="Real-time frequency spectrum display (FFT analyzer)",
    )

    # Passive sink: receives rendered buffers without running as a processor.
    is_processing_module = False

    def __init__(self):
        """Initialize spectrum analyzer module."""
        super().__init__(
            width=420,
            height=260,
            color=QColor(100, 80, 120),
        )

        # Add input port
        self.in_port = self.add_input("In")

        # Thread-safe buffer for audio samples
        self._sample_buffer = None
        self._buffer_lock = None
        try:
            self._buffer_lock = threading.Lock()
        except ImportError:
            self._buffer_lock = None
            logger.warning("Threading not available - spectrum may have issues")

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Create spectrum analyzer widget
        self.spectrum_display = SpectrumAnalyzer()
        self.spectrum_display.setMinimumSize(400, 150)
        layout.addWidget(self.spectrum_display)

        # Add peak frequency display
        stats_layout = QHBoxLayout()

        self.peak_freq_label = QLabel("Peak: -- Hz")
        self.peak_freq_label.setStyleSheet("color: #4ecdc4; font-weight: bold;")
        stats_layout.addWidget(self.peak_freq_label)

        stats_layout.addStretch()

        self.level_label = QLabel("Level: -- dB")
        self.level_label.setStyleSheet("color: #f39c12; font-weight: bold;")
        stats_layout.addWidget(self.level_label)

        layout.addLayout(stats_layout)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Sample rate for frequency calculation
        self._sample_rate = audio_config.sample_rate

        # Register for sample rate updates
        audio_config.add_sample_rate_listener(self._on_sample_rate_changed)

        # Visualization timer (20 Hz = 50ms, on UI thread)
        # Using QTimer is simpler and works correctly with Qt event loop
        from PyQt6.QtCore import QTimer

        self._viz_timer = QTimer()
        self._viz_timer.setInterval(50)  # 50ms = 20 Hz
        self._viz_timer.timeout.connect(self._update_display)
        self._viz_timer.start()
        logger.info("Spectrum visualization timer started (20 Hz / 50ms)")

    def _on_sample_rate_changed(self, new_sample_rate: int):
        """Handle sample rate changes.

        Args:
            new_sample_rate: New sample rate in Hz
        """
        self._sample_rate = new_sample_rate

    def _update_samples(self, samples: np.ndarray):
        """Update the display with new audio samples.

        Args:
            samples: Audio samples (mono or stereo)
        """
        try:
            logger.debug(
                f"Spectrum: Updating with {len(samples)} samples, shape={samples.shape}"
            )

            # Convert stereo to mono for FFT
            if len(samples.shape) == 2:
                samples = np.mean(samples, axis=1)

            # Update spectrum display
            self.spectrum_display.set_samples(samples)

            # Calculate and display peak frequency
            if len(samples) > 0:
                self._update_peak_frequency(samples)

        except Exception as e:
            logger.warning(f"Error updating spectrum: {e}")

    def _update_peak_frequency(self, samples: np.ndarray):
        """Calculate and display the peak frequency and level.

        Args:
            samples: Audio samples
        """
        try:
            # Calculate level in dB from time-domain peak amplitude (to match waveform)
            peak_amplitude = np.max(np.abs(samples))
            peak_level = (
                20 * np.log10(peak_amplitude + 1e-10) if peak_amplitude > 0 else -100.0
            )

            # Compute FFT
            n = min(len(samples), 4096)
            if n < 256:
                self.level_label.setText(f"Level: {peak_level:.1f} dB")
                return

            # Apply window
            window = np.hanning(n)
            windowed = samples[:n] * window

            # FFT
            fft = np.fft.rfft(windowed)
            magnitude = np.abs(fft)

            # Find peak
            peak_idx = np.argmax(magnitude)

            # Convert to frequency
            freq_resolution = self._sample_rate / (2 * len(magnitude))
            peak_freq = peak_idx * freq_resolution

            # Update labels
            if peak_freq < 1000:
                self.peak_freq_label.setText(f"Peak: {peak_freq:.1f} Hz")
            else:
                self.peak_freq_label.setText(f"Peak: {peak_freq / 1000:.2f} kHz")

            self.level_label.setText(f"Level: {peak_level:.1f} dB")

        except Exception as e:
            logger.debug(f"Error calculating peak frequency: {e}")

    # AudioModuleInterface implementation
    def get_required_inputs(self) -> list[str]:
        """Spectrum analyzer requires input to visualize.

        Returns:
            List of required input port names
        """
        return []  # Optional input - show "No Signal" if not connected

    def _update_display(self):
        """Update the spectrum display from rendered port tap history."""
        # Check if input is connected
        if not self.in_port.is_connected:
            # No input - clear display
            self.spectrum_display.clear()
            self.peak_freq_label.setText("Peak: -- Hz")
            self.level_label.setText("Level: -- dB")
            return

        # FFT needs at least 1024 samples for good frequency resolution
        samples = get_visualizer_samples(self.in_port, num_samples=1024)

        # Validate and display samples
        if not validate_samples(samples):
            self.spectrum_display.clear()
            self.peak_freq_label.setText("Peak: -- Hz")
            self.level_label.setText("Level: -- dB")
            return

        # Update display with samples
        if samples:
            self._update_samples(samples)


class SpectrumAnalyzer(QWidget):
    """Widget for displaying audio spectrum in real-time.

    Shows the frequency-domain representation of audio signals using FFT.
    """

    def __init__(self, parent: QWidget | None = None):
        """Initialize the spectrum analyzer.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        self.setMinimumSize(400, 150)
        self.fft_data: np.ndarray | None = None
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
