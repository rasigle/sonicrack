"""Spectrum analyzer module for visualizing audio frequency content."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from sonicrack.config.audio_config import audio_config
from sonicrack.gui.modules.visualization.visualizer_utils import (
    get_visualizer_samples,
    stop_visualizer_timer,
)
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module

logger = logging.getLogger(__name__)


def _as_mono_float32(samples: Any) -> np.ndarray | None:
    """Convert mono/stereo samples to a finite mono float32 array.

    QWidget work stays on the UI thread. This function only sanitizes already
    rendered tap-history samples.
    """
    if samples is None:
        return None

    arr = np.asarray(samples, dtype=np.float32)

    if arr.size == 0:
        return None

    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)

    # Common stereo layouts:
    #   (N, 2): frame-major stereo
    #   (2, N): channel-major stereo
    if arr.ndim == 2:
        if arr.shape[1] == 2:
            arr = arr.mean(axis=1)
        elif arr.shape[0] == 2:
            arr = arr.mean(axis=0)
        else:
            arr = arr.ravel()
    else:
        arr = arr.ravel()

    if arr.size == 0:
        return None

    return np.clip(arr.astype(np.float32, copy=False), -1.0, 1.0)


@dataclass
class SpectrumResult:
    """Prepared spectrum data and stats for display."""

    bars: np.ndarray
    peak_frequency_hz: float | None
    level_db: float


@register_module()
class SpectrumModule(ModuleWidget):
    """Spectrum analyzer module for real-time frequency visualization.

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

    is_processing_module = False

    def __init__(self):
        """Initialize spectrum analyzer module."""
        super().__init__(
            width=420,
            height=260,
            color=QColor(100, 80, 120),
        )

        self.in_port = self.add_input("In")

        self._sample_rate = int(audio_config.sample_rate)
        self._fft_size = 2048

        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        self.spectrum_display = SpectrumAnalyzer(
            sample_rate=self._sample_rate,
            fft_size=self._fft_size,
        )
        self.spectrum_display.setMinimumSize(400, 150)
        layout.addWidget(self.spectrum_display)

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

        audio_config.add_sample_rate_listener(self._on_sample_rate_changed)

        # 20 Hz is a good analyzer refresh rate and should not fight the audio
        # thread. CoarseTimer lets Qt coalesce UI work more efficiently.
        self._viz_timer = QTimer(self)
        self._viz_timer.setTimerType(Qt.TimerType.CoarseTimer)
        self._viz_timer.setInterval(50)
        self._viz_timer.timeout.connect(self._update_display)
        self._viz_timer.start()

        logger.info("Spectrum visualization timer started")

    def _on_sample_rate_changed(self, new_sample_rate: int) -> None:
        """Handle sample-rate changes."""
        self._sample_rate = int(new_sample_rate)
        self.spectrum_display.set_sample_rate(self._sample_rate)

    def _update_display(self) -> None:
        """Update the spectrum display from rendered port tap history."""
        try:
            if not self.in_port.is_connected:
                self._show_no_signal()
                return

            # Ask for enough samples to get useful bass resolution, but keep
            # this bounded so the UI thread cost stays predictable.
            samples = get_visualizer_samples(self.in_port, num_samples=self._fft_size)

            mono = _as_mono_float32(samples)
            if mono is None or mono.size < 256:
                self._show_no_signal()
                return

            result = self.spectrum_display.set_samples(mono)

            if result is None:
                self._show_no_signal()
                return

            self._update_labels(result)

        except Exception:
            logger.exception("Error in SpectrumModule._update_display")

    def _show_no_signal(self) -> None:
        """Clear display and labels without redundant UI churn."""
        self.spectrum_display.clear()
        self._set_label_text(self.peak_freq_label, "Peak: -- Hz")
        self._set_label_text(self.level_label, "Level: -- dB")

    def _update_labels(self, result: SpectrumResult) -> None:
        """Update peak and level labels."""
        if result.peak_frequency_hz is None:
            peak_text = "Peak: -- Hz"
        elif result.peak_frequency_hz < 1000.0:
            peak_text = f"Peak: {result.peak_frequency_hz:.1f} Hz"
        else:
            peak_text = f"Peak: {result.peak_frequency_hz / 1000.0:.2f} kHz"

        self._set_label_text(self.peak_freq_label, peak_text)
        self._set_label_text(self.level_label, f"Level: {result.level_db:.1f} dB")

    @staticmethod
    def _set_label_text(label: QLabel, text: str) -> None:
        """Avoid extra layout/paint work when text is unchanged."""
        if label.text() != text:
            label.setText(text)

    def get_required_inputs(self) -> list[str]:
        """Spectrum has optional input and shows 'No Signal' when disconnected."""
        return []

    def shutdown(self, graceful: bool = True) -> None:
        """Stop visualization updates before the module is deleted."""
        del graceful
        stop_visualizer_timer(self, self._update_display)

        try:
            audio_config.remove_sample_rate_listener(self._on_sample_rate_changed)
        except Exception:
            logger.debug(
                "Could not remove spectrum sample-rate listener",
                exc_info=True,
            )


class SpectrumAnalyzer(QWidget):
    """Widget for displaying real-time frequency spectrum.

    FFT and bucket preparation happen in set_samples(). paintEvent() only draws
    cached bars and cached grid/background.
    """

    MIN_FREQ_HZ = 20.0
    FLOOR_DB = -80.0
    CEILING_DB = 0.0

    def __init__(
        self,
        parent: QWidget | None = None,
        sample_rate: int = 44100,
        fft_size: int = 2048,
    ):
        """Initialize the spectrum analyzer."""
        super().__init__(parent)

        self.setMinimumSize(400, 150)

        self.sample_rate = int(sample_rate)
        self.fft_size = int(fft_size)

        self.fft_bins = 96
        self.bars: np.ndarray | None = None

        self.bg_color_top = QColor(20, 20, 25)
        self.bg_color_bottom = QColor(14, 14, 18)

        self.grid_color = QColor(40, 40, 45)
        self.text_color = QColor(110, 110, 115)
        self.bar_color = QColor(255, 150, 0)
        self.bar_hot_color = QColor(255, 200, 0)
        self.peak_color = QColor(255, 0, 0)

        self._window: np.ndarray | None = None
        self._window_size: int | None = None
        self._freqs: np.ndarray | None = None

        self._bucket_indices: list[np.ndarray] = []
        self._bucket_key: tuple[int, int, int] | None = None

        self._grid_pixmap: QPixmap | None = None
        self._grid_key: tuple[int, int] | None = None

        self._prepare_fft_cache()

    def set_sample_rate(self, sample_rate: int) -> None:
        """Update sample rate and invalidate frequency-dependent caches."""
        sample_rate = int(sample_rate)
        if sample_rate == self.sample_rate:
            return

        self.sample_rate = sample_rate
        self._freqs = None
        self._bucket_indices = []
        self._bucket_key = None
        self._grid_pixmap = None
        self._grid_key = None
        self._prepare_fft_cache()
        self.update()

    def set_fft_size(self, fft_size: int) -> None:
        """Set FFT size and invalidate related caches."""
        fft_size = max(256, int(fft_size))
        if fft_size == self.fft_size:
            return

        self.fft_size = fft_size
        self._window = None
        self._window_size = None
        self._freqs = None
        self._bucket_indices = []
        self._bucket_key = None
        self._prepare_fft_cache()
        self.update()

    def set_samples(self, samples: np.ndarray | None) -> SpectrumResult | None:
        """Set audio samples, compute one FFT, and cache display bars."""
        mono = _as_mono_float32(samples)

        if mono is None or mono.size < 256:
            self.clear()
            return None

        n = min(mono.size, self.fft_size)
        if n < 256:
            self.clear()
            return None

        # Use the most recent n samples so the analyzer tracks current signal.
        mono = mono[-n:]

        self._ensure_window(n)
        self._ensure_freqs(n)
        self._ensure_bucket_indices(n)

        assert self._window is not None

        level_db = self._time_domain_level_db(mono)

        windowed = mono * self._window

        fft = np.fft.rfft(windowed)
        magnitude = np.abs(fft).astype(np.float32, copy=False)

        # Normalize roughly against Hann window gain so dB is stable and does
        # not constantly pin to 0 dB just because FFT length changed.
        window_gain = max(float(np.sum(self._window)) * 0.5, 1e-12)
        magnitude = magnitude / window_gain

        magnitude = np.maximum(magnitude, 1e-10)
        db = 20.0 * np.log10(magnitude)

        # Ignore DC for peak-frequency reporting. A DC offset should not become
        # the musical/audio peak.
        peak_frequency_hz = self._find_peak_frequency(db)

        normalized = np.clip(
            (db - self.FLOOR_DB) / (self.CEILING_DB - self.FLOOR_DB),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)

        self.bars = self._bucketize(normalized)
        self.update()

        return SpectrumResult(
            bars=self.bars,
            peak_frequency_hz=peak_frequency_hz,
            level_db=level_db,
        )

    def clear(self) -> None:
        """Clear the display."""
        if self.bars is None:
            return

        self.bars = None
        self.update()

    def _prepare_fft_cache(self) -> None:
        """Prepare caches for the current FFT size/sample rate."""
        self._ensure_window(self.fft_size)
        self._ensure_freqs(self.fft_size)
        self._ensure_bucket_indices(self.fft_size)

    def _ensure_window(self, n: int) -> None:
        """Cache Hann window for a given FFT length."""
        if self._window is not None and self._window_size == n:
            return

        self._window = np.hanning(n).astype(np.float32)
        self._window_size = n

    def _ensure_freqs(self, n: int) -> None:
        """Cache rFFT frequency bins."""
        if self._freqs is not None and self._freqs.size == (n // 2 + 1):
            return

        self._freqs = np.fft.rfftfreq(n, d=1.0 / float(self.sample_rate)).astype(
            np.float32
        )

    def _ensure_bucket_indices(self, n: int) -> None:
        """Build cached logarithmic bucket indices."""
        key = (n, self.sample_rate, self.fft_bins)
        if self._bucket_key == key and self._bucket_indices:
            return

        self._ensure_freqs(n)
        assert self._freqs is not None

        nyquist = max(float(self.sample_rate) * 0.5, self.MIN_FREQ_HZ * 2.0)
        max_freq = max(self.MIN_FREQ_HZ * 2.0, nyquist)

        edges = np.geomspace(self.MIN_FREQ_HZ, max_freq, self.fft_bins + 1)

        bucket_indices: list[np.ndarray] = []
        for i in range(self.fft_bins):
            low = edges[i]
            high = edges[i + 1]

            idx = np.where((self._freqs >= low) & (self._freqs < high))[0]

            # Low-frequency buckets can be empty when FFT size is small. Use the
            # nearest bin so every visual bucket has a defined value.
            if idx.size == 0:
                nearest = int(np.argmin(np.abs(self._freqs - ((low + high) * 0.5))))
                idx = np.array([nearest], dtype=np.int64)

            bucket_indices.append(idx)

        self._bucket_indices = bucket_indices
        self._bucket_key = key

    def _bucketize(self, normalized: np.ndarray) -> np.ndarray:
        """Convert FFT-bin magnitudes to log-spaced display bars."""
        if not self._bucket_indices:
            return normalized[: self.fft_bins].astype(np.float32, copy=False)

        bars = np.empty(len(self._bucket_indices), dtype=np.float32)

        for i, idx in enumerate(self._bucket_indices):
            safe_idx = idx[idx < normalized.size]
            if safe_idx.size == 0:
                bars[i] = 0.0
            else:
                bars[i] = float(np.max(normalized[safe_idx]))

        return bars

    def _find_peak_frequency(self, db: np.ndarray) -> float | None:
        """Find peak frequency, ignoring DC and very-low-frequency bins."""
        if db.size <= 1:
            return None

        self._ensure_freqs((db.size - 1) * 2)
        assert self._freqs is not None

        valid = np.where(self._freqs >= self.MIN_FREQ_HZ)[0]
        valid = valid[valid < db.size]

        if valid.size == 0:
            return None

        peak_idx = int(valid[np.argmax(db[valid])])
        return float(self._freqs[peak_idx])

    @staticmethod
    def _time_domain_level_db(samples: np.ndarray) -> float:
        """Return peak level from time-domain samples."""
        if samples.size == 0:
            return -100.0

        peak = float(np.max(np.abs(samples)))
        if peak <= 0.0:
            return -100.0

        return float(20.0 * np.log10(max(peak, 1e-10)))

    def resizeEvent(self, event) -> None:
        """Invalidate cached background when resized."""
        self._grid_pixmap = None
        self._grid_key = None
        super().resizeEvent(event)

    def _ensure_grid_pixmap(self) -> QPixmap:
        """Return cached background/grid pixmap."""
        width = max(1, self.width())
        height = max(1, self.height())
        key = (width, height)

        if self._grid_pixmap is not None and self._grid_key == key:
            return self._grid_pixmap

        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        gradient = QLinearGradient(0, 0, 0, height)
        gradient.setColorAt(0, self.bg_color_top)
        gradient.setColorAt(1, self.bg_color_bottom)
        painter.fillRect(0, 0, width, height, gradient)

        self._draw_grid(painter, width, height)

        painter.end()

        self._grid_pixmap = pixmap
        self._grid_key = key
        return pixmap

    def _draw_grid(self, painter: QPainter, width: int, height: int) -> None:
        """Draw cached spectrum grid and labels."""
        painter.setPen(QPen(self.grid_color, 1))

        # Horizontal dB grid.
        for i in range(5):
            y = int(height * i / 4)
            painter.drawLine(0, y, width, y)

        # Vertical log-frequency guide lines.
        freqs = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        nyquist = max(float(self.sample_rate) * 0.5, self.MIN_FREQ_HZ * 2.0)
        max_freq = min(20000.0, nyquist)

        for freq in freqs:
            if freq < self.MIN_FREQ_HZ or freq > max_freq:
                continue

            x = self._freq_to_x(freq, width, max_freq)
            painter.drawLine(int(x), 0, int(x), height)

        painter.setPen(self.text_color)
        painter.setFont(QFont("Arial", 8))

        labels = [20, 200, 1000, 2000, 10000, 20000]
        for freq in labels:
            if freq > max_freq:
                continue

            x = self._freq_to_x(freq, width, max_freq)

            if freq < 1000:
                label = f"{freq}Hz"
            elif freq == 1000:
                label = "1k"
            else:
                label = f"{int(freq / 1000)}k"

            painter.drawText(int(x) - 14, height - 5, label)

    def _freq_to_x(self, freq: float, width: int, max_freq: float) -> float:
        """Map frequency to x coordinate using logarithmic scale."""
        min_f = self.MIN_FREQ_HZ
        max_f = max(max_freq, min_f * 2.0)

        pos = np.log(freq / min_f) / np.log(max_f / min_f)
        return float(np.clip(pos, 0.0, 1.0) * width)

    def paintEvent(self, event) -> None:
        """Paint the spectrum from cached bars."""
        del event

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        width = self.width()
        height = self.height()

        painter.drawPixmap(0, 0, self._ensure_grid_pixmap())

        if self.bars is None or self.bars.size == 0:
            painter.setPen(self.text_color)
            painter.setFont(QFont("Arial", 12))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Signal")
            painter.end()
            return

        self._draw_bars(painter, width, height)
        painter.end()

    def _draw_bars(self, painter: QPainter, width: int, height: int) -> None:
        """Draw cached spectrum bars."""
        assert self.bars is not None

        usable_height = max(1, height - 20)
        bar_count = int(self.bars.size)
        bar_width = width / max(1, bar_count)

        for i, magnitude in enumerate(self.bars):
            mag = float(np.clip(magnitude, 0.0, 1.0))

            x = int(i * bar_width)
            bar_h = int(mag * usable_height)
            y = height - bar_h - 15

            if mag > 0.9:
                color = self.peak_color
            elif mag > 0.7:
                color = self.bar_hot_color
            else:
                color = self.bar_color

            painter.fillRect(
                x,
                int(y),
                max(1, int(bar_width) - 1),
                max(1, bar_h),
                color,
            )
