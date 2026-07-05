"""Waveform display module for visualizing audio signals."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.module_registry import register_module
from src.gui.modules.visualization.visualizer_utils import (
    get_visualizer_samples,
    stop_visualizer_timer,
)
from src.gui.widgets.knob_widget import Knob
from src.gui.widgets.module_widget import ModuleWidget

logger = logging.getLogger(__name__)


def _as_float32_1d(samples: Any) -> np.ndarray | None:
    """Convert input samples to a finite float32 1D array.

    Silence is valid. NaN and infinities are sanitized so the painter never
    receives invalid coordinates.
    """
    if samples is None:
        return None

    arr = np.asarray(samples, dtype=np.float32)

    if arr.size == 0:
        return None

    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    arr = np.clip(arr.ravel(), -1.0, 1.0)

    if arr.size == 0:
        return None

    return arr


def _downsample_minmax(samples: np.ndarray, max_points: int) -> np.ndarray:
    """Downsample audio for display while preserving visible peaks.

    Simple decimation can miss short transients. Min/max envelope downsampling
    preserves peaks and gives a stable oscilloscope-style display.
    """
    samples = np.asarray(samples, dtype=np.float32).ravel()

    if samples.size == 0:
        return samples

    max_points = max(2, int(max_points))

    if samples.size <= max_points:
        return samples

    # We output two points per bucket: min and max.
    bucket_count = max(1, max_points // 2)
    bucket_size = max(1, samples.size // bucket_count)
    usable = bucket_count * bucket_size

    if usable <= 0:
        return samples[-max_points:]

    trimmed = samples[-usable:]
    buckets = trimmed.reshape(bucket_count, bucket_size)

    mins = buckets.min(axis=1)
    maxs = buckets.max(axis=1)

    envelope = np.empty(bucket_count * 2, dtype=np.float32)
    envelope[0::2] = mins
    envelope[1::2] = maxs

    return envelope[:max_points]


@dataclass
class WaveformStats:
    """Small immutable-ish container for display stats."""

    min_value: float = 0.0
    max_value: float = 0.0
    peak_db: float = -100.0
    sample_text: str = "Samples: 0"


@register_module()
class WaveformModule(ModuleWidget):
    """Professional waveform display module for real-time audio visualization.

    This is a passive visualizer. It never renders upstream modules itself.
    Audio output and AudioEngine's monitor timer render the graph and write tap
    history to ports; this widget only reads recent cached tap samples.
    """

    runtime_kind = "passive_sink"

    metadata = ModuleMetadata(
        title="Waveform",
        category=ModuleCategory.VISUALIZATION,
        description="Professional oscilloscope-style waveform display",
    )

    # Passive sink: receives rendered buffers without running as a processor.
    is_processing_module = False

    def __init__(self):
        """Initialize enhanced waveform display module."""
        super().__init__(
            width=500,
            height=340,
            color=QColor(60, 70, 90),
        )

        self.in_port_l = self.add_input("L/Mono")
        self.in_port_r = self.add_input("R")

        self._is_frozen = False
        self._frozen_samples: np.ndarray | None = None
        self._last_display_samples: np.ndarray | None = None

        # Visualization timer.
        #
        # 50 ms = 20 Hz, responsive enough for a scope, but still bounded so it
        # should not compete with audio processing. Since paintEvent is now cheap
        # and cached, this can be adjusted safely if needed.
        self._viz_timer = QtCore.QTimer(self)
        self._viz_timer.setTimerType(Qt.TimerType.CoarseTimer)
        self._viz_timer.setInterval(50)
        self._viz_timer.timeout.connect(self._update_display)

        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        controls_row = QHBoxLayout()

        self.timerange_knob = Knob(
            label="Time",
            min_value=64,
            max_value=8192,
            default_value=2048,
            logarithmic=False,
            callback=self._on_timerange_changed,
        )
        self.timerange_knob.setFixedSize(88, 78)
        controls_row.addWidget(self.timerange_knob)

        controls_row.addStretch()

        self.freeze_button = QtWidgets.QPushButton("Freeze")
        self.freeze_button.setCheckable(True)
        self.freeze_button.setFixedSize(80, 30)
        self.freeze_button.setStyleSheet("""
            QPushButton {
                background-color: #3a4555;
                color: #ffffff;
                border: 2px solid #4a5565;
                border-radius: 5px;
                font-weight: bold;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #4a5565;
                border: 2px solid #5a6575;
            }
            QPushButton:checked {
                background-color: #ff6b6b;
                border: 2px solid #ff8b8b;
            }
            QPushButton:pressed {
                background-color: #2a3545;
            }
        """)
        self.freeze_button.clicked.connect(self._on_freeze_toggled)
        controls_row.addWidget(self.freeze_button)

        layout.addLayout(controls_row)

        self.waveform_display = WaveformDisplay()
        self.waveform_display.setMinimumSize(480, 200)
        layout.addWidget(self.waveform_display)

        stats_layout = QHBoxLayout()

        self.min_label = QLabel("Min: 0.000")
        self.min_label.setStyleSheet(
            "color: #ff6b6b; font-weight: bold; font-size: 11px;"
        )
        stats_layout.addWidget(self.min_label)

        self.peak_label = QLabel("Peak: -100.0 dB")
        self.peak_label.setStyleSheet(
            "color: #ffd93d; font-weight: bold; font-size: 11px;"
        )
        stats_layout.addWidget(self.peak_label)

        stats_layout.addStretch()

        self.samples_label = QLabel("Samples: 0")
        self.samples_label.setStyleSheet("color: #a0a0a0; font-size: 10px;")
        stats_layout.addWidget(self.samples_label)

        stats_layout.addStretch()

        self.max_label = QLabel("Max: 0.000")
        self.max_label.setStyleSheet(
            "color: #4ecdc4; font-weight: bold; font-size: 11px;"
        )
        stats_layout.addWidget(self.max_label)

        layout.addLayout(stats_layout)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        self._viz_timer.start()
        logger.info("Waveform visualization timer started")

    def _on_timerange_changed(self, value: float) -> None:
        """Handle timerange knob change."""
        samples = int(value)
        self.waveform_display.set_display_samples(samples)
        logger.debug("Waveform timerange changed to %s samples", samples)

    def _on_freeze_toggled(self, checked: bool) -> None:
        """Handle freeze button toggle."""
        self._is_frozen = checked

        if checked:
            # Capture the current visible/prepared samples once.
            if self._last_display_samples is not None:
                self._frozen_samples = self._last_display_samples.copy()
            else:
                self._frozen_samples = None

            self.freeze_button.setText("Frozen")
            logger.info("Waveform frozen")
        else:
            self._frozen_samples = None
            self.freeze_button.setText("Freeze")
            logger.info("Waveform unfrozen")

    def _update_samples(
        self,
        samples_l: np.ndarray | None,
        samples_r: np.ndarray | None,
    ) -> None:
        """Update the display with new audio samples."""
        if self._is_frozen:
            if self._frozen_samples is not None:
                self.waveform_display.set_samples(self._frozen_samples)
                self._update_stats(self._frozen_samples)
            return

        left = _as_float32_1d(samples_l)
        right = _as_float32_1d(samples_r)

        if left is None and right is None:
            self._last_display_samples = None
            self.waveform_display.clear()
            self._update_stats(None)
            return

        if left is not None and right is not None:
            min_len = min(left.size, right.size)
            if min_len == 0:
                self._last_display_samples = None
                self.waveform_display.clear()
                self._update_stats(None)
                return

            display_samples = np.column_stack(
                (left[-min_len:], right[-min_len:])
            ).astype(np.float32, copy=False)

        elif left is not None:
            display_samples = left

        elif right is not None:
            display_samples = right

        else:
            self._last_display_samples = None
            self.waveform_display.clear()
            self._update_stats(None)
            return

        self._last_display_samples = display_samples
        self.waveform_display.set_samples(display_samples)
        self._update_stats(display_samples)

    def _update_stats(self, samples: np.ndarray | None) -> None:
        """Update statistics labels, avoiding redundant setText calls."""
        stats = self._calculate_stats(samples)

        self._set_label_text(self.min_label, f"Min: {stats.min_value:.3f}")
        self._set_label_text(self.max_label, f"Max: {stats.max_value:.3f}")
        self._set_label_text(self.peak_label, f"Peak: {stats.peak_db:.1f} dB")
        self._set_label_text(self.samples_label, stats.sample_text)

    @staticmethod
    def _calculate_stats(samples: np.ndarray | None) -> WaveformStats:
        """Calculate safe display statistics."""
        if samples is None:
            return WaveformStats()

        arr = np.asarray(samples, dtype=np.float32)

        if arr.size == 0:
            return WaveformStats()

        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)

        min_val = float(np.min(arr))
        max_val = float(np.max(arr))

        peak_amplitude = float(np.max(np.abs(arr)))
        if peak_amplitude <= 0.0:
            peak_db = -100.0
        else:
            peak_db = float(20.0 * np.log10(max(peak_amplitude, 1e-10)))

        if arr.ndim == 1:
            sample_text = f"Samples: {arr.shape[0]}"
        elif arr.ndim == 2:
            sample_text = f"Samples: {arr.shape[0]} x {arr.shape[1]}"
        else:
            sample_text = f"Samples: {arr.size}"

        return WaveformStats(
            min_value=min_val,
            max_value=max_val,
            peak_db=peak_db,
            sample_text=sample_text,
        )

    @staticmethod
    def _set_label_text(label: QLabel, text: str) -> None:
        """Avoid triggering layout/paint work if the label text is unchanged."""
        if label.text() != text:
            label.setText(text)

    def get_required_inputs(self) -> list[str]:
        """Waveform has optional inputs and shows 'No Signal' when disconnected."""
        return []

    def _update_display(self) -> None:
        """Update waveform from rendered port tap history."""
        try:
            if self._is_frozen:
                if self._frozen_samples is not None:
                    self.waveform_display.set_samples(self._frozen_samples)
                    self._update_stats(self._frozen_samples)
                return

            l_connected = self.in_port_l.is_connected
            r_connected = self.in_port_r.is_connected

            if not l_connected and not r_connected:
                self._last_display_samples = None
                self.waveform_display.clear()
                self._update_stats(None)
                return

            num_samples = int(self.timerange_knob.get_value())

            samples_l = (
                get_visualizer_samples(self.in_port_l, num_samples)
                if l_connected
                else None
            )
            samples_r = (
                get_visualizer_samples(self.in_port_r, num_samples)
                if r_connected
                else None
            )

            if samples_l is None and samples_r is None:
                self._last_display_samples = None
                self.waveform_display.clear()
                self._update_stats(None)
                return

            self._update_samples(samples_l, samples_r)

        except Exception:
            logger.exception("Error in WaveformModule._update_display")

    def shutdown(self, graceful: bool = True) -> None:
        """Stop visualization updates before the module is deleted."""
        del graceful
        stop_visualizer_timer(self, self._update_display)


class WaveformDisplay(QWidget):
    """Widget for displaying real-time waveform data.

    Expensive work is done when samples change. paintEvent only draws cached
    background/grid and cached waveform paths.
    """

    def __init__(self, parent: QWidget | None = None):
        """Initialize the waveform display."""
        super().__init__(parent)

        self.setMinimumSize(480, 200)

        self.display_samples = 2048
        self.is_stereo = False
        self._has_signal = False

        self._mono_path: QPainterPath | None = None
        self._mono_fill_path: QPainterPath | None = None

        self._left_path: QPainterPath | None = None
        self._left_fill_path: QPainterPath | None = None
        self._right_path: QPainterPath | None = None
        self._right_fill_path: QPainterPath | None = None

        self._grid_pixmap: QPixmap | None = None
        self._grid_key: tuple[int, int, bool] | None = None

        self._last_prepared_key: tuple[int, int, bool, int] | None = None
        self._prepared_samples: np.ndarray | None = None

        self.bg_color_top = QColor(15, 18, 22)
        self.bg_color_bottom = QColor(20, 23, 28)

        self.grid_color = QColor(30, 35, 40)
        self.grid_highlight_color = QColor(45, 52, 60)
        self.center_line_color = QColor(60, 70, 80)

        self.wave_color_mono = QColor(80, 180, 255)
        self.wave_fill_mono = QColor(80, 180, 255, 60)

        self.wave_color_left = QColor(80, 255, 120)
        self.wave_fill_left = QColor(80, 255, 120, 50)

        self.wave_color_right = QColor(255, 120, 100)
        self.wave_fill_right = QColor(255, 120, 100, 50)

        self.text_color = QColor(120, 130, 140)

    def set_display_samples(self, num_samples: int) -> None:
        """Set the number of source samples represented in the display."""
        new_value = max(128, min(int(num_samples), 16384))
        if self.display_samples == new_value:
            return

        self.display_samples = new_value

        # Reprepare from the last source sample buffer, if present.
        if self._prepared_samples is not None:
            self.set_samples(self._prepared_samples)
        else:
            self.update()

    def set_samples(self, samples: np.ndarray | None) -> None:
        """Set audio samples and prepare cached painter paths."""
        if samples is None:
            self.clear()
            return

        arr = np.asarray(samples, dtype=np.float32)

        if arr.size == 0:
            self.clear()
            return

        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        arr = np.clip(arr, -1.0, 1.0)

        # Normalize channel-major stereo, if needed.
        if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] > 2:
            arr = arr.T

        if arr.ndim == 2 and arr.shape[1] == 2:
            is_stereo = True
        else:
            is_stereo = False
            arr = arr.ravel()

        self.is_stereo = is_stereo
        self._has_signal = True
        self._prepared_samples = arr

        self._prepare_paths(arr)
        self.update()

    def clear(self) -> None:
        """Clear the display."""
        if not self._has_signal and self._mono_path is None:
            return

        self._has_signal = False
        self.is_stereo = False
        self._prepared_samples = None
        self._last_prepared_key = None

        self._mono_path = None
        self._mono_fill_path = None
        self._left_path = None
        self._left_fill_path = None
        self._right_path = None
        self._right_fill_path = None

        self.update()

    def resizeEvent(self, event) -> None:
        """Invalidate cached geometry when the widget size changes."""
        self._grid_pixmap = None
        self._grid_key = None
        self._last_prepared_key = None

        if self._prepared_samples is not None:
            self._prepare_paths(self._prepared_samples)

        super().resizeEvent(event)

    def _target_point_count(self) -> int:
        """Return a bounded point count based on current widget width."""
        return max(128, min(self.display_samples, max(128, self.width() * 2)))

    def _prepare_paths(self, samples: np.ndarray) -> None:
        """Prepare cached QPainterPaths for the current samples."""
        width = max(1, self.width())
        height = max(1, self.height())
        target = self._target_point_count()

        key = (width, height, self.is_stereo, target)

        # Always reprepare when new samples arrive. The key is mostly useful for
        # resize/display changes and future extension with versioned buffers.
        self._last_prepared_key = key

        self._mono_path = None
        self._mono_fill_path = None
        self._left_path = None
        self._left_fill_path = None
        self._right_path = None
        self._right_fill_path = None

        if self.is_stereo:
            stereo = np.asarray(samples, dtype=np.float32)
            if stereo.ndim != 2 or stereo.shape[1] != 2 or stereo.shape[0] == 0:
                self._has_signal = False
                return

            left = _downsample_minmax(stereo[:, 0], target)
            right = _downsample_minmax(stereo[:, 1], target)

            left_center = height / 4.0
            right_center = 3.0 * height / 4.0
            y_scale = (height / 4.0) * 0.85

            self._left_path, self._left_fill_path = self._build_wave_paths(
                left,
                left_center,
                y_scale,
                width,
            )
            self._right_path, self._right_fill_path = self._build_wave_paths(
                right,
                right_center,
                y_scale,
                width,
            )

        else:
            mono = np.asarray(samples, dtype=np.float32).ravel()
            if mono.size == 0:
                self._has_signal = False
                return

            mono = _downsample_minmax(mono, target)

            center_y = height / 2.0
            y_scale = (height / 2.0) * 0.85

            self._mono_path, self._mono_fill_path = self._build_wave_paths(
                mono,
                center_y,
                y_scale,
                width,
            )

    @staticmethod
    def _build_wave_paths(
        samples: np.ndarray,
        center_y: float,
        y_scale: float,
        width: int,
    ) -> tuple[QPainterPath, QPainterPath]:
        """Build line and fill paths for a waveform."""
        samples = np.asarray(samples, dtype=np.float32).ravel()

        path = QPainterPath()
        fill_path = QPainterPath()

        if samples.size == 0:
            return path, fill_path

        x_scale = width / max(1, samples.size - 1)

        first_y = center_y - float(samples[0]) * y_scale
        path.moveTo(0.0, first_y)

        for i in range(1, samples.size):
            x = float(i) * x_scale
            y = center_y - float(samples[i]) * y_scale
            path.lineTo(x, y)

        fill_path = QPainterPath(path)
        fill_path.lineTo(float(width), center_y)
        fill_path.lineTo(0.0, center_y)
        fill_path.closeSubpath()

        return path, fill_path

    def _ensure_grid_pixmap(self) -> QPixmap:
        """Return cached background/grid pixmap, rebuilding only when needed."""
        width = max(1, self.width())
        height = max(1, self.height())
        key = (width, height, self.is_stereo)

        if self._grid_pixmap is not None and self._grid_key == key:
            return self._grid_pixmap

        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        gradient = QLinearGradient(0, 0, 0, height)
        gradient.setColorAt(0, self.bg_color_top)
        gradient.setColorAt(1, self.bg_color_bottom)
        painter.fillRect(0, 0, width, height, QBrush(gradient))

        self._draw_grid(painter, width, height)

        painter.end()

        self._grid_pixmap = pixmap
        self._grid_key = key

        return pixmap

    def _draw_grid(self, painter: QPainter, width: int, height: int) -> None:
        """Draw oscilloscope-style grid into the cached pixmap."""
        painter.setPen(QPen(self.grid_color, 1))

        num_v_lines = 10
        for i in range(num_v_lines + 1):
            x = int(width * i / num_v_lines)
            painter.drawLine(x, 0, x, height)

        painter.setPen(QPen(self.grid_highlight_color, 1))
        center_x = int(width / 2)
        painter.drawLine(center_x, 0, center_x, height)

        painter.setPen(QPen(self.grid_color, 1))
        num_h_lines = 8
        for i in range(num_h_lines + 1):
            y = int(height * i / num_h_lines)
            painter.drawLine(0, y, width, y)

        if self.is_stereo:
            painter.setPen(QPen(self.center_line_color, 2))
            quarter_y = int(height / 4)
            three_quarter_y = int(3 * height / 4)
            painter.drawLine(0, quarter_y, width, quarter_y)
            painter.drawLine(0, three_quarter_y, width, three_quarter_y)

            painter.setPen(QPen(self.grid_highlight_color, 2))
            center_y = int(height / 2)
            painter.drawLine(0, center_y, width, center_y)
        else:
            painter.setPen(QPen(self.center_line_color, 2))
            center_y = int(height / 2)
            painter.drawLine(0, center_y, width, center_y)

    def paintEvent(self, event) -> None:
        """Paint the cached waveform display."""
        del event

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        painter.drawPixmap(0, 0, self._ensure_grid_pixmap())

        width = self.width()
        height = self.height()
        center_y = height / 2.0

        if not self._has_signal:
            painter.setPen(self.text_color)
            painter.setFont(QFont("Arial", 12))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Signal")
            painter.end()
            return

        if self.is_stereo:
            self._paint_stereo(painter, width, height, center_y)
        else:
            self._paint_mono(painter, width, height, center_y)

        painter.end()

    def _paint_mono(
        self,
        painter: QPainter,
        width: int,
        height: int,
        center_y: float,
    ) -> None:
        """Paint cached mono waveform paths."""
        if self._mono_path is None or self._mono_fill_path is None:
            return

        y_scale = (height / 2.0) * 0.85

        painter.fillPath(self._mono_fill_path, QBrush(self.wave_fill_mono))

        painter.setPen(
            QPen(
                self.wave_color_mono,
                2.0,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
        )
        painter.drawPath(self._mono_path)

        painter.setPen(self.text_color)
        painter.setFont(QFont("Arial", 8))
        painter.drawText(width - 30, int(center_y - y_scale) + 12, "+1.0")
        painter.drawText(width - 30, int(center_y) + 4, "0.0")
        painter.drawText(width - 30, int(center_y + y_scale) + 12, "-1.0")

    def _paint_stereo(
        self,
        painter: QPainter,
        width: int,
        height: int,
        center_y: float,
    ) -> None:
        """Paint cached stereo waveform paths."""
        left_center = height / 4.0
        y_scale = (height / 4.0) * 0.85

        if self._left_path is not None and self._left_fill_path is not None:
            painter.fillPath(self._left_fill_path, QBrush(self.wave_fill_left))
            painter.setPen(
                QPen(
                    self.wave_color_left,
                    2.0,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
            )
            painter.drawPath(self._left_path)

        if self._right_path is not None and self._right_fill_path is not None:
            painter.fillPath(self._right_fill_path, QBrush(self.wave_fill_right))
            painter.setPen(
                QPen(
                    self.wave_color_right,
                    2.0,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
            )
            painter.drawPath(self._right_path)

        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        painter.setPen(self.wave_color_left)
        painter.drawText(10, 20, "L")

        painter.setPen(self.wave_color_right)
        painter.drawText(10, int(center_y) + 20, "R")

        painter.setPen(self.text_color)
        painter.setFont(QFont("Arial", 8))
        painter.drawText(width - 30, int(left_center - y_scale) + 12, "+1.0")
        painter.drawText(width - 30, int(left_center) + 4, "0.0")
        painter.drawText(width - 30, int(left_center + y_scale) + 12, "-1.0")
