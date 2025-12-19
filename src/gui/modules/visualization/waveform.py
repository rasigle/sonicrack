"""Waveform display module for visualizing audio signals."""

import logging
import threading

import numpy as np
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPen, QColor, QPainterPath, QLinearGradient, QBrush, \
    QFont
from PyQt6.QtWidgets import QWidget, QLabel, QHBoxLayout

from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.module_registry import register_module
from src.gui.widgets.knob_widget import Knob
from src.gui.widgets.module_widget import ModuleWidget

logger = logging.getLogger(__name__)


@register_module()
class WaveformModule(ModuleWidget):
    """Professional waveform display module for real-time audio visualization.

    This is an enhanced visualization module that displays audio waveforms with:
    - Dual inputs: L/Mono and R channels
    - Adjustable timerange (samples displayed)
    - Freeze trigger to capture and hold waveforms
    - Professional gradient visuals with grid overlay

    **Usage:**
    - Connect audio to "L/Mono" port for mono, or both "L/Mono" and "R" for stereo
    - Adjust "Time" knob to change the number of samples displayed
    - Click "Freeze" button to capture current waveform
    - Great for oscilloscope-style monitoring and debugging
    """

    metadata = ModuleMetadata(
        title="Waveform",
        category=ModuleCategory.VISUALIZATION,
        description="Professional oscilloscope-style waveform display",
    )

    # Mark as non-processing to exclude from audio chain pulling
    is_processing_module = False

    def __init__(self):
        """Initialize enhanced waveform display module."""
        super().__init__(
            width=500,
            height=340,
            color=QColor(60, 70, 90),
        )

        # Add dual input ports for Left/Mono and Right channels
        self.in_port_l = self.add_input("L/Mono")
        self.in_port_r = self.add_input("R")

        # Freeze state
        self._is_frozen = False
        self._frozen_samples = None

        # Thread-safe buffer for audio samples
        self._sample_buffer = None
        self._buffer_lock = None
        try:
            self._buffer_lock = threading.Lock()
        except ImportError:
            self._buffer_lock = None
            logger.warning("Threading not available - waveform may have issues")

        # Visualization timer (20 Hz = 50ms, on UI thread)
        self._viz_timer = QtCore.QTimer()
        self._viz_timer.setInterval(50)  # 50ms = 20 Hz
        self._viz_timer.timeout.connect(self._update_display)
        self._viz_timer.start()
        logger.info("Waveform visualization timer started (20 Hz / 50ms)")

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Create controls row with timerange knob and freeze button
        controls_row = QHBoxLayout()

        # Timerange knob (512 to 8192 samples)
        self.timerange_knob = Knob(
            label="Time",
            min_value=512,
            max_value=8192,
            default_value=2048,
            logarithmic=False,
            callback=self._on_timerange_changed
        )
        self.timerange_knob.setFixedSize(60, 80)
        controls_row.addWidget(self.timerange_knob)

        controls_row.addStretch()

        # Freeze trigger button
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

        # Create waveform display widget
        self.waveform_display = WaveformDisplay()
        self.waveform_display.setMinimumSize(480, 200)
        layout.addWidget(self.waveform_display)

        # Add statistics display
        stats_layout = QHBoxLayout()

        self.min_label = QLabel("Min: 0.000")
        self.min_label.setStyleSheet("color: #ff6b6b; font-weight: bold; font-size: 11px;")
        stats_layout.addWidget(self.min_label)

        self.peak_label = QLabel("Peak: 0.0 dB")
        self.peak_label.setStyleSheet("color: #ffd93d; font-weight: bold; font-size: 11px;")
        stats_layout.addWidget(self.peak_label)

        stats_layout.addStretch()

        self.samples_label = QLabel("Samples: 0")
        self.samples_label.setStyleSheet("color: #a0a0a0; font-size: 10px;")
        stats_layout.addWidget(self.samples_label)

        stats_layout.addStretch()

        self.max_label = QLabel("Max: 0.000")
        self.max_label.setStyleSheet("color: #4ecdc4; font-weight: bold; font-size: 11px;")
        stats_layout.addWidget(self.max_label)

        layout.addLayout(stats_layout)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

    def _on_timerange_changed(self, value: float):
        """Handle timerange knob change.

        Args:
            value: New timerange value (number of samples to display)
        """
        samples = int(value)
        self.waveform_display.set_display_samples(samples)
        logger.debug(f"Waveform timerange changed to {samples} samples")

    def _on_freeze_toggled(self, checked: bool):
        """Handle freeze button toggle.

        Args:
            checked: True if freeze is enabled
        """
        self._is_frozen = checked
        if checked:
            logger.info("Waveform frozen - capturing current display")
            self.freeze_button.setText("Frozen")
        else:
            logger.info("Waveform unfrozen - resuming live display")
            self.freeze_button.setText("Freeze")
            self._frozen_samples = None

    def _update_samples(self, samples_l: np.ndarray | None, samples_r: np.ndarray | None):
        """Update the display with new audio samples.

        Args:
            samples_l: Left/Mono channel samples
            samples_r: Right channel samples (optional)
        """
        try:
            # If frozen, keep displaying frozen samples
            if self._is_frozen:
                if self._frozen_samples is not None:
                    self.waveform_display.set_samples(self._frozen_samples)
                return

            # Handle empty/None inputs
            if samples_l is None and samples_r is None:
                self.waveform_display.clear()
                self._update_stats(None)
                return

            # Convert to numpy arrays
            if samples_l is not None:
                samples_l = np.asarray(samples_l).ravel().astype(np.float32)
            if samples_r is not None:
                samples_r = np.asarray(samples_r).ravel().astype(np.float32)

            # Combine into stereo if both channels present
            if samples_l is not None and samples_r is not None:
                # Make sure both have same length
                min_len = min(len(samples_l), len(samples_r))
                samples_l = samples_l[:min_len]
                samples_r = samples_r[:min_len]
                display_samples = np.stack([samples_l, samples_r], axis=1)
            elif samples_l is not None:
                # Only left channel - treat as mono
                display_samples = samples_l
            else:
                # Only right channel - treat as mono
                display_samples = samples_r

            # Store frozen samples if not frozen yet
            if not self._is_frozen:
                self._frozen_samples = display_samples

            # Update display
            self.waveform_display.set_samples(display_samples)
            self._update_stats(display_samples)

        except Exception as e:
            logger.warning(f"Error updating waveform: {e}")

    def _update_stats(self, samples: np.ndarray | None):
        """Update statistics labels.

        Args:
            samples: Audio samples to analyze
        """
        if samples is None or samples.size == 0:
            self.min_label.setText("Min: 0.000")
            self.max_label.setText("Max: 0.000")
            self.peak_label.setText("Peak: -∞ dB")
            self.samples_label.setText("Samples: 0")
            return

        # Calculate statistics
        min_val = float(np.min(samples))
        max_val = float(np.max(samples))
        peak_amplitude = np.max(np.abs(samples))
        peak_db = 20 * np.log10(peak_amplitude + 1e-10) if peak_amplitude > 0 else -100.0

        # Update labels
        self.min_label.setText(f"Min: {min_val:.3f}")
        self.max_label.setText(f"Max: {max_val:.3f}")
        self.peak_label.setText(f"Peak: {peak_db:.1f} dB")

        # Show sample count
        if samples.ndim == 1:
            self.samples_label.setText(f"Samples: {len(samples)}")
        else:
            self.samples_label.setText(f"Samples: {samples.shape[0]} × {samples.shape[1]}")

    def get_required_inputs(self) -> list[str]:
        """Waveform requires at least one input to visualize.

        Returns:
            List of required input port names
        """
        return []  # Optional inputs - show "No Signal" if not connected

    def _update_display(self):
        """Update the waveform display at 20 Hz (independent of audio rate).

        HYBRID MODE:
        - PASSIVE when output module is playing: reads buffered samples (no interference)
        - ACTIVE when standalone: actively pulls samples (enables visualization without output)
        """
        try:
            # If frozen, keep showing frozen samples but still update display
            if self._is_frozen:
                if self._frozen_samples is not None:
                    self.waveform_display.set_samples(self._frozen_samples)
                    self._update_stats(self._frozen_samples)
                return

            # Check if at least one input is connected
            l_connected = self.in_port_l.is_connected
            r_connected = self.in_port_r.is_connected

            if not l_connected and not r_connected:
                # No input - clear display
                self.waveform_display.clear()
                self._update_stats(None)
                return

            # Get current timerange
            num_samples = int(self.timerange_knob.get_value())

            # ALWAYS use active mode - directly generate samples for real-time visualization
            # This ensures we get fresh samples on each timer tick
            samples_l = None
            samples_r = None

            if l_connected:
                # Directly trigger sample generation from connected module
                for connected_port in self.in_port_l.connected_to:
                    if connected_port.parent_module and hasattr(connected_port.parent_module, 'process'):
                        try:
                            # Generate fresh samples
                            connected_port.parent_module.process(num_samples)
                            if connected_port.value is not None:
                                if isinstance(connected_port.value, np.ndarray) and connected_port.value.size > 0:
                                    samples_l = connected_port.value.copy()
                                    logger.debug(f"Waveform: Got L samples, shape={samples_l.shape}")
                                    break
                        except Exception as e:
                            logger.debug(f"Error generating L samples: {e}")

            if r_connected:
                # Directly trigger sample generation from connected module
                for connected_port in self.in_port_r.connected_to:
                    if connected_port.parent_module and hasattr(connected_port.parent_module, 'process'):
                        try:
                            # Generate fresh samples
                            connected_port.parent_module.process(num_samples)
                            if connected_port.value is not None:
                                if isinstance(connected_port.value, np.ndarray) and connected_port.value.size > 0:
                                    samples_r = connected_port.value.copy()
                                    logger.debug(f"Waveform: Got R samples, shape={samples_r.shape}")
                                    break
                        except Exception as e:
                            logger.debug(f"Error generating R samples: {e}")

            # Update display with samples
            if samples_l is not None or samples_r is not None:
                self._update_samples(samples_l, samples_r)
            else:
                # No samples received - show "No Signal"
                self.waveform_display.clear()
                self._update_stats(None)

        except Exception as e:
            logger.error(f"Error in _update_display: {e}", exc_info=True)

    def process(self, num_samples: int = 1):
        """Process method for audio path.

        For visualizers: This is a NO-OP. Visualizers observe samples via observe_samples()
        which is called explicitly by modules that want to share their output.

        Visualizers are NOT part of the audio processing chain to avoid any interference.

        Args:
            num_samples: Number of samples (ignored)
        """
        pass  # Visualizers don't process - they only observe


class WaveformDisplay(QWidget):
    """Professional widget for displaying audio waveforms in real-time.

    Features gradient fills, precise grid overlay, and oscilloscope-style rendering.
    """

    def __init__(self, parent: QWidget | None = None):
        """Initialize the waveform display.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        self.setMinimumSize(480, 200)
        self.samples: np.ndarray | None = None
        self.display_samples = 2048  # Number of samples to display (adjustable)
        self.is_stereo = False

        # Professional visual properties
        self.bg_color = QColor(15, 18, 22)
        self.grid_color = QColor(30, 35, 40)
        self.grid_highlight_color = QColor(45, 52, 60)
        self.center_line_color = QColor(60, 70, 80)

        # Waveform colors with alpha for gradient effect
        self.wave_color_mono = QColor(80, 180, 255)
        self.wave_fill_mono = QColor(80, 180, 255, 60)

        self.wave_color_left = QColor(80, 255, 120)
        self.wave_fill_left = QColor(80, 255, 120, 50)

        self.wave_color_right = QColor(255, 120, 100)
        self.wave_fill_right = QColor(255, 120, 100, 50)

        self.text_color = QColor(120, 130, 140)
        self.label_color = QColor(200, 210, 220)

    def set_display_samples(self, num_samples: int):
        """Set the number of samples to display.

        Args:
            num_samples: Number of samples for the timerange
        """
        self.display_samples = max(128, min(num_samples, 16384))
        self.update()

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
                step = max(1, samples.size // self.display_samples)
                self.samples = samples[::step][: self.display_samples]
            else:
                self.samples = samples
        else:
            # stereo: samples shape is (N,2)
            if samples.shape[0] > self.display_samples:
                step = max(1, samples.shape[0] // self.display_samples)
                self.samples = samples[::step][: self.display_samples, :]
            else:
                self.samples = samples

        self.update()

    def clear(self):
        """Clear the display."""
        self.samples = None
        self.update()

    def _draw_professional_grid(self, painter: QPainter, width: int, height: int):
        """Draw professional oscilloscope-style grid.

        Args:
            painter: QPainter instance
            width: Widget width
            height: Widget height
        """
        # Draw vertical grid lines (time divisions)
        painter.setPen(QPen(self.grid_color, 1))
        num_v_lines = 10
        for i in range(num_v_lines + 1):
            x = width * i / num_v_lines
            painter.drawLine(int(x), 0, int(x), height)

        # Highlight center vertical line
        painter.setPen(QPen(self.grid_highlight_color, 1))
        center_x = width / 2
        painter.drawLine(int(center_x), 0, int(center_x), height)

        # Draw horizontal grid lines (amplitude divisions)
        painter.setPen(QPen(self.grid_color, 1))
        if self.is_stereo:
            # Stereo: divide into two sections
            num_h_lines = 8
            for i in range(num_h_lines + 1):
                y = height * i / num_h_lines
                painter.drawLine(0, int(y), width, int(y))

            # Highlight center lines for each channel
            painter.setPen(QPen(self.center_line_color, 2))
            quarter_y = height / 4
            three_quarter_y = 3 * height / 4
            painter.drawLine(0, int(quarter_y), width, int(quarter_y))
            painter.drawLine(0, int(three_quarter_y), width, int(three_quarter_y))

            # Draw separator between channels
            painter.setPen(QPen(self.grid_highlight_color, 2))
            center_y = height / 2
            painter.drawLine(0, int(center_y), width, int(center_y))
        else:
            # Mono: standard grid
            num_h_lines = 8
            for i in range(num_h_lines + 1):
                y = height * i / num_h_lines
                painter.drawLine(0, int(y), width, int(y))

            # Highlight center line
            painter.setPen(QPen(self.center_line_color, 2))
            center_y = height / 2
            painter.drawLine(0, int(center_y), width, int(center_y))

    def _draw_waveform_with_fill(self, painter: QPainter, samples: np.ndarray,
                                   center_y: float, y_scale: float,
                                   width: int, wave_color: QColor, fill_color: QColor):
        """Draw waveform with gradient fill effect.

        Args:
            painter: QPainter instance
            samples: Sample data to draw
            center_y: Center Y position
            y_scale: Y scaling factor
            width: Widget width
            wave_color: Color for waveform line
            fill_color: Color for fill gradient
        """
        if len(samples) == 0:
            return

        x_scale = width / max(1, len(samples) - 1)

        # Create waveform path
        path = QPainterPath()
        first_sample = np.clip(samples[0], -1, 1)
        path.moveTo(0, center_y - first_sample * y_scale)

        for i, sample in enumerate(samples):
            x = i * x_scale
            sample = np.clip(sample, -1, 1)
            y = center_y - sample * y_scale
            path.lineTo(x, y)

        # Draw filled area under waveform
        fill_path = QPainterPath(path)
        fill_path.lineTo(width, center_y)
        fill_path.lineTo(0, center_y)
        fill_path.closeSubpath()

        painter.fillPath(fill_path, QBrush(fill_color))

        # Draw waveform line with glow effect
        painter.setPen(QPen(wave_color, 2.5, Qt.PenStyle.SolidLine,
                           Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawPath(path)

    def paintEvent(self, event):
        """Paint the professional waveform display."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw background with gradient
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(15, 18, 22))
        gradient.setColorAt(1, QColor(20, 23, 28))
        painter.fillRect(self.rect(), QBrush(gradient))

        width = self.width()
        height = self.height()
        center_y = height / 2

        # Draw professional grid
        self._draw_professional_grid(painter, width, height)

        # Draw waveform
        if self.samples is not None and len(self.samples) > 0:
            if self.is_stereo:
                # Draw stereo waveforms with fills
                left_center = height / 4
                right_center = 3 * height / 4
                y_scale = (height / 4) * 0.85

                # Left channel
                self._draw_waveform_with_fill(
                    painter, self.samples[:, 0], left_center, y_scale,
                    width, self.wave_color_left, self.wave_fill_left
                )

                # Right channel
                self._draw_waveform_with_fill(
                    painter, self.samples[:, 1], right_center, y_scale,
                    width, self.wave_color_right, self.wave_fill_right
                )

                # Draw channel labels with better styling
                painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
                painter.setPen(self.wave_color_left)
                painter.drawText(10, 20, "L")
                painter.setPen(self.wave_color_right)
                painter.drawText(10, int(center_y) + 20, "R")

                # Draw scale markers
                painter.setPen(self.text_color)
                painter.setFont(QFont("Arial", 8))
                painter.drawText(width - 30, int(left_center - y_scale) + 12, "+1.0")
                painter.drawText(width - 30, int(left_center) + 4, "0.0")
                painter.drawText(width - 30, int(left_center + y_scale) + 12, "-1.0")
            else:
                # Draw mono waveform with fill
                y_scale = (height / 2) * 0.85

                self._draw_waveform_with_fill(
                    painter, self.samples, center_y, y_scale,
                    width, self.wave_color_mono, self.wave_fill_mono
                )

                # Draw scale markers
                painter.setPen(self.text_color)
                painter.setFont(QFont("Arial", 8))
                painter.drawText(width - 30, int(center_y - y_scale) + 12, "+1.0")
                painter.drawText(width - 30, int(center_y) + 4, "0.0")
                painter.drawText(width - 30, int(center_y + y_scale) + 12, "-1.0")
        else:
            # Draw "No Signal" with better styling
            painter.setPen(self.text_color)
            painter.setFont(QFont("Arial", 12))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Signal")


