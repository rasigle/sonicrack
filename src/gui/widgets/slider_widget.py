"""Slider widgets for parameter control."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget


class VSlider(QWidget):
    """Vertical slider with label and value display."""

    value_changed = pyqtSignal(float)

    def __init__(
        self,
        label: str = "",
        min_value: float = 0.0,
        max_value: float = 1.0,
        default_value: float | None = None,
        parent: QWidget | None = None,
    ):
        """Initialize vertical slider.

        Args:
            label: Label text
            min_value: Minimum value
            max_value: Maximum value
            default_value: Default value
            parent: Parent widget
        """
        super().__init__(parent)

        self.min_value = min_value
        self.max_value = max_value

        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)

        # Value label
        self.value_label = QLabel()
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_label.setStyleSheet("color: #c2e5ff;")
        layout.addWidget(self.value_label)

        # Slider
        self.slider = QSlider(Qt.Orientation.Vertical)
        self.slider.setMinimum(0)
        self.slider.setMaximum(1000)
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider, 1)

        # Label
        self.label = QLabel(label)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("color: #edf1f5;")
        layout.addWidget(self.label)

        self.setLayout(layout)

        # Set default value
        default = default_value if default_value is not None else min_value
        self.set_value(default)

    def _on_slider_changed(self, slider_value: int):
        """Handle slider value change."""
        value = self._slider_to_value(slider_value)
        self.value_label.setText(f"{value:.2f}")
        self.value_changed.emit(value)

    def _value_to_slider(self, value: float) -> int:
        """Convert value to slider position."""
        norm = (value - self.min_value) / (self.max_value - self.min_value)
        return int(norm * 1000)

    def _slider_to_value(self, slider_value: int) -> float:
        """Convert slider position to value."""
        norm = slider_value / 1000.0
        return self.min_value + norm * (self.max_value - self.min_value)

    def get_value(self) -> float:
        """Get current value."""
        return self._slider_to_value(self.slider.value())

    def set_value(self, value: float):
        """Set current value."""
        slider_value = self._value_to_slider(value)
        self.slider.setValue(slider_value)


class HSlider(QWidget):
    """Horizontal slider with label and value display."""

    value_changed = pyqtSignal(float)

    def __init__(
        self,
        label: str = "",
        min_value: float = 0.0,
        max_value: float = 1.0,
        default_value: float | None = None,
        parent: QWidget | None = None,
    ):
        """Initialize horizontal slider.

        Args:
            label: Label text
            min_value: Minimum value
            max_value: Maximum value
            default_value: Default value
            parent: Parent widget
        """
        super().__init__(parent)

        self.min_value = min_value
        self.max_value = max_value

        layout = QHBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)

        # Label
        self.label = QLabel(label)
        self.label.setStyleSheet("color: #edf1f5;")
        layout.addWidget(self.label)

        # Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(1000)
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider, 1)

        # Value label
        self.value_label = QLabel()
        self.value_label.setMinimumWidth(50)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.value_label.setStyleSheet("color: #c2e5ff;")
        layout.addWidget(self.value_label)

        self.setLayout(layout)

        # Set default value
        default = default_value if default_value is not None else min_value
        self.set_value(default)

    def _on_slider_changed(self, slider_value: int):
        """Handle slider value change."""
        value = self._slider_to_value(slider_value)
        self.value_label.setText(f"{value:.2f}")
        self.value_changed.emit(value)

    def _value_to_slider(self, value: float) -> int:
        """Convert value to slider position."""
        norm = (value - self.min_value) / (self.max_value - self.min_value)
        return int(norm * 1000)

    def _slider_to_value(self, slider_value: int) -> float:
        """Convert slider position to value."""
        norm = slider_value / 1000.0
        return self.min_value + norm * (self.max_value - self.min_value)

    def get_value(self) -> float:
        """Get current value."""
        return self._slider_to_value(self.slider.value())

    def set_value(self, value: float):
        """Set current value."""
        slider_value = self._value_to_slider(value)
        self.slider.setValue(slider_value)
