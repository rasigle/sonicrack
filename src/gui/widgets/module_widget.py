"""Base module widget for the modular synth interface."""

from __future__ import annotations

from abc import ABCMeta
from typing import Any, TYPE_CHECKING

from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QLinearGradient, QFont
from PyQt6.QtWidgets import (
    QGraphicsWidget,
    QGraphicsItem,
    QWidget,
    QVBoxLayout,
    QGraphicsProxyWidget,
)

from src.gui.audio_module_interface import AudioModuleInterface
from src.gui.patch_canvas import Port


if TYPE_CHECKING:
    from src.gui.audio_module_interface import ModuleCategory


# Create a compatible metaclass that combines QGraphicsWidget's metaclass with ABCMeta
class ModuleWidgetMeta(type(QGraphicsWidget), ABCMeta):
    """Combined metaclass for QGraphicsWidget and ABC."""

    pass


# Visual constants
MODULE_TYPE_FONT = QFont("Arial", 8, QFont.Weight.Bold)
MODULE_CATEGORY_FONT = QFont("Arial", 7)


class ModuleWidget(QGraphicsWidget, AudioModuleInterface, metaclass=ModuleWidgetMeta):
    """Base class for all modular synth modules.

    Each module represents an audio component (oscillator, envelope, etc.)
    with visual controls and connection ports.

    Implements AudioModuleInterface to enable generic patch compilation.
    """

    # === Visual Constants ===
    TITLE_BAR_HEIGHT = 42
    TITLE_BAR_HEIGHT_WITH_NAME = 50
    BORDER_RADIUS = 8
    SELECTION_BORDER_WIDTH = 3
    NORMAL_BORDER_WIDTH = 2

    # Colors
    COLOR_SELECTION_BORDER = QColor(255, 200, 0)
    COLOR_NORMAL_BORDER = QColor(30, 30, 30)
    COLOR_TITLE_BAR_BG = QColor(30, 30, 30, 200)
    COLOR_MODULE_TYPE = QColor(150, 150, 150)
    COLOR_CUSTOM_NAME = QColor(255, 255, 100)
    COLOR_CATEGORY = QColor(130, 130, 130)
    COLOR_CATEGORY_NO_NAME = QColor(200, 200, 200)
    COLOR_PORT_LABEL = QColor(220, 220, 220)

    # Signals
    parameter_changed = pyqtSignal(str, object)  # (param_name, value)

    def __init__(
        self,
        width: int = 200,
        height: int = 150,
        color: QColor | None = None,
    ):
        """Initialize a module widget.

        Args:
            width: Width of the module
            height: Height of the module
            color: Accent color for the module
        """
        super().__init__()

        self.module_width = width
        self.module_height = height
        self.module_color = color or QColor(80, 120, 180)

        # FIX: Initialize missing attributes
        self.custom_name = ""
        self.component = None

        # Ports
        self.input_ports: list[Port] = []
        self.output_ports: list[Port] = []

        # Parameter registry for automatic get/set (widget, getter, setter)
        self._parameters: dict[str, tuple[Any, str, str]] = {}

        # Make module movable and selectable
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)

    # === UI Construction Helpers ===
    @staticmethod
    def _create_controls_container() -> QWidget:
        """Create the standard controls container widget.

        This creates a transparent QWidget ready to have a layout added.
        Reduces boilerplate in module implementations.

        Returns:
            Configured QWidget ready for layout

        Example:
            ```python
            self.controls_widget = self._create_controls_container()
            layout = self._create_standard_layout()
            # ... add widgets to layout ...
            self.controls_widget.setLayout(layout)
            self.proxy = self._add_controls_to_module(self.controls_widget)
            ```
        """
        widget = QWidget()
        widget.setStyleSheet("background: transparent;")
        return widget

    def _add_controls_to_module(self, controls_widget: QWidget) -> QGraphicsProxyWidget:
        """Add controls widget to the module as a proxy.

        This handles the boilerplate of creating a QGraphicsProxyWidget and
        positioning it below the title bar.

        Args:
            controls_widget: The widget containing controls

        Returns:
            The proxy widget (for further customization if needed)

        Example:
            ```python
            self.proxy = self._add_controls_to_module(self.controls_widget)
            ```
        """
        proxy = QGraphicsProxyWidget(self)
        proxy.setWidget(controls_widget)
        proxy.setPos(0, self.TITLE_BAR_HEIGHT)
        return proxy

    @staticmethod
    def _create_standard_layout(spacing: int = 5) -> QVBoxLayout:
        """Create a standard vertical layout with default margins.

        Args:
            spacing: Spacing between widgets (default: 5)

        Returns:
            Configured layout

        Example:
            ```python
            layout = self._create_standard_layout()
            layout.addWidget(self.knob1)
            layout.addWidget(self.knob2)
            ```
        """
        layout = QVBoxLayout()
        layout.setContentsMargins(spacing, spacing, spacing, spacing)
        layout.setSpacing(spacing)
        return layout

    # === Parameter Management ===

    def register_parameter(
        self,
        name: str,
        widget: Any,
        getter: str = "get_value",
        setter: str = "set_value",
    ):
        """Register a parameter for automatic get/set in presets.

        This allows automatic parameter management without needing to override
        get_parameters() and set_parameters() in every module.

        Args:
            name: Parameter name for presets
            widget: Widget with value (Knob, Slider, ComboBox, etc.)
            getter: Method name to get value (default: "get_value")
            setter: Method name to set value (default: "set_value")

        Example:
            ```python
            self.freq_knob = Knob("Freq", 20, 2000, 440)
            self.register_parameter("frequency", self.freq_knob)

            # Now get_parameters() and set_parameters() work automatically!
            ```
        """
        self._parameters[name] = (widget, getter, setter)

    def get_parameters(self) -> dict[str, Any]:
        """Get all registered parameters automatically.

        Override this if you need custom parameter handling,
        or use register_parameter() for automatic handling.

        Returns:
            Dictionary of parameter names to values
        """
        params = {}
        for name, (widget, getter, _) in self._parameters.items():
            if hasattr(widget, getter):
                params[name] = getattr(widget, getter)()
        return params

    def set_parameters(self, params: dict[str, Any]):
        """Set all registered parameters automatically.

        Override this if you need custom parameter handling,
        or use register_parameter() for automatic handling.

        Args:
            params: Dictionary of parameter names to values
        """
        for name, value in params.items():
            if name in self._parameters:
                widget, _, setter = self._parameters[name]
                if hasattr(widget, setter):
                    getattr(widget, setter)(value)

    # === Naming ===

    def set_custom_name(self, name: str):
        """Set a custom name for this module instance.

        Args:
            name: Custom name to display
        """
        self.custom_name = name
        self.update()  # Trigger repaint

    def get_custom_name(self) -> str:
        """Get the custom name for this module.

        Returns:
            Custom name, or empty string if not set
        """
        return self.custom_name

    def get_display_name(self) -> str:
        """Get the name to display (custom name or module title).

        Returns:
            Name to display
        """
        return self.custom_name if self.custom_name else self.module_title

    # === Port Management ===

    def boundingRect(self) -> QRectF:
        """Return the bounding rectangle of the module."""
        return QRectF(0, 0, self.module_width, self.module_height)

    def paint(self, painter: QPainter, option, widget=None):
        """Paint the module."""
        rect = self.boundingRect()

        # Module background with gradient
        gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        gradient.setColorAt(0, self.module_color.lighter(120))
        gradient.setColorAt(1, self.module_color)

        painter.setBrush(QBrush(gradient))

        # Border
        if self.isSelected():
            painter.setPen(QPen(QColor(255, 200, 0), 3))
        else:
            painter.setPen(QPen(QColor(30, 30, 30), 2))

        painter.drawRoundedRect(rect, 8, 8)

        # Title bar (taller to fit more info)
        title_bar_height = 50 if self.custom_name else 42
        title_rect = QRectF(0, 0, self.module_width, title_bar_height)
        painter.fillRect(title_rect, QColor(30, 30, 30, 200))

        from PyQt6.QtGui import QFont

        # Module type (always shown at top)
        type_rect = QRectF(0, 3, self.module_width, 14)
        painter.setPen(QColor(150, 150, 150))
        font_type = MODULE_TYPE_FONT
        painter.setFont(font_type)
        painter.drawText(type_rect, Qt.AlignmentFlag.AlignCenter, self.module_title)

        # Custom name (if set, shown prominently)
        if self.custom_name:
            name_rect = QRectF(0, 16, self.module_width, 18)
            painter.setPen(QColor(255, 255, 100))
            font_name = QFont("Arial", 10, QFont.Weight.Bold)
            painter.setFont(font_name)
            painter.drawText(name_rect, Qt.AlignmentFlag.AlignCenter, self.custom_name)

            # Component type below custom name
            if self.module_category:
                comp_rect = QRectF(0, 34, self.module_width, 12)
                painter.setPen(QColor(130, 130, 130))
                font_comp = MODULE_CATEGORY_FONT
                painter.setFont(font_comp)
                painter.drawText(
                    comp_rect,
                    Qt.AlignmentFlag.AlignCenter,
                    f"[{self.module_category}]",
                )
        else:
            # No custom name - show component type
            if self.module_category:
                comp_rect = QRectF(0, 20, self.module_width, 18)
                painter.setPen(QColor(200, 200, 200))
                font_comp = MODULE_CATEGORY_FONT
                painter.setFont(font_comp)
                painter.drawText(
                    comp_rect,
                    Qt.AlignmentFlag.AlignCenter,
                    f"({self.module_category})",
                )

        # Draw port labels
        painter.setPen(QColor(220, 220, 220))
        font_port = QFont("Arial", 6)
        painter.setFont(font_port)

        # Input port labels (on the left)
        for port in self.input_ports:
            port_y = port.pos().y()
            label_rect = QRectF(5, port_y - 6, 50, 12)
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignLeft, port.port_name)

        # Output port labels (on the right)
        for port in self.output_ports:
            port_y = port.pos().y()
            label_rect = QRectF(self.module_width - 55, port_y - 6, 50, 12)
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignRight, port.port_name)

    def add_input_port(self, name: str) -> Port:
        """Add an input port to the module.

        Args:
            name: Name of the input port

        Returns:
            The created port
        """
        port = Port("input", name, self, len(self.input_ports))
        port.setParentItem(self)
        self.input_ports.append(port)
        self._update_port_positions()
        return port

    def add_output_port(self, name: str) -> Port:
        """Add an output port to the module.

        Args:
            name: Name of the output port

        Returns:
            The created port
        """
        port = Port("output", name, self, len(self.output_ports))
        port.setParentItem(self)
        self.output_ports.append(port)
        self._update_port_positions()
        return port

    def _update_port_positions(self):
        """Update the positions of all ports."""
        # Position input ports on the left side
        input_spacing = self.module_height / (len(self.input_ports) + 1)
        for i, port in enumerate(self.input_ports):
            y = input_spacing * (i + 1)
            port.setPos(-port.radius, y)

        # Position output ports on the right side
        output_spacing = self.module_height / (len(self.output_ports) + 1)
        for i, port in enumerate(self.output_ports):
            y = output_spacing * (i + 1)
            port.setPos(self.module_width + port.radius, y)

    # === Event Handling ===

    def contextMenuEvent(self, event):
        """Handle right-click context menu."""
        from PyQt6.QtWidgets import QMenu, QInputDialog

        menu = QMenu()
        rename_action = menu.addAction("Rename...")
        delete_action = menu.addAction("Delete")

        action = menu.exec(event.screenPos())

        if action == rename_action:
            # Show rename dialog
            current_name = self.get_display_name()
            new_name, ok = QInputDialog.getText(
                None, "Rename Module", "Enter new name:", text=current_name
            )
            if ok:
                self.set_custom_name(new_name)

        elif action == delete_action:
            # Delete this module
            if self.scene():
                # Remove connected cables first
                cables_to_remove = []
                for port in self.input_ports + self.output_ports:
                    cables_to_remove.extend(port.cables[:])

                for cable in cables_to_remove:
                    cable.remove()

                # Remove the module
                self.scene().removeItem(self)

    def mousePressEvent(self, event):
        """Handle mouse press to enable dragging from anywhere on the module."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Check if we clicked on a port (ports handle their own events)
            for port in self.input_ports + self.output_ports:
                port_rect = port.boundingRect().translated(port.pos())
                if port_rect.contains(event.pos()):
                    # Let the port handle it
                    super().mousePressEvent(event)
                    return

            # Check if we're in the title bar area (always draggable)
            title_bar_height = 50 if self.custom_name else 42
            if event.pos().y() <= title_bar_height:
                # Title bar click - always allow dragging
                self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
                super().mousePressEvent(event)
                return

            # For other areas, check if we clicked on a child widget
            # If not on a child widget, enable dragging
            child_item = (
                self.scene().itemAt(
                    self.mapToScene(event.pos()), self.scene().views()[0].transform()
                )
                if self.scene()
                else None
            )

            # If we clicked directly on the module (not on child controls), enable
            # dragging
            if child_item == self:
                self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move for dragging."""
        # Ensure dragging works even over child widgets
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        super().mouseReleaseEvent(event)

    def update_component(self):
        """Update the audio component with current parameter values."""
        if self.component:
            # Subclasses should implement parameter updates
            pass

    def create_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ) -> Any:
        raise NotImplementedError

    @property
    def module_category(self) -> ModuleCategory:
        raise NotImplementedError()

    @property
    def module_title(self) -> str:
        raise NotImplementedError()
