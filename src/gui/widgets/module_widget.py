"""Base module widget for the modular synth interface."""

from __future__ import annotations

import logging
from abc import ABCMeta
from typing import Any

from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QLinearGradient, QFont
from PyQt6.QtWidgets import (
    QGraphicsWidget,
    QGraphicsItem,
    QWidget,
    QVBoxLayout,
    QGraphicsProxyWidget,
)

from src.gui.core.module import AudioModule
from src.gui.dialogs.module_info_dialog import ModuleInfoDialog
from src.gui.widgets.port_widget import PortWidget

logger = logging.getLogger(__name__)


# Create a compatible metaclass that combines QGraphicsWidget's metaclass with ABCMeta
class ModuleWidgetMeta(type(QGraphicsWidget), ABCMeta):
    """Combined metaclass for QGraphicsWidget and ABC."""

    pass


# Visual constants
MODULE_TYPE_FONT = QFont("Arial", 8, QFont.Weight.Bold)
MODULE_CATEGORY_FONT = QFont("Arial", 7)


class ModuleWidget(QGraphicsWidget, AudioModule, metaclass=ModuleWidgetMeta):
    """Base class for all modular synth audio modules.

    Each module represents a component able to perform audio. This could be oscillators,
    filters, envelope, effects, ... or visualization modules.

    Each module contains visual controls and connection ports.

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

        self.custom_name = ""

        # Caching for pull-based architecture (Phase 3)
        self._cache_valid = False
        self._cached_samples = None
        self._cache_num_samples = 0

        # Ports
        self.input_ports: list[PortWidget] = []
        self.output_ports: list[PortWidget] = []

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

    def _create_standard_layout(self, spacing: int = 5) -> QVBoxLayout:
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
        self._create_portwidgets()

        layout = QVBoxLayout()
        layout.setContentsMargins(spacing, spacing, spacing, spacing)
        layout.setSpacing(spacing)
        return layout

    def _create_portwidgets(self):
        for in_port in self.inputs.values():
            port = PortWidget(in_port, self)
            port.setParentItem(self)
            self.input_ports.append(port)

        for out_port in self.outputs.values():
            port = PortWidget(out_port, self)
            port.setParentItem(self)
            self.output_ports.append(port)
        self._update_port_positions()

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

    # === Pull-Based Audio Processing (Caching) ===

    def ensure_samples_ready(self, num_samples: int):
        """Ensure audio samples are generated for this processing cycle.

        This method implements the pull-based architecture with caching:
        1. Check if samples are already cached for this cycle
        2. If not, call process() to generate samples
        3. Cache the result to prevent redundant processing

        This is called by Port.read() when downstream modules request samples.

        Args:
            num_samples: Number of samples to generate
        """
        # Check if cache is valid and has the right number of samples
        if self._cache_valid and self._cache_num_samples == num_samples:
            return  # Already generated for this cycle

        # Generate samples by calling process()
        self.process(num_samples)

        # Mark cache as valid
        self._cache_valid = True
        self._cache_num_samples = num_samples

    def invalidate_cache(self):
        """Invalidate the sample cache at the start of each audio cycle.

        This should be called by the audio engine at the start of each
        processing cycle to ensure all modules regenerate their samples.
        """
        self._cache_valid = False
        self._cache_num_samples = 0
        self._cached_samples = None

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
        metadata = self.metadata
        return self.custom_name if self.custom_name else metadata.title

    # === Port Management ===

    def boundingRect(self) -> QRectF:
        """Return the bounding rectangle of the module."""
        return QRectF(0, 0, self.module_width, self.module_height)

    def shape(self):
        """Return the shape for collision detection.

        This ensures the entire module area is clickable and draggable.
        """
        from PyQt6.QtGui import QPainterPath

        path = QPainterPath()
        path.addRect(self.boundingRect())
        return path

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

        # Module type (always shown at top)
        type_rect = QRectF(0, 3, self.module_width, 14)
        painter.setPen(QColor(150, 150, 150))
        font_type = MODULE_TYPE_FONT
        painter.setFont(font_type)
        painter.drawText(type_rect, Qt.AlignmentFlag.AlignCenter, self.metadata.title)

        # Custom name (if set, shown prominently)
        if self.custom_name:
            name_rect = QRectF(0, 16, self.module_width, 18)
            painter.setPen(QColor(255, 255, 100))
            font_name = QFont("Arial", 10, QFont.Weight.Bold)
            painter.setFont(font_name)
            painter.drawText(name_rect, Qt.AlignmentFlag.AlignCenter, self.custom_name)

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
        info_action = menu.addAction("Module Info...")

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
            scene = self.scene()
            if scene:
                # Store references before removing from scene
                from src.gui.patch_canvas import PatchCanvas

                canvas = scene.parent() if scene else None
                views = scene.views() if scene else []

                # Clear all port data to prevent stale audio
                for port in self.input_ports + self.output_ports:
                    try:
                        if hasattr(port, "port") and port.port:
                            port.port.clear()
                    except (RuntimeError, AttributeError):
                        pass  # Port might already be cleared or deleted

                # Collect all cables to remove
                cables_to_remove = []
                for port in self.input_ports + self.output_ports:
                    try:
                        cables_to_remove.extend(port.cables[:])
                    except (RuntimeError, AttributeError):
                        pass  # Port might be deleted

                # Remove cables silently (without triggering signals)
                # This prevents multiple patch recompilations during deletion
                for cable in cables_to_remove:
                    try:
                        # Disconnect the underlying Port data models
                        if cable.start_port and cable.end_port:
                            if hasattr(cable.start_port, "port") and hasattr(
                                cable.end_port, "port"
                            ):
                                if cable.start_port.port and cable.end_port.port:
                                    cable.start_port.port.disconnect(
                                        cable.end_port.port
                                    )

                        # Remove cable from UI
                        if cable.start_port:
                            cable.start_port.remove_cable(cable)
                        if cable.end_port:
                            cable.end_port.remove_cable(cable)
                        if cable.scene():
                            cable.scene().removeItem(cable)
                    except (RuntimeError, AttributeError) as e:
                        # Cable or port might already be deleted - this is okay
                        logger.debug(f"Ignoring error during cable cleanup: {e}")
                        pass

                # Remove the module from scene (after this, self.scene() becomes None)
                try:
                    scene.removeItem(self)
                except RuntimeError:
                    pass  # Already removed

                # Mark patch as modified and trigger playback restart
                if canvas and isinstance(canvas, PatchCanvas):
                    # Get main window to trigger patch update
                    for view in views:
                        main_window = view.window()
                        if hasattr(main_window, "_mark_patch_modified"):
                            main_window._mark_patch_modified()

                        # Stop and restart playback to refresh audio callback
                        # This ensures we're not using stale connections/components
                        if hasattr(main_window, "_restart_output_playback"):
                            logger.info(
                                "Module deleted - restarting playback to refresh audio"
                            )
                            main_window._restart_output_playback()
                        elif hasattr(main_window, "_start_output_playback"):
                            # Fallback: just call start (which should detect
                            # disconnections)
                            main_window._start_output_playback()
                        break

        elif action == info_action:
            # Show module info dialog

            dialog = ModuleInfoDialog(self)
            dialog.exec()

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

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ) -> Any:
        raise NotImplementedError
