"""Base module widget for the modular synth interface."""

from __future__ import annotations

import contextlib
import logging
from abc import ABCMeta
from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsProxyWidget,
    QGraphicsWidget,
    QVBoxLayout,
    QWidget,
)

from sonicrack.gui.dialogs.module_info_dialog import ModuleInfoDialog
from sonicrack.gui.widgets.port_widget import PortWidget
from sonicrack.patching.module import AudioModule
from sonicrack.runtime.helpers import silence, write_silence_if_disconnected

logger = logging.getLogger(__name__)


# Create a compatible metaclass that combines QGraphicsWidget's metaclass with ABCMeta
class ModuleWidgetMeta(type(QGraphicsWidget), ABCMeta):  # type: ignore[misc]
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
    TITLE_BAR_HEIGHT = 44
    TITLE_BAR_HEIGHT_WITH_NAME = 54
    BORDER_RADIUS = 8
    SELECTION_BORDER_WIDTH = 3
    NORMAL_BORDER_WIDTH = 2
    POWER_BUTTON_SIZE = 18

    # Colors
    COLOR_SELECTION_BORDER = QColor(255, 200, 0)
    COLOR_NORMAL_BORDER = QColor(12, 14, 16)
    COLOR_TITLE_BAR_BG = QColor(18, 20, 23, 235)
    COLOR_MODULE_TYPE = QColor(232, 236, 240)
    COLOR_CUSTOM_NAME = QColor(255, 255, 100)
    COLOR_CATEGORY = QColor(160, 168, 174)
    COLOR_CATEGORY_NO_NAME = QColor(200, 200, 200)
    COLOR_PANEL_TOP = QColor(46, 50, 55)
    COLOR_PANEL_BOTTOM = QColor(26, 29, 33)
    COLOR_INACTIVE_OVERLAY = QColor(0, 0, 0, 105)

    # Signals
    parameter_changed = pyqtSignal(str, object)  # (param_name, value)

    # Runtime classification used by AudioEngine/render specs. Subclasses with
    # DSP behavior should override this with a registered kind from core.runtime.
    runtime_kind = "unknown"

    # Visual-only sink modules set this to False so they can receive rendered
    # buffers without being treated as processors inside the graph.
    is_processing_module = True

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
        self.is_active = True

        # Ports
        self.input_ports: list[PortWidget] = []
        self.output_ports: list[PortWidget] = []

        # Parameter registry for automatic get/set (widget, getter, setter)
        self._parameters: dict[str, tuple[Any, str, str]] = {}
        self._parameter_values: dict[str, Any] = {}
        self.parameter_changed.connect(self._cache_parameter_value)

        # Make module movable and selectable
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)

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
        widget.setStyleSheet("""
            QWidget {
                background: transparent;
                color: #edf1f5;
            }
            QLabel {
                color: #edf1f5;
                background: transparent;
                font-size: 11px;
            }
            QComboBox {
                color: #f3f6f8;
                background-color: #262c33;
                border: 1px solid #55606d;
                border-radius: 4px;
                padding: 3px 6px;
                min-height: 22px;
            }
            QComboBox QAbstractItemView {
                color: #f3f6f8;
                background-color: #20252b;
                selection-background-color: #2d6f9f;
            }
            QSlider::groove:horizontal {
                height: 5px;
                background: #4b5561;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #5ec4ff;
                border: 1px solid #101418;
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            """)
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
        controls_widget.setFixedWidth(self.module_width)
        proxy = QGraphicsProxyWidget(self)
        proxy.setWidget(controls_widget)
        proxy.setPos(0, self._title_bar_height())
        return proxy

    def _create_standard_layout(self, spacing: int = 10) -> QVBoxLayout:
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
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(spacing)
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        return layout

    def _begin_controls(self, *, spacing: int = 10) -> QVBoxLayout:
        """Create the controls container and return its layout.

        Call after ports are registered so ``_create_standard_layout`` can
        build port widgets. Finish with :meth:`_finish_controls`.
        """
        self.controls_widget = self._create_controls_container()
        return self._create_standard_layout(spacing=spacing)

    def _finish_controls(self, layout: QVBoxLayout) -> None:
        """Attach the finished controls layout to the module."""
        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

    def _install_sample_rate_listener(
        self,
        callback: Callable[[int], None] | None = None,
    ) -> None:
        """Listen for global sample-rate changes and clean up on destroy.

        Args:
            callback: Optional handler. Defaults to
                :meth:`_on_global_sample_rate_changed`.
        """
        from sonicrack.config.audio_config import audio_config

        listener = (
            callback if callback is not None else self._on_global_sample_rate_changed
        )
        previous = getattr(self, "_sample_rate_listener", None)
        if previous is not None and previous is not listener:
            audio_config.remove_sample_rate_listener(previous)

        self._sample_rate_listener = listener
        audio_config.add_sample_rate_listener(listener)
        if not getattr(self, "_sample_rate_cleanup_connected", False):
            self.destroyed.connect(self._cleanup_audio_config_listeners)
            self._sample_rate_cleanup_connected = True

    def _cleanup_audio_config_listeners(self, *_args: object) -> None:
        """Remove registered global sample-rate listeners during teardown."""
        from sonicrack.config.audio_config import audio_config

        listener = getattr(self, "_sample_rate_listener", None)
        if listener is not None:
            audio_config.remove_sample_rate_listener(listener)
            self._sample_rate_listener = None

    def _on_global_sample_rate_changed(self, new_sample_rate: int) -> None:
        """Default sample-rate handler: rebuild via ``create_engine_component``.

        Subclasses with custom rebuild logic should override this method (or
        pass a dedicated callback to :meth:`_install_sample_rate_listener`).
        """
        del new_sample_rate
        create = getattr(self, "create_engine_component", None)
        if callable(create):
            self.component = create()

    def _write_silence(self, num_samples: int, output_port: Any | None = None) -> None:
        """Write a silent buffer to the module output port."""
        port = (
            output_port
            if output_port is not None
            else getattr(self, "out_port", None)
        )
        if port is None:
            return
        port.write(silence(num_samples))

    def _require_input_or_silence(
        self,
        num_samples: int,
        *,
        input_port: Any | None = None,
        output_port: Any | None = None,
    ) -> bool:
        """Write silence and return True when the required audio input is missing."""
        in_port = (
            input_port if input_port is not None else getattr(self, "in_port", None)
        )
        out_port = (
            output_port
            if output_port is not None
            else getattr(self, "out_port", None)
        )
        if in_port is None or out_port is None:
            return False
        return write_silence_if_disconnected(in_port, out_port, num_samples)

    def bind_parameter_knob(
        self,
        knob: Any,
        param_name: str,
        *,
        value_label: Any | None = None,
        format_value: Callable[[float], str] | None = None,
        on_change: Callable[[float], None] | None = None,
    ) -> None:
        """Connect a knob to ``parameter_changed`` (and optional value label)."""

        def _handler(*_args: object) -> None:
            value = knob.get_value()
            if value_label is not None and format_value is not None:
                value_label.setText(format_value(value))
            if on_change is not None:
                on_change(value)
            self.parameter_changed.emit(param_name, value)

        knob.value_changed.connect(_handler)

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
        if hasattr(widget, getter):
            self._parameter_values[name] = getattr(widget, getter)()

    def get_parameters(self) -> dict[str, Any]:
        """Get all registered parameters automatically.

        Override this if you need custom parameter handling,
        or use register_parameter() for automatic handling.

        Returns:
            Dictionary of parameter names to values
        """
        params = {"active": self.is_active}
        for name in self._parameters:
            if name in self._parameter_values:
                params[name] = self._parameter_values[name]
        return params

    def _cache_parameter_value(self, name: str, value: Any) -> None:
        if name in self._parameters:
            self._parameter_values[name] = value

    def set_parameters(self, params: dict[str, Any]):
        """Set all registered parameters automatically.

        Override this if you need custom parameter handling,
        or use register_parameter() for automatic handling.

        Args:
            params: Dictionary of parameter names to values
        """
        active_value = params.get("active")
        if isinstance(active_value, bool):
            self.set_active(active_value)

        for name, value in params.items():
            if name == "active":
                continue
            if name in self._parameters:
                widget, _, setter = self._parameters[name]
                if hasattr(widget, setter):
                    getattr(widget, setter)(value)
                self._parameter_values[name] = value

    # === Active / Bypass State ===

    def set_active(self, active: bool) -> None:
        """Set whether this module participates in patch compilation."""
        if self.is_active == active:
            return
        self.is_active = active
        controls_widget = getattr(self, "controls_widget", None)
        if controls_widget is not None:
            controls_widget.setEnabled(active)
        if not active:
            self._clear_output_ports()
        self.invalidate_cache()
        self.update()
        self._notify_patch_changed()

    def toggle_active(self) -> None:
        """Toggle the module active/bypassed state."""
        self.set_active(not self.is_active)

    def _notify_patch_changed(self) -> None:
        """Notify the main window that module state changed."""
        scene = self.scene()
        if scene is None:
            return

        for view in scene.views():
            main_window = view.window()
            if main_window is None:
                continue
            if hasattr(main_window, "_mark_patch_modified"):
                main_window._mark_patch_modified()
            if hasattr(main_window, "_restart_output_playback"):
                main_window._restart_output_playback()
            break

    def invalidate_cache(self):
        """Hook for modules that own per-cycle runtime state."""

    def _clear_output_ports(self) -> None:
        """Clear visible output values when a module is bypassed."""
        for port in self.output_ports:
            port.write(0.0)

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

    def _title_bar_height(self) -> int:
        """Return the current title bar height."""
        if self.custom_name:
            return self.TITLE_BAR_HEIGHT_WITH_NAME
        return self.TITLE_BAR_HEIGHT

    def _power_button_rect(self) -> QRectF:
        """Return the title-bar power button hit rectangle."""
        margin = 10
        size = self.POWER_BUTTON_SIZE
        y = (self._title_bar_height() - size) / 2
        return QRectF(margin, y, size, size)

    def _tooltip_text_at(self, pos: QPointF) -> str:
        """Return context-sensitive tooltip text for the module surface."""
        if self._power_button_rect().contains(pos):
            return "Click the power button to bypass this module."
        if 0 <= pos.y() <= self._title_bar_height():
            return self.metadata.description
        return ""

    def _update_tooltip_at(self, pos: QPointF) -> None:
        """Update the native Qt tooltip for the current hover position."""
        self.setToolTip(self._tooltip_text_at(pos))

    def shape(self):
        """Return the shape for collision detection.

        This ensures the entire module area is clickable and draggable.
        """
        from PyQt6.QtGui import QPainterPath

        path = QPainterPath()
        path.addRect(self.boundingRect())
        return path

    def paint(self, painter: QPainter | None, option, widget=None):
        """Paint the module."""
        if painter is None:
            return

        rect = self.boundingRect()

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Rack panel background with subtle vertical shading.
        gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        gradient.setColorAt(0, self.COLOR_PANEL_TOP)
        gradient.setColorAt(1, self.COLOR_PANEL_BOTTOM)

        painter.setBrush(QBrush(gradient))

        # Border
        if self.isSelected():
            painter.setPen(
                QPen(self.COLOR_SELECTION_BORDER, self.SELECTION_BORDER_WIDTH)
            )
        else:
            painter.setPen(QPen(self.COLOR_NORMAL_BORDER, self.NORMAL_BORDER_WIDTH))

        painter.drawRoundedRect(rect, self.BORDER_RADIUS, self.BORDER_RADIUS)

        # Left accent rail gives each module family a rack identity.
        accent_rect = QRectF(0, 0, 6, self.module_height)
        accent_color = self.module_color if self.is_active else QColor(82, 86, 90)
        painter.fillRect(accent_rect, accent_color)

        # Title bar.
        title_bar_height = self._title_bar_height()
        title_rect = QRectF(0, 0, self.module_width, title_bar_height)
        painter.fillRect(title_rect, self.COLOR_TITLE_BAR_BG)
        painter.setPen(QPen(self.module_color.darker(105), 1))
        painter.drawLine(
            8,
            title_bar_height - 1,
            self.module_width - 8,
            title_bar_height - 1,
        )

        # Power icon and status lamp.
        power_rect = self._power_button_rect()
        lamp_color = QColor(100, 230, 140) if self.is_active else QColor(92, 96, 100)
        painter.setPen(QPen(QColor(8, 10, 12), 1))
        painter.setBrush(lamp_color)
        painter.drawEllipse(power_rect)
        painter.setPen(
            QPen(
                QColor(245, 248, 245),
                2,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
            )
        )
        center = power_rect.center()
        painter.drawLine(
            QPointF(center.x(), power_rect.top() + 4),
            QPointF(center.x(), center.y() + 1),
        )
        painter.drawArc(
            power_rect.adjusted(4, 5, -4, -3),
            int(205 * 16),
            int(130 * 16),
        )

        # Module type (always shown at top)
        type_rect = QRectF(34, 5, self.module_width - 44, 16)
        title_color = (
            self.COLOR_MODULE_TYPE if self.is_active else QColor(150, 154, 158)
        )
        painter.setPen(title_color)
        font_type = MODULE_TYPE_FONT
        painter.setFont(font_type)
        painter.drawText(type_rect, Qt.AlignmentFlag.AlignLeft, self.metadata.title)

        # Category label.
        category_rect = QRectF(34, 21, self.module_width - 44, 13)
        painter.setPen(self.COLOR_CATEGORY)
        painter.setFont(MODULE_CATEGORY_FONT)
        painter.drawText(
            category_rect,
            Qt.AlignmentFlag.AlignLeft,
            self.metadata.category.value.upper(),
        )

        # Custom name (if set, shown prominently)
        if self.custom_name:
            name_rect = QRectF(34, 35, self.module_width - 44, 16)
            painter.setPen(self.COLOR_CUSTOM_NAME)
            font_name = QFont("Arial", 10, QFont.Weight.Bold)
            painter.setFont(font_name)
            painter.drawText(name_rect, Qt.AlignmentFlag.AlignLeft, self.custom_name)

        if not self.is_active:
            painter.fillRect(
                rect.adjusted(6, title_bar_height, 0, 0),
                self.COLOR_INACTIVE_OVERLAY,
            )

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

    def _refresh_attached_cables(self) -> None:
        """Keep cable geometry in sync after this module moves or resizes."""
        seen: set[int] = set()
        for port in self.input_ports + self.output_ports:
            for cable in port.cables:
                cable_id = id(cable)
                if cable_id in seen:
                    continue
                seen.add(cable_id)
                refresh = getattr(cable, "refresh_geometry", None)
                if callable(refresh):
                    refresh()
                else:
                    cable.update()

    def itemChange(self, change, value):  # noqa: N802 - Qt API
        """Update attached cables when the module is moved on the canvas."""
        result = super().itemChange(change, value)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._refresh_attached_cables()
        return result

    # === Event Handling ===

    def contextMenuEvent(self, event):
        """Handle right-click context menu."""
        from PyQt6.QtWidgets import QInputDialog, QMenu

        menu = QMenu()
        active_action = menu.addAction("Bypass" if self.is_active else "Activate")
        rename_action = menu.addAction("Rename...")
        delete_action = menu.addAction("Delete")
        info_action = menu.addAction("Module Info...")

        action = menu.exec(event.screenPos())

        if action == active_action:
            self.toggle_active()

        elif action == rename_action:
            # Show rename dialog
            current_name = self.get_display_name()
            new_name, ok = QInputDialog.getText(
                None, "Rename Module", "Enter new name:", text=current_name
            )
            if ok:
                self.set_custom_name(new_name)

        elif action == delete_action:
            self.delete_from_patch()

        elif action == info_action:
            # Show module info dialog

            dialog = ModuleInfoDialog(self)
            dialog.exec()

    def mousePressEvent(self, event):
        """Handle mouse press to enable dragging from anywhere on the module."""
        if event.button() == Qt.MouseButton.LeftButton:
            if self._power_button_rect().contains(event.pos()):
                self.toggle_active()
                event.accept()
                return

            # Check if we clicked on a port (ports handle their own events)
            for port in self.input_ports + self.output_ports:
                port_rect = port.boundingRect().translated(port.pos())
                if port_rect.contains(event.pos()):
                    # Let the port handle it
                    super().mousePressEvent(event)
                    return

            # Check if we're in the title bar area (always draggable)
            if event.pos().y() <= self._title_bar_height():
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

    def hoverEnterEvent(self, event):
        """Set the tooltip before Qt's native tooltip delay starts."""
        self._update_tooltip_at(event.pos())
        super().hoverEnterEvent(event)

    def hoverMoveEvent(self, event):
        """Keep the tooltip scoped to the current header subregion."""
        self._update_tooltip_at(event.pos())
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        """Clear region-specific tooltip text when leaving the module."""
        self.setToolTip("")
        super().hoverLeaveEvent(event)

    def delete_from_patch(self) -> None:
        """Remove this module using the same lifecycle path as canvas deletion.

        Cables are removed through ``PatchCanvas.delete_cable`` so remaining
        neighbor modules receive ``cable_disconnected`` and can update
        connection-dependent state (e.g. modulated components).
        """
        scene = self.scene()
        if scene is None:
            return

        from sonicrack.gui.widgets.patch_canvas import PatchCanvas

        canvas = scene.parent()
        is_canvas = isinstance(canvas, PatchCanvas)

        # Collect unique cables attached to this module.
        cables_to_remove: list = []
        seen_cable_ids: set[int] = set()
        for port in self.input_ports + self.output_ports:
            with contextlib.suppress(RuntimeError, AttributeError):
                for cable in port.cables[:]:
                    cable_id = id(cable)
                    if cable_id not in seen_cable_ids:
                        seen_cable_ids.add(cable_id)
                        cables_to_remove.append(cable)

        # Disconnect each cable and notify peers before removing the module.
        for cable in cables_to_remove:
            with contextlib.suppress(RuntimeError, AttributeError):
                if is_canvas:
                    # Emits cable_disconnected so neighbors demote modulated state.
                    canvas.delete_cable(cable, emit_signal=True)
                else:
                    cable.remove()

        # Clear all port data to prevent stale audio.
        for port in self.input_ports + self.output_ports:
            if hasattr(port, "port") and port.port:
                with contextlib.suppress(RuntimeError, AttributeError):
                    port.port.clear()

        # Remove the module from the scene before notifying the main window so
        # playback recompilation sees the final canvas state.
        if scene is not None:
            with contextlib.suppress(RuntimeError):
                scene.removeItem(self)

        if is_canvas:
            canvas.module_deleted.emit(self)

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ) -> Any:
        raise NotImplementedError

    def get_runtime_spec(self):
        """Return this module's runtime declaration for the graph renderer."""
        from sonicrack.runtime.specs import RuntimeModuleSpec

        return RuntimeModuleSpec(
            kind=self.runtime_kind,
            processor=self.process_runtime,
            input_names=tuple(self.inputs),
            output_names=tuple(self.outputs),
            parameter_names=tuple(self._parameters),
        )

    def process_runtime(self, num_samples: int, parameters) -> None:
        """Default runtime no-op for passive sinks and unknown modules."""
        del num_samples, parameters
