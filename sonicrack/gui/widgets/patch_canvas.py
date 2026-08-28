"""Modular cable patching system for connecting audio components."""

from __future__ import annotations

import contextlib
import logging
from typing import cast

from PyQt6 import QtCore
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QColor, QCursor, QPainter, QPen, QWheelEvent
from PyQt6.QtWidgets import (
    QApplication,
    QGraphicsProxyWidget,
    QGraphicsScene,
    QGraphicsView,
    QMessageBox,
    QWidget,
)

from sonicrack.gui.modules.output.output import OutputModule
from sonicrack.gui.widgets.cable_widget import Cable
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.gui.widgets.port_widget import PortWidget
from sonicrack.patching.module import ModuleCategory
from sonicrack.patching.port import PortType, port_signals_compatible

logger = logging.getLogger(__name__)


class PatchCanvas(QGraphicsView):
    """The main canvas for the modular patching system.

    This is where modules are placed and connected with cables.
    """

    MIN_ZOOM = 0.25
    MAX_ZOOM = 3.0
    DEFAULT_ZOOM = 1.0
    ZOOM_STEP = 1.15

    # Emitted when a cable is connected
    cable_connected = QtCore.pyqtSignal(PortWidget, PortWidget)

    # Emitted when a cable is disconnected
    cable_disconnected = QtCore.pyqtSignal(PortWidget, PortWidget)

    # Emitted when a module is deleted
    module_deleted = QtCore.pyqtSignal(object)

    # Emitted when the view zoom factor changes (absolute scale)
    zoom_changed = QtCore.pyqtSignal(float)

    def __init__(self, parent=None):
        """Initialize the patch canvas."""
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # Canvas properties
        self.setSceneRect(-2000, -2000, 4000, 4000)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        # Prefer dirty-region updates; full-viewport mode was expensive during
        # pan/zoom/cable drag with many modules and bezier cables.
        self.setViewportUpdateMode(
            QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate
        )
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

        # Background — dark rack case; grid is painted in drawBackground.
        self.setBackgroundBrush(QColor(22, 24, 28))
        self._grid_minor = 24
        self._grid_major = 96
        self._grid_minor_color = QColor(36, 40, 46)
        self._grid_major_color = QColor(48, 54, 62)

        # Zoom state (absolute scale relative to identity transform)
        self._zoom_factor = self.DEFAULT_ZOOM

        # Middle-mouse pan state
        self._panning = False
        self._pan_start: QPoint | None = None

        # Cable dragging state
        self.dragging_cable: Cable | None = None
        self.drag_start_port: PortWidget | None = None

    @property
    def zoom_factor(self) -> float:
        """Current absolute zoom scale (1.0 = 100%)."""
        return self._zoom_factor

    def set_zoom(self, factor: float, *, anchor_under_mouse: bool = False) -> None:
        """Set the absolute zoom factor, clamped to MIN/MAX_ZOOM."""
        target = max(self.MIN_ZOOM, min(self.MAX_ZOOM, float(factor)))
        if abs(target - self._zoom_factor) < 1e-9:
            return

        if anchor_under_mouse:
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        else:
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

        # Scale relatively from the current zoom so anchors stay stable.
        relative = target / self._zoom_factor if self._zoom_factor else target
        self.scale(relative, relative)
        self._zoom_factor = target
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.zoom_changed.emit(self._zoom_factor)

    def drawBackground(self, painter: QPainter | None, rect) -> None:  # noqa: N802
        """Fill the case and draw a subtle rack grid."""
        if painter is None:
            return
        painter.fillRect(rect, QColor(22, 24, 28))
        left = int(rect.left()) - (int(rect.left()) % self._grid_minor)
        top = int(rect.top()) - (int(rect.top()) % self._grid_minor)
        right = int(rect.right())
        bottom = int(rect.bottom())

        minor_pen = QPen(self._grid_minor_color, 1)
        major_pen = QPen(self._grid_major_color, 1)
        x = left
        while x <= right:
            painter.setPen(major_pen if x % self._grid_major == 0 else minor_pen)
            painter.drawLine(x, top, x, bottom)
            x += self._grid_minor
        y = top
        while y <= bottom:
            painter.setPen(major_pen if y % self._grid_major == 0 else minor_pen)
            painter.drawLine(left, y, right, y)
            y += self._grid_minor

    def zoom_in(self) -> None:
        """Zoom in by one step."""
        self.set_zoom(self._zoom_factor * self.ZOOM_STEP)

    def zoom_out(self) -> None:
        """Zoom out by one step."""
        self.set_zoom(self._zoom_factor / self.ZOOM_STEP)

    def reset_zoom(self) -> None:
        """Reset zoom to 100% (identity transform)."""
        if abs(self._zoom_factor - self.DEFAULT_ZOOM) < 1e-9:
            return
        self.resetTransform()
        self._zoom_factor = self.DEFAULT_ZOOM
        self.zoom_changed.emit(self._zoom_factor)

    def _proxy_and_widget_at(
        self, view_pos: QPoint
    ) -> tuple[QGraphicsProxyWidget, QWidget] | None:
        """Return the proxy and deepest embedded widget under a view position.

        Modules host controls via ``QGraphicsProxyWidget``. When the cursor is
        over those controls, wheel events should go to them (knobs, step cells)
        instead of zooming the canvas.
        """
        item = self.itemAt(view_pos)
        while item is not None:
            if isinstance(item, QGraphicsProxyWidget):
                root = item.widget()
                if root is None:
                    return None
                scene_pos = self.mapToScene(view_pos)
                local = item.mapFromScene(scene_pos).toPoint()
                child = root.childAt(local)
                return item, (child if child is not None else root)
            item = item.parentItem()
        return None

    def _deliver_wheel_to_embedded(self, event: QWheelEvent) -> bool:
        """Forward a wheel event into an embedded control under the cursor.

        Returns True if a target widget accepted the event.
        """
        view_pos = event.position().toPoint()
        hit = self._proxy_and_widget_at(view_pos)
        if hit is None:
            return False

        proxy, target = hit
        embedded_root = proxy.widget()
        if embedded_root is None:
            return False

        # Embedded widgets often lack reliable global geometry; map through the
        # proxy/scene instead of mapFromGlobal.
        scene_pos = self.mapToScene(view_pos)
        root_pos = proxy.mapFromScene(scene_pos).toPoint()

        widget: QWidget | None = target
        while widget is not None:
            if widget is embedded_root:
                local_point = root_pos
            else:
                local_point = widget.mapFrom(embedded_root, root_pos)
            forwarded = QWheelEvent(
                QPointF(local_point),
                event.globalPosition(),
                event.pixelDelta(),
                event.angleDelta(),
                event.buttons(),
                event.modifiers(),
                event.phase(),
                event.inverted(),
            )
            QApplication.sendEvent(widget, forwarded)
            if forwarded.isAccepted():
                return True
            if widget is embedded_root:
                break
            widget = widget.parentWidget()
        return False

    def wheelEvent(self, event):  # noqa: N802 - Qt API
        """Zoom with the scroll wheel, or adjust controls under the cursor."""
        if event is None:
            return

        # Knobs / step cells / sliders under the cursor take priority over zoom.
        if self._deliver_wheel_to_embedded(event):
            event.accept()
            return

        delta = event.angleDelta().y()
        if delta == 0:
            # High-res trackpads may report pixel deltas instead.
            delta = event.pixelDelta().y()
        if delta > 0:
            self.set_zoom(self._zoom_factor * self.ZOOM_STEP, anchor_under_mouse=True)
            event.accept()
            return
        if delta < 0:
            self.set_zoom(self._zoom_factor / self.ZOOM_STEP, anchor_under_mouse=True)
            event.accept()
            return

        super().wheelEvent(event)

    def mousePressEvent(self, event):
        """Handle mouse press for pan, cable creation, and selection."""
        if event is None:
            return

        # Middle mouse button: enter pan mode
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            event.accept()
            return

        item = self._port_at(event.pos(), PortType.OUTPUT)
        if item is not None:
            # Start dragging a cable from this port
            self.drag_start_port = item
            self.dragging_cable = Cable(item)
            self._scene.addItem(self.dragging_cable)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move for pan and cable dragging."""
        if event is None:
            return

        if self._panning and self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            h_bar = self.horizontalScrollBar()
            v_bar = self.verticalScrollBar()
            if h_bar is not None:
                h_bar.setValue(h_bar.value() - delta.x())
            if v_bar is not None:
                v_bar.setValue(v_bar.value() - delta.y())
            event.accept()
            return

        if self.dragging_cable:
            pos = self.mapToScene(event.pos())
            self.dragging_cable.set_temp_end_pos(pos)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release for pan and cable connection."""
        if event is None:
            return

        if event.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            self._pan_start = None
            self.unsetCursor()
            event.accept()
            return

        if self.dragging_cable:
            item = self._port_at(event.pos(), PortType.INPUT)
            start_port = self.drag_start_port

            if start_port is None:
                self.dragging_cable.remove()
                self.dragging_cable = None
                event.accept()
                return

            if item is not None:
                # Check if trying to connect to the same module
                if item.parent_module == start_port.parent_module:
                    # Self-connection not allowed - show error
                    QMessageBox.warning(
                        self,
                        "Invalid Connection",
                        "Cannot connect a module's output to its own input.\n\n"
                        "Self-connections would create an infinite feedback loop.",
                    )
                    # Remove the invalid cable
                    self.dragging_cable.remove()
                elif self._find_cable(start_port, item) is not None:
                    QMessageBox.warning(
                        self,
                        "Invalid Connection",
                        "These ports are already connected.",
                    )
                    self.dragging_cable.remove()
                else:
                    # Check if this connection would create a cycle
                    cycle_info = self._would_create_cycle(start_port, item)
                    if cycle_info:
                        # Connection would create a cycle - show error
                        QMessageBox.warning(
                            self,
                            "Invalid Connection",
                            "Infinite Loop Detected. This connection would create an "
                            f"infinite feedback loop:\n\n{cycle_info}\n\n"
                            "Please check your connections and avoid creating cycles.",
                        )
                        # Remove the invalid cable
                        self.dragging_cable.remove()
                    elif not self._ports_are_compatible(start_port, item):
                        QMessageBox.warning(
                            self,
                            "Invalid Connection",
                            self._incompatible_connection_message(start_port, item),
                        )
                        self.dragging_cable.remove()
                    else:
                        # Valid connection
                        self.dragging_cable.set_end_port(item)
                        self.cable_connected.emit(start_port, item)
            else:
                # Invalid connection, remove cable
                self.dragging_cable.remove()

            self.dragging_cable = None
            self.drag_start_port = None
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def _would_create_cycle(
        self, start_port: PortWidget, end_port: PortWidget
    ) -> str | None:
        """Check if adding a connection would create a cycle.

        Args:
            start_port: The output port (source)
            end_port: The input port (destination)

        Returns:
            String describing the cycle if one would be created, None otherwise
        """
        source_module = start_port.parent_module
        dest_module = end_port.parent_module

        # Build graph of existing connections
        graph: dict[ModuleWidget, list[ModuleWidget]] = {}

        # Get all existing cables from the scene
        for item in self._scene.items():
            if isinstance(item, Cable) and item.start_port and item.end_port:
                src = item.start_port.parent_module
                dst = item.end_port.parent_module
                if src not in graph:
                    graph[src] = []
                graph[src].append(dst)

        # Add the proposed connection temporarily
        if source_module not in graph:
            graph[source_module] = []
        graph[source_module].append(dest_module)

        # Check if this creates a cycle using DFS
        visited: set[ModuleWidget] = set()
        rec_stack: set[ModuleWidget] = set()

        def dfs(module: ModuleWidget, current_path: list[ModuleWidget]) -> str | None:
            """Depth-first search to detect cycles."""
            visited.add(module)
            rec_stack.add(module)
            current_path.append(module)

            for neighbor in graph.get(module, []):
                if neighbor not in visited:
                    cycle = dfs(neighbor, current_path)
                    if cycle:
                        return cycle
                elif neighbor in rec_stack:
                    # Found a cycle! Build description
                    cycle_start = current_path.index(neighbor)
                    cycle_modules = current_path[cycle_start:] + [neighbor]
                    cycle_names = [m.metadata.title for m in cycle_modules]
                    return " -> ".join(cycle_names)

            current_path.pop()
            rec_stack.remove(module)
            return None

        # Check for cycles starting from the source module
        cycle_description = dfs(source_module, [])
        return cycle_description

    def keyPressEvent(self, event):
        """Handle key press for deleting cables and modules."""
        if event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            # Get selected items
            selected_items = self._scene.selectedItems()

            # Collect all cables and modules to delete
            cables_to_delete = []
            modules_to_delete = []

            for item in selected_items:
                if isinstance(item, Cable):
                    cables_to_delete.append(item)
                elif isinstance(item, ModuleWidget):
                    modules_to_delete.append(item)

            # Delete cables for selected modules FIRST (collect their cables)
            for module in modules_to_delete:
                for port in module.input_ports + module.output_ports:
                    # Add cables to deletion list (avoid duplicates)
                    for cable in port.cables[:]:
                        if cable not in cables_to_delete:
                            cables_to_delete.append(cable)

            # Remove cables first (port models disconnect in Cable.remove).
            # Collect pairs so every remaining neighbor is notified after cleanup.
            disconnected_ports: list[tuple[PortWidget, PortWidget]] = []
            for cable in cables_to_delete:
                if cable.start_port and cable.end_port:
                    disconnected_ports.append((cable.start_port, cable.end_port))
                cable.remove()

            # Delete all modules
            for module in modules_to_delete:
                # Clear port data to prevent stale audio
                for port in module.input_ports + module.output_ports:
                    with contextlib.suppress(RuntimeError, AttributeError):
                        # port might be already cleared
                        port.port.clear()

                # Emit signal that module is being deleted
                self.module_deleted.emit(module)

                # Remove the module itself
                with contextlib.suppress(RuntimeError):
                    # already removed
                    self._scene.removeItem(module)

            # Notify every disconnected pair so modulated modules (and peers)
            # update connection-dependent component state for all cables, not
            # only the first one.
            for start_port, end_port in disconnected_ports:
                with contextlib.suppress(RuntimeError, AttributeError):
                    if start_port and end_port:
                        self.cable_disconnected.emit(start_port, end_port)

            event.accept()
            return

        super().keyPressEvent(event)

    def add_module(self, module: ModuleWidget, pos: QPointF | None = None):
        """Add a module to the canvas.

        Args:
            module: The module widget to add
            pos: Optional position to place the module
        """
        self._scene.addItem(module)
        if pos:
            module.setPos(pos)
        else:
            # Place at center of view
            viewport = self.viewport()
            if viewport is None:
                raise RuntimeError("Patch canvas viewport is not available")
            center = self.mapToScene(viewport.rect().center())
            module.setPos(center)

    def delete_cable(self, cable: Cable, emit_signal: bool = True):
        """Delete a cable and optionally emit the disconnection signal.

        This is the single centralized method for cable deletion to ensure
        consistent behavior: always remove from scene BEFORE emitting signal.
        This ensures that when the patch recompiles, get_connections() won't
        find the deleted cable.

        Args:
            cable: The cable to delete
            emit_signal: Whether to emit cable_disconnected signal (default True)
        """
        if not cable:
            return

        # Store port references before removing
        start_port = cable.start_port
        end_port = cable.end_port

        # Remove cable from scene FIRST (before signal)
        # This is critical: if we emit the signal first, the recompilation
        # will still see this cable in get_connections()
        cable.remove()

        # Emit disconnection signal AFTER removal
        if emit_signal and start_port and end_port:
            self.cable_disconnected.emit(start_port, end_port)

    def create_connection(
        self, start_port: PortWidget, end_port: PortWidget
    ) -> Cable | None:
        """Create a cable connection between two ports.

        This is a helper method for programmatically creating cables,
        e.g., when loading presets or patches.

        Args:
            start_port: Output port (source)
            end_port: Input port (destination)

        Returns:
            The created Cable object, or None if connection is invalid.
            If the same pair is already cabled, returns the existing cable
            without emitting another connect signal.
        """
        # Validate ports
        if start_port.port_type != "output" or end_port.port_type != PortType.INPUT:
            logger.warning("Invalid port types for cable creation")
            return None

        # Check for self-connection
        if start_port.parent_module == end_port.parent_module:
            logger.warning("Cannot create self-connection")
            return None

        existing = self._find_cable(start_port, end_port)
        if existing is not None:
            logger.debug(
                "Connection already exists: %s -> %s",
                start_port.port_name,
                end_port.port_name,
            )
            return existing

        if not self._ports_are_compatible(start_port, end_port):
            logger.warning(
                "Skipping incompatible connection: %s",
                self._incompatible_connection_message(start_port, end_port),
            )
            return None

        cycle_info = self._would_create_cycle(start_port, end_port)
        if cycle_info:
            logger.warning(
                "Skipping cyclic connection (%s): %s -> %s",
                cycle_info,
                start_port.port_name,
                end_port.port_name,
            )
            return None

        # Create and add cable
        cable = Cable(start_port, end_port)
        self._scene.addItem(cable)

        # Emit signal
        self.cable_connected.emit(start_port, end_port)

        return cable

    def _find_cable(self, start_port: PortWidget, end_port: PortWidget) -> Cable | None:
        """Return an existing cable between the given ports, if any."""
        for item in self._scene.items():
            if (
                isinstance(item, Cable)
                and item.start_port is start_port
                and item.end_port is end_port
            ):
                return item
        return None

    @staticmethod
    def _ports_are_compatible(start_port: PortWidget, end_port: PortWidget) -> bool:
        return port_signals_compatible(start_port.port, end_port.port)

    @staticmethod
    def _incompatible_connection_message(
        start_port: PortWidget, end_port: PortWidget
    ) -> str:
        return (
            "Cannot connect incompatible signal types:\n\n"
            f"{start_port.parent_module.get_display_name()}:{start_port.port_name} "
            f"outputs {start_port.port.signal}\n"
            f"{end_port.parent_module.get_display_name()}:{end_port.port_name} "
            f"expects {end_port.port.signal}\n\n"
            "Use a matching pitch-CV, gate/trigger, control CV, or audio port."
        )

    def get_connections(self) -> list[tuple[PortWidget, PortWidget]]:
        """Get all cable connections in the canvas.

        Returns:
            List of (output_port, input_port) tuples
        """
        connections: list[tuple[PortWidget, PortWidget]] = []
        for item in self._scene.items():
            if isinstance(item, Cable) and item.start_port and item.end_port:
                connections.append((item.start_port, item.end_port))
        return connections

    def get_modules(self) -> list[ModuleWidget]:
        """Get all modules currently on the canvas.

        Returns:
            List of ModuleWidget instances
        """
        return [item for item in self._scene.items() if isinstance(item, ModuleWidget)]

    def get_modules_by_category(self, category: ModuleCategory) -> list[ModuleWidget]:
        """Get all modules of a specific category currently on the canvas.

        Args:
            category: The ModuleCategory to filter by

        Returns:
            List of ModuleWidget instances in the specified category
        """
        return [
            item for item in self.get_modules() if item.metadata.category == category
        ]

    def get_output_module(self) -> OutputModule | None:
        """Get the output module on the canvas, if any.

        Returns:
            The OutputModule instance, or None if not found
        """
        for module in self.get_modules():
            if module.metadata.category == ModuleCategory.OUTPUT:
                return cast(OutputModule, module)
        return None

    def clear_all(self):
        """Clear all modules and cables from the canvas.

        Cables and port models are fully disconnected before the scene is
        wiped so no ghost ``connected_to`` links keep old graphs alive.
        """
        # Disconnect every cable (port models + cable lists) before wipe.
        for item in list(self._scene.items()):
            if isinstance(item, Cable):
                with contextlib.suppress(RuntimeError, AttributeError):
                    item.remove()

        # Disconnect any remaining port model links and clear values.
        for module in self.get_modules():
            for port_widget in module.input_ports + module.output_ports:
                with contextlib.suppress(RuntimeError, AttributeError):
                    port_widget.port.disconnect()
                    port_widget.port.clear()

        # Then clear the scene
        self._scene.clear()
        self.dragging_cable = None
        self.drag_start_port = None
        self._panning = False
        self._pan_start = None

    def _port_at(
        self, view_pos, port_type: PortType | str | None = None
    ) -> PortWidget | None:
        """Return the topmost PortWidget under a view position, if any.

        Uses ``items()`` so ports under labels/proxies can still be hit.
        """
        for item in self.items(view_pos):
            if not isinstance(item, PortWidget):
                continue
            if port_type is None or item.port_type == port_type:
                return item
        return None
