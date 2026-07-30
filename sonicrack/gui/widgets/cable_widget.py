from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QPainter, QPainterPath, QPainterPathStroker, QPen
from PyQt6.QtWidgets import QGraphicsItem

from sonicrack.gui.widgets.signal_style import cable_color_for_signal
from sonicrack.patching.port import PortSignal, PortType

if TYPE_CHECKING:
    from sonicrack.gui.widgets.port_widget import PortWidget

# Padding for pen width, hover stroke, and cubic-curve bulge.
_GEOMETRY_MARGIN = 20.0
_HIT_STROKE_WIDTH = 15.0


class Cable(QGraphicsItem):
    """A cable connecting two ports.

    Cables route signals between module outputs and inputs. Stroke color
    follows the source port's signal kind (audio, V/Oct, gate, trigger, …).

    Geometry is cached so ``prepareGeometryChange()`` still sees the previous
    path after attached ports move (Qt only learns the old rect if we keep it
    until prepare is called).
    """

    def __init__(self, start_port: PortWidget, end_port: PortWidget | None = None):
        """Initialize a cable.

        Args:
            start_port: The output port widget where the cable starts
            end_port: The input port widget where the cable ends (can be None for
                dragging)
        """
        super().__init__()
        self.start_port = start_port
        self.end_port = end_port
        self.temp_end_pos: QPointF | None = None
        self.is_hovered = False  # Track hover state
        self._geom_start = QPointF()
        self._geom_end = QPointF()

        # Make cable selectable and interactive
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(
            Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton
        )
        self.setZValue(-1)  # Draw cables behind modules

        if start_port:
            start_port.add_cable(self)
        if end_port:
            end_port.add_cable(self)

        # Snapshot endpoints without notifying the scene (not yet inserted).
        self._store_live_endpoints()

    def set_end_port(self, port: PortWidget):
        """Set the end port of the cable.

        Args:
            port: The port widget to connect to
        """
        self.prepareGeometryChange()
        if self.end_port:
            self.end_port.remove_cable(self)
        self.end_port = port
        self.temp_end_pos = None
        port.add_cable(self)
        self._store_live_endpoints()
        self.update()

    def set_temp_end_pos(self, pos: QPointF):
        """Set a temporary end position while dragging."""
        self.prepareGeometryChange()
        self.temp_end_pos = pos
        self._store_live_endpoints()
        self.update()

    def refresh_geometry(self) -> None:
        """Invalidate scene geometry after attached ports move.

        Must keep the previous cached endpoints until ``prepareGeometryChange``
        finishes so Qt can clear the old cable trail.
        """
        self.prepareGeometryChange()
        self._store_live_endpoints()
        self.update()

    def _live_endpoints(self) -> tuple[QPointF, QPointF] | None:
        """Return current start/end positions in scene coordinates."""
        if not self.start_port:
            return None

        start = self.start_port.get_scene_pos()
        if self.end_port:
            end = self.end_port.get_scene_pos()
        elif self.temp_end_pos is not None:
            end = QPointF(self.temp_end_pos)
        else:
            end = QPointF(start)
        return start, end

    def _store_live_endpoints(self) -> None:
        """Copy live port positions into the geometry cache."""
        live = self._live_endpoints()
        if live is None:
            self._geom_start = QPointF()
            self._geom_end = QPointF()
            return
        self._geom_start, self._geom_end = live

    def boundingRect(self) -> QRectF:
        """Return the bounding rectangle of the cable (item/scene coords)."""
        rect = QRectF(self._geom_start, self._geom_end).normalized()
        margin = _GEOMETRY_MARGIN
        return rect.adjusted(-margin, -margin, margin, margin)

    def _curve_path(self, start: QPointF, end: QPointF) -> QPainterPath:
        """Build the cubic bezier used for painting and hit-testing."""
        path = QPainterPath()
        path.moveTo(start)

        ctrl_offset = abs(end.x() - start.x()) * 0.5
        ctrl1 = QPointF(start.x() + ctrl_offset, start.y())
        ctrl2 = QPointF(end.x() - ctrl_offset, end.y())
        path.cubicTo(ctrl1, ctrl2, end)
        return path

    def paint(self, painter: QPainter | None, option, widget=None):
        """Paint the cable as a curved line."""
        if painter is None or not self.start_port:
            return

        start = self.mapFromScene(self._geom_start)
        end = self.mapFromScene(self._geom_end)
        path = self._curve_path(start, end)

        selected = self.isSelected()
        color = cable_color_for_signal(
            self.signal_kind(),
            selected=selected,
            hovered=self.is_hovered,
        )
        width = 4 if selected or self.is_hovered else 3

        pen = QPen(color, width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        painter.setPen(pen)
        painter.drawPath(path)

    def signal_kind(self) -> PortSignal:
        """Return the signal kind used for this cable's color.

        Prefers the output/source port when both ends are known so the cable
        reflects what is actually being sent.
        """
        start = self.start_port
        end = self.end_port

        if start is not None and end is not None:
            if start.port_type == PortType.OUTPUT:
                return start.port.signal
            if end.port_type == PortType.OUTPUT:
                return end.port.signal
            return start.port.signal

        if start is not None:
            return start.port.signal
        if end is not None:
            return end.port.signal
        return PortSignal.UNKNOWN

    def shape(self) -> QPainterPath:
        """Return the shape for collision detection (wider than visual cable)."""
        if not self.start_port:
            return QPainterPath()

        start = self.mapFromScene(self._geom_start)
        end = self.mapFromScene(self._geom_end)
        path = self._curve_path(start, end)

        stroker = QPainterPathStroker()
        stroker.setWidth(_HIT_STROKE_WIDTH)
        stroker.setCapStyle(Qt.PenCapStyle.RoundCap)
        stroker.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        return stroker.createStroke(path)

    def remove(self):
        """Remove this cable from the scene and disconnect from ports.

        This method is defensive against race conditions and handles cases where
        ports might already be deleted or None.
        """
        # Store local references to avoid accessing potentially deleted objects
        start = self.start_port
        end = self.end_port

        # Disconnect the underlying Port data models FIRST
        # Check if ports still exist and have valid port models
        if (
            start
            and end
            and hasattr(start, "port")
            and hasattr(end, "port")
            and start.port
            and end.port
        ):
            # Port might have been deleted - this is okay during cleanup
            with contextlib.suppress(RuntimeError, AttributeError):
                start.port.disconnect(end.port)

        # Remove cable from port widget's cable list
        if start:
            with contextlib.suppress(RuntimeError, AttributeError):
                # Port might already be deleted
                start.remove_cable(self)

        if end:
            with contextlib.suppress(RuntimeError, AttributeError):
                # Port might be deleted
                end.remove_cable(self)

        # Remove from scene
        scene = self.scene()
        if scene:
            with contextlib.suppress(RuntimeError):
                # Already removed or scene deleted
                scene.removeItem(self)

        # Clear references to help garbage collection
        self.start_port = None
        self.end_port = None

    def hoverEnterEvent(self, event):
        """Handle mouse hover enter."""
        self.is_hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """Handle mouse hover leave."""
        self.is_hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def contextMenuEvent(self, event):
        """Handle right-click context menu."""
        from PyQt6.QtGui import QAction
        from PyQt6.QtWidgets import QMenu

        # Select this cable to highlight it visually
        self.setSelected(True)

        # Create context menu
        menu = QMenu()

        # Add delete action
        delete_action = QAction("Delete Connection", menu)
        delete_action.triggered.connect(self._on_delete_requested)
        menu.addAction(delete_action)

        # Show menu at cursor position
        # Get the view to show the menu
        if self.scene() and self.scene().views():
            view = self.scene().views()[0]
            menu.exec(view.mapToGlobal(view.mapFromScene(event.scenePos())))

        event.accept()

    def _on_delete_requested(self):
        """Handle delete request from context menu."""
        # Use centralized delete method
        from sonicrack.gui.widgets.patch_canvas import PatchCanvas

        scene = self.scene()
        parent = scene.parent() if scene is not None else None
        if isinstance(parent, PatchCanvas):
            canvas = parent
            canvas.delete_cable(self, emit_signal=True)
