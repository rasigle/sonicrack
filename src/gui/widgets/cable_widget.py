from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QPainter, QPen, QColor, QPainterPath
from PyQt6.QtWidgets import QGraphicsItem

if TYPE_CHECKING:
    from src.gui.widgets.port_widget import PortWidget


class Cable(QGraphicsItem):
    """A cable connecting two ports.

    Cables route audio signals between module outputs and inputs.

    This is a UI component that works with PortWidget (the visual representation).
    The actual connection logic is handled by PortModel (contained in PortWidget).
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

    def set_end_port(self, port: PortWidget):
        """Set the end port of the cable.

        Args:
            port: The port widget to connect to
        """
        if self.end_port:
            self.end_port.remove_cable(self)
        self.end_port = port
        self.temp_end_pos = None
        port.add_cable(self)
        self.update()

    def set_temp_end_pos(self, pos: QPointF):
        """Set a temporary end position while dragging."""
        self.temp_end_pos = pos
        self.update()

    def boundingRect(self) -> QRectF:
        """Return the bounding rectangle of the cable."""
        if not self.start_port:
            return QRectF()

        start = self.start_port.get_scene_pos()
        if self.end_port:
            end = self.end_port.get_scene_pos()
        elif self.temp_end_pos:
            end = self.temp_end_pos
        else:
            end = start

        return QRectF(start, end).normalized().adjusted(-10, -10, 10, 10)

    def paint(self, painter: QPainter, option, widget=None):
        """Paint the cable as a curved line."""
        if not self.start_port:
            return

        start = self.mapFromScene(self.start_port.get_scene_pos())
        if self.end_port:
            end = self.mapFromScene(self.end_port.get_scene_pos())
        elif self.temp_end_pos:
            end = self.mapFromScene(self.temp_end_pos)
        else:
            return

        # Draw a cubic bezier curve
        path = QPainterPath()
        path.moveTo(start)

        # Control points for smooth curve
        ctrl_offset = abs(end.x() - start.x()) * 0.5
        ctrl1 = QPointF(start.x() + ctrl_offset, start.y())
        ctrl2 = QPointF(end.x() - ctrl_offset, end.y())

        path.cubicTo(ctrl1, ctrl2, end)

        # Cable color based on selection and hover state
        if self.isSelected():
            color = QColor(255, 200, 0)  # Yellow when selected
            width = 4
        elif self.is_hovered:
            color = QColor(150, 150, 150)  # Lighter gray when hovered
            width = 4
        else:
            color = QColor(100, 100, 100)  # Normal gray
            width = 3

        pen = QPen(color, width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        painter.setPen(pen)
        painter.drawPath(path)

    def shape(self) -> QPainterPath:
        """Return the shape for collision detection (wider than visual cable)."""
        if not self.start_port:
            path = QPainterPath()
            return path

        start = self.mapFromScene(self.start_port.get_scene_pos())
        if self.end_port:
            end = self.mapFromScene(self.end_port.get_scene_pos())
        elif self.temp_end_pos:
            end = self.mapFromScene(self.temp_end_pos)
        else:
            path = QPainterPath()
            return path

        # Create path with same curve as visual
        path = QPainterPath()
        path.moveTo(start)

        ctrl_offset = abs(end.x() - start.x()) * 0.5
        ctrl1 = QPointF(start.x() + ctrl_offset, start.y())
        ctrl2 = QPointF(end.x() - ctrl_offset, end.y())
        path.cubicTo(ctrl1, ctrl2, end)

        # Create wider stroke for easier clicking (15 pixels wide for better
        # interaction)
        from PyQt6.QtGui import QPainterPathStroker

        stroker = QPainterPathStroker()
        stroker.setWidth(15)  # Increased from 10 to 15 for easier clicking
        stroker.setCapStyle(Qt.PenCapStyle.RoundCap)
        stroker.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        return stroker.createStroke(path)

    def remove(self):
        """Remove this cable from the scene and disconnect from ports."""
        if self.start_port:
            self.start_port.remove_cable(self)
        if self.end_port:
            self.end_port.remove_cable(self)
        if self.scene():
            self.scene().removeItem(self)

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
        from PyQt6.QtWidgets import QMenu
        from PyQt6.QtGui import QAction

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
        from src.gui.patch_canvas import PatchCanvas

        if self.scene() and isinstance(self.scene().parent(), PatchCanvas):
            canvas: PatchCanvas = self.scene().parent()
            canvas.delete_cable(self, emit_signal=True)
