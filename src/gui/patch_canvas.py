"""Modular cable patching system for connecting audio components."""

from __future__ import annotations

from typing import TYPE_CHECKING
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItem
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QColor, QPainterPath

if TYPE_CHECKING:
    from src.gui.widgets.module_widget import ModuleWidget


class Port(QGraphicsItem):
    """A connection port on a module (input or output).

    Ports can be connected with cables to route audio signals.
    """

    def __init__(
        self,
        port_type: str,  # "input" or "output"
        port_name: str,
        parent_module: ModuleWidget,
        index: int = 0,
    ):
        """Initialize a port.

        Args:
            port_type: Either "input" or "output"
            port_name: Display name of the port
            parent_module: The module widget this port belongs to
            index: Index of this port in the module's port list
        """
        super().__init__()
        self.port_type = port_type
        self.port_name = port_name
        self.parent_module = parent_module
        self.index = index
        self.cables: list[Cable] = []
        self.radius = 8

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges)
        self.setAcceptHoverEvents(True)
        self.hovered = False

    def boundingRect(self) -> QRectF:
        """Return the bounding rectangle of the port."""
        r = self.radius + 2
        return QRectF(-r, -r, r * 2, r * 2)

    def paint(self, painter: QPainter, option, widget=None):
        """Paint the port."""
        # Port color based on type
        if self.port_type == "input":
            color = QColor(100, 200, 100) if self.hovered else QColor(80, 180, 80)
        else:
            color = QColor(200, 100, 100) if self.hovered else QColor(180, 80, 80)

        painter.setBrush(color)
        painter.setPen(QPen(Qt.GlobalColor.black, 2))
        painter.drawEllipse(QPointF(0, 0), self.radius, self.radius)

        # Draw inner circle if connected
        if self.cables:
            painter.setBrush(QColor(255, 255, 100))
            painter.drawEllipse(QPointF(0, 0), self.radius * 0.5, self.radius * 0.5)

    def hoverEnterEvent(self, event):
        """Handle mouse hover enter."""
        self.hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """Handle mouse hover leave."""
        self.hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def add_cable(self, cable: Cable):
        """Add a cable connection to this port."""
        self.cables.append(cable)
        self.update()

    def remove_cable(self, cable: Cable):
        """Remove a cable connection from this port."""
        if cable in self.cables:
            self.cables.remove(cable)
            self.update()

    def get_scene_pos(self) -> QPointF:
        """Get the center position of this port in scene coordinates."""
        return self.scenePos()


class Cable(QGraphicsItem):
    """A cable connecting two ports.

    Cables route audio signals between module outputs and inputs.
    """

    def __init__(self, start_port: Port, end_port: Port | None = None):
        """Initialize a cable.

        Args:
            start_port: The output port where the cable starts
            end_port: The input port where the cable ends (can be None for dragging)
        """
        super().__init__()
        self.start_port = start_port
        self.end_port = end_port
        self.temp_end_pos: QPointF | None = None

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setZValue(-1)  # Draw cables behind modules

        if start_port:
            start_port.add_cable(self)
        if end_port:
            end_port.add_cable(self)

    def set_end_port(self, port: Port):
        """Set the end port of the cable."""
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

        # Cable color
        color = QColor(255, 200, 0) if self.isSelected() else QColor(100, 100, 100)
        pen = QPen(color, 3)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        painter.setPen(pen)
        painter.drawPath(path)

    def remove(self):
        """Remove this cable from the scene and disconnect from ports."""
        if self.start_port:
            self.start_port.remove_cable(self)
        if self.end_port:
            self.end_port.remove_cable(self)
        if self.scene():
            self.scene().removeItem(self)


class PatchCanvas(QGraphicsView):
    """The main canvas for the modular patching system.

    This is where modules are placed and connected with cables.
    """

    cable_connected = pyqtSignal(Port, Port)  # Emitted when a cable is connected
    cable_disconnected = pyqtSignal(Port, Port)  # Emitted when a cable is disconnected

    def __init__(self, parent=None):
        """Initialize the patch canvas."""
        super().__init__(parent)

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # Canvas properties
        self.setSceneRect(-2000, -2000, 4000, 4000)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)

        # Background
        self.setBackgroundBrush(QColor(45, 45, 48))

        # Cable dragging state
        self.dragging_cable: Cable | None = None
        self.drag_start_port: Port | None = None

    def mousePressEvent(self, event):
        """Handle mouse press for cable creation."""
        item = self.itemAt(event.pos())

        if isinstance(item, Port):
            # Start dragging a cable from this port
            if item.port_type == "output":
                self.drag_start_port = item
                self.dragging_cable = Cable(item)
                self.scene.addItem(self.dragging_cable)
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move for cable dragging."""
        if self.dragging_cable:
            pos = self.mapToScene(event.pos())
            self.dragging_cable.set_temp_end_pos(pos)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release for cable connection."""
        if self.dragging_cable:
            item = self.itemAt(event.pos())

            if isinstance(item, Port) and item.port_type == "input":
                # Valid connection
                self.dragging_cable.set_end_port(item)
                self.cable_connected.emit(self.drag_start_port, item)
            else:
                # Invalid connection, remove cable
                self.dragging_cable.remove()

            self.dragging_cable = None
            self.drag_start_port = None
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        """Handle key press for deleting cables and modules."""
        if event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            # Get selected items
            selected_items = self.scene.selectedItems()

            # Delete selected cables
            for item in selected_items:
                if isinstance(item, Cable):
                    if item.start_port and item.end_port:
                        self.cable_disconnected.emit(item.start_port, item.end_port)
                    item.remove()

            # Delete selected modules (and their connected cables)
            from src.gui.widgets.module_widget import ModuleWidget

            for item in selected_items:
                if isinstance(item, ModuleWidget):
                    # First, remove all cables connected to this module's ports
                    cables_to_remove = []
                    for port in item.input_ports + item.output_ports:
                        cables_to_remove.extend(port.cables[:])  # Copy list

                    for cable in cables_to_remove:
                        if cable.start_port and cable.end_port:
                            self.cable_disconnected.emit(
                                cable.start_port, cable.end_port
                            )
                        cable.remove()

                    # Remove the module itself
                    self.scene.removeItem(item)

            event.accept()
            return

        super().keyPressEvent(event)

    def add_module(self, module: ModuleWidget, pos: QPointF | None = None):
        """Add a module to the canvas.

        Args:
            module: The module widget to add
            pos: Optional position to place the module
        """
        self.scene.addItem(module)
        if pos:
            module.setPos(pos)
        else:
            # Place at center of view
            center = self.mapToScene(self.viewport().rect().center())
            module.setPos(center)

    def get_connections(self) -> list[tuple[Port, Port]]:
        """Get all cable connections in the canvas.

        Returns:
            List of (output_port, input_port) tuples
        """
        connections = []
        for item in self.scene.items():
            if isinstance(item, Cable) and item.start_port and item.end_port:
                connections.append((item.start_port, item.end_port))
        return connections

    def clear_all(self):
        """Clear all modules and cables from the canvas."""
        self.scene.clear()
        self.dragging_cable = None
        self.drag_start_port = None
