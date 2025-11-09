"""A simple implementation of a signal port for a GUI application."""
from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QPainter, QPen, QColor
from PyQt6.QtWidgets import QGraphicsItem


if TYPE_CHECKING:
    from src.gui.widgets.module_widget import ModuleWidget
    from src.gui.cable import Cable

class Port(QGraphicsItem):

    """A simple signal port with a single float value."""
    def __init__(self,
        port_type: str,  # "input" or "output"
        port_name: str,
        parent_module: ModuleWidget,
        index: int = 0,
        ):
        self.value = 0.0
        self.connected_to: Port | None = None  # another Port

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

    def connect(self, other: Port):
        """Connect this port to another port."""
        self.connected_to = other

    def read(self):
        """Read the value from the connected port."""
        if self.connected_to:
            return self.connected_to.value
        return 0.0

    def write(self, value):
        """Write a value to the port."""
        self.value = value

    def disconnect(self):
        """Disconnect the port from any connected port."""
        self.connected_to = None

    @property
    def is_connected(self) -> bool:
        """Check if the port is connected to another port.

        Returns:
            True if connected, False otherwise.
        """
        return self.connected_to is not None


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
