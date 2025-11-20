"""Qt graphics widget for port visualization.

This module provides the UI layer for ports, handling rendering and interaction.
The actual port logic is in src.gui.port_model.PortModel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QPainter, QPen, QColor
from PyQt6.QtWidgets import QGraphicsItem

from src.gui.core.port import Port

if TYPE_CHECKING:
    from src.gui.widgets.module_widget import ModuleWidget
    from src.gui.widgets.cable_widget import Cable


class PortWidget(QGraphicsItem):
    """Qt graphics item for rendering and interacting with a port.

    This class handles only UI concerns:
    - Visual rendering
    - Mouse interaction
    - Cable visualization

    The actual port logic (value, connections, read/write) is in PortModel.
    Uses composition: contains a PortModel instance.
    """

    def __init__(
        self,
        port: Port,
        parent_module: ModuleWidget,
    ):
        """Initialize the port widget.

        Args:
            port: The port to wrap
            parent_module: The parent module widget
        """
        super().__init__()

        self.port = port
        self.parent_module = parent_module

        # UI-specific attributes
        self.cables: list[Cable] = []
        self.radius = 8
        self.hovered = False

        # Qt setup
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges)
        self.setAcceptHoverEvents(True)

    # ========================================================================
    # Property Delegation - Expose model properties for backward compatibility
    # ========================================================================

    @property
    def port_type(self) -> str:
        """Get port type from model."""
        return self.port.port_type

    @property
    def port_name(self) -> str:
        """Get port name from model."""
        return self.port.port_name

    @property
    def index(self) -> int:
        """Get port index from model."""
        return self.port.index

    @property
    def value(self) -> float:
        """Get port value from model."""
        return self.port.value

    @value.setter
    def value(self, val: float):
        """Set port value in model."""
        self.port.value = val

    @property
    def connected_to(self) -> PortWidget | None:
        """Get connected port widget.

        Note: This returns the PortWidget wrapper, not the PortModel.
        For internal use, access self.model.connected_to for the model.
        """
        if self.port.connected_to is None:
            return None
        # Find the PortWidget that wraps this PortModel
        # This is a bit tricky - we need to maintain a registry or search
        # For now, we'll return None and rely on cable tracking
        return None

    @property
    def is_connected(self) -> bool:
        """Check if port is connected."""
        return self.port.is_connected

    # ========================================================================
    # Logic Methods - Delegate to Model
    # ========================================================================

    def connect(self, other: PortWidget) -> None:
        """Connect this port to another port.

        Args:
            other: The port widget to connect to
        """
        if isinstance(other, PortWidget):
            self.port.connect(other.port)
        elif hasattr(other, "model"):
            # Handle wrapped PortModel
            self.port.connect(other.port)
        else:
            # Assume it's a PortModel directly
            from src.gui.core.port import Port

            if isinstance(other, Port):
                self.port.connect(other)

    def disconnect(self) -> None:
        """Disconnect from any connected port."""
        self.port.disconnect()

    def read(self) -> float:
        """Read value from connected port.

        Returns:
            Value from connected port, or 0.0 if not connected
        """
        return self.port.read()

    def write(self, value: float) -> None:
        """Write a value to this port.

        Args:
            value: The value to write
        """
        self.port.write(value)

    # ========================================================================
    # UI Methods - Rendering and Interaction
    # ========================================================================

    def boundingRect(self) -> QRectF:
        """Return the bounding rectangle of the port.

        Returns:
            Bounding rectangle for painting
        """
        r = self.radius + 2
        return QRectF(-r, -r, r * 2, r * 2)

    def paint(self, painter: QPainter, option, widget=None):
        """Paint the port.

        Args:
            painter: QPainter for drawing
            option: Style options
            widget: Optional widget
        """
        # Port color based on type
        if self.port_type == "input":
            color = QColor(100, 200, 100) if self.hovered else QColor(80, 180, 80)
        else:
            color = QColor(200, 100, 100) if self.hovered else QColor(180, 80, 80)

        painter.setBrush(color)
        painter.setPen(QPen(Qt.GlobalColor.black, 2))
        painter.drawEllipse(QPointF(0, 0), self.radius, self.radius)

        # Draw inner circle if connected (based on cables, not model connection)
        if self.cables:
            painter.setBrush(QColor(255, 255, 100))
            painter.drawEllipse(QPointF(0, 0), self.radius * 0.5, self.radius * 0.5)

    def hoverEnterEvent(self, event):
        """Handle mouse hover enter.

        Args:
            event: Hover event
        """
        self.hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """Handle mouse hover leave.

        Args:
            event: Hover event
        """
        self.hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    # ========================================================================
    # Cable Management (UI Concern)
    # ========================================================================

    def add_cable(self, cable: Cable) -> None:
        """Add a cable connection to this port.

        Args:
            cable: The cable to add
        """
        self.cables.append(cable)
        self.update()

    def remove_cable(self, cable: Cable) -> None:
        """Remove a cable connection from this port.

        Args:
            cable: The cable to remove
        """
        if cable in self.cables:
            self.cables.remove(cable)
            self.update()

    def get_scene_pos(self) -> QPointF:
        """Get the center position of this port in scene coordinates.

        Returns:
            Scene position as QPointF
        """
        return self.scenePos()

    # ========================================================================
    # Debugging
    # ========================================================================

    def __repr__(self) -> str:
        """String representation for debugging.

        Returns:
            Debug string
        """
        return (
            f"PortWidget(model={repr(self.port)}, "
            f"cables={len(self.cables)}, hovered={self.hovered})"
        )
