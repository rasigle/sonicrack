"""Qt graphics widget for port visualization.

This module provides the UI layer for ports, handling rendering and interaction.
The actual port logic is in sonicrack.gui.port_model.PortModel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PyQt6.QtWidgets import QGraphicsItem

from sonicrack.gui.widgets.signal_style import (
    label_for_port_signal,
    port_color_for_signal,
)
from sonicrack.patching.port import Port, PortType

if TYPE_CHECKING:
    from sonicrack.gui.widgets.cable_widget import Cable
    from sonicrack.gui.widgets.module_widget import ModuleWidget


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
        self.label_width = 46
        self.hovered = False

        # Qt setup
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges)
        self.setAcceptHoverEvents(True)
        self.setToolTip(self._tooltip_text())

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
    def value(self) -> float | np.ndarray:
        """Get port value from model."""
        return self.port.value

    @value.setter
    def value(self, val: float | np.ndarray):
        """Set port value in model."""
        self.port.value = val

    @property
    def connected_to(self) -> PortWidget | None:
        """Return one connected PortWidget if any, else None.

        Uses cable UI links first. For multi-connection ports only the first
        cable peer is returned; use ``port.connected_to`` / ``cables`` for the
        full set.
        """
        for cable in self.cables:
            if cable.start_port is self and cable.end_port is not None:
                return cable.end_port
            if cable.end_port is self and cable.start_port is not None:
                return cable.start_port
        return None

    @property
    def is_connected(self) -> bool:
        """Check if port is connected."""
        return self.port.is_connected

    # ========================================================================
    # Logic Methods - Delegate to Model
    # ========================================================================

    def connect(self, other: PortWidget | Port) -> None:
        """Connect this port to another port.

        Args:
            other: The port widget or raw port to connect to
        """
        if isinstance(other, PortWidget):
            self.port.connect(other.port)
        elif isinstance(other, Port):
            self.port.connect(other)

    def disconnect(self) -> None:
        """Disconnect from any connected port."""
        self.port.disconnect()

    def read(self) -> float | np.ndarray:
        """Read value from connected port.

        Returns:
            Value from connected port, or 0.0 if not connected
        """
        return self.port.read()

    def write(self, value: float | np.ndarray) -> None:
        """Write a value to this port.

        Args:
            value: The value to write
        """
        self.port.write(value)

    def _tooltip_text(self) -> str:
        """Return the tooltip shown when hovering over this port."""
        module_name = self.parent_module.get_display_name()
        direction = "Input" if self.port_type == PortType.INPUT else "Output"
        signal_label = label_for_port_signal(self.port.signal)
        return (
            f"{module_name} {direction}: {self.port_name}\n"
            f"Port Type: {self.port_type}\n"
            f"Signal: {signal_label}"
        )

    # ========================================================================
    # UI Methods - Rendering and Interaction
    # ========================================================================

    def boundingRect(self) -> QRectF:
        """Return the bounding rectangle of the port.

        Returns:
            Bounding rectangle for painting
        """
        r = self.radius + 2
        width = max(r * 2, self.label_width)
        return QRectF(-width / 2, -r, width, r * 2 + 13)

    def paint(self, painter: QPainter | None, option, widget=None):
        """Paint the port.

        Args:
            painter: QPainter for drawing
            option: Style options
            widget: Optional widget
        """
        if painter is None:
            return

        # Jack color follows signal kind (audio / V/Oct / gate / trigger / …)
        color = port_color_for_signal(self.port.signal, hovered=self.hovered)

        painter.setBrush(color)
        painter.setPen(QPen(Qt.GlobalColor.black, 2))
        painter.drawEllipse(QPointF(0, 0), self.radius, self.radius)

        # Draw inner circle if connected (based on cables, not model connection)
        if self.cables:
            painter.setBrush(QColor(255, 255, 100))
            painter.drawEllipse(QPointF(0, 0), self.radius * 0.5, self.radius * 0.5)

        font = QFont("Arial", 6)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        label_text = metrics.elidedText(
            self.port_name,
            Qt.TextElideMode.ElideRight,
            self.label_width,
        )
        label_rect = QRectF(
            -self.label_width / 2,
            self.radius + 1,
            self.label_width,
            10,
        )
        text_color = QColor(232, 236, 240)
        if not self.parent_module.is_active:
            text_color = QColor(150, 154, 158)
        painter.setPen(text_color)
        painter.drawText(
            label_rect,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            label_text,
        )

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

    def itemChange(self, change, value):  # noqa: N802 - Qt API
        """Keep cable geometry in sync when this port moves with its module."""
        if change == QGraphicsItem.GraphicsItemChange.ItemScenePositionHasChanged:
            for cable in self.cables:
                refresh = getattr(cable, "refresh_geometry", None)
                if callable(refresh):
                    refresh()
                else:
                    cable.update()
        return super().itemChange(change, value)

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

    def __repr__(self) -> str:
        """String representation for debugging.

        Returns:
            Debug string
        """
        return (
            f"PortWidget(model={repr(self.port)}, "
            f"cables={len(self.cables)}, hovered={self.hovered})"
        )
