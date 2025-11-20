"""Modular cable patching system for connecting audio components."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QPainter, QColor
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene

from src.gui.widgets.cable_widget import Cable
from src.gui.widgets.port_widget import PortWidget

if TYPE_CHECKING:
    from src.gui.core.module import ModuleCategory
    from src.gui.modules.output.output import OutputModule
    from src.gui.widgets.module_widget import ModuleWidget


logger = logging.getLogger(__name__)


class PatchCanvas(QGraphicsView):
    """The main canvas for the modular patching system.

    This is where modules are placed and connected with cables.
    """

    cable_connected = pyqtSignal(
        PortWidget, PortWidget
    )  # Emitted when a cable is connected
    cable_disconnected = pyqtSignal(
        PortWidget, PortWidget
    )  # Emitted when a cable is disconnected
    module_deleted = pyqtSignal(object)  # Emitted when a module is deleted

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
        self.drag_start_port: PortWidget | None = None

    def mousePressEvent(self, event):
        """Handle mouse press for cable creation."""
        item = self.itemAt(event.pos())

        if isinstance(item, PortWidget):
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

            if isinstance(item, PortWidget) and item.port_type == "input":
                # Check if trying to connect to the same module
                if item.parent_module == self.drag_start_port.parent_module:
                    # Self-connection not allowed - show error
                    from PyQt6.QtWidgets import QMessageBox

                    QMessageBox.warning(
                        self,
                        "Invalid Connection",
                        "Cannot connect a module's output to its own input.\n\n"
                        "Self-connections would create an infinite feedback loop.",
                    )
                    # Remove the invalid cable
                    self.dragging_cable.remove()
                else:
                    # Check if this connection would create a cycle
                    cycle_info = self._would_create_cycle(self.drag_start_port, item)
                    if cycle_info:
                        # Connection would create a cycle - show error
                        from PyQt6.QtWidgets import QMessageBox

                        QMessageBox.warning(
                            self,
                            "Infinite Loop Detected. This connection would create an "
                            "infinite feedback loop:\n\n{cycle_info}\n\n"
                            "Please check your connections and avoid creating cycles.",
                        )
                        # Remove the invalid cable
                        self.dragging_cable.remove()
                    else:
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
        graph = {}  # module -> list of modules it connects to

        # Get all existing cables from the scene
        for item in self.scene.items():
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
        visited = set()
        rec_stack = set()

        def dfs(module, current_path):
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
                    return " → ".join(cycle_names)

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
            selected_items = self.scene.selectedItems()

            # Delete selected cables
            for item in selected_items:
                if isinstance(item, Cable):
                    self.delete_cable(item, emit_signal=True)

            # Delete selected modules (and their connected cables)
            from src.gui.widgets.module_widget import ModuleWidget

            for item in selected_items:
                if isinstance(item, ModuleWidget):
                    # First, remove all cables connected to this module's ports
                    cables_to_remove = []
                    for port in item.input_ports + item.output_ports:
                        cables_to_remove.extend(port.cables[:])  # Copy list

                    for cable in cables_to_remove:
                        self.delete_cable(cable, emit_signal=True)

                    # Emit signal that module is being deleted
                    self.module_deleted.emit(item)

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
            The created Cable object, or None if connection is invalid
        """
        # Validate ports
        if start_port.port_type != "output" or end_port.port_type != "input":
            logger.warning("Invalid port types for cable creation")
            return None

        # Check for self-connection
        if start_port.parent_module == end_port.parent_module:
            logger.warning("Cannot create self-connection")
            return None

        # Create and add cable
        cable = Cable(start_port, end_port)
        self.scene.addItem(cable)

        # Emit signal
        self.cable_connected.emit(start_port, end_port)

        return cable

    def get_connections(self) -> list[tuple[PortWidget, PortWidget]]:
        """Get all cable connections in the canvas.

        Returns:
            List of (output_port, input_port) tuples
        """
        connections = []
        for item in self.scene.items():
            if isinstance(item, Cable) and item.start_port and item.end_port:
                connections.append((item.start_port, item.end_port))
        return connections

    def get_modules(self) -> list[ModuleWidget]:
        """Get all modules currently on the canvas.

        Returns:
            List of ModuleWidget instances
        """
        from src.gui.widgets.module_widget import ModuleWidget

        return [item for item in self.scene.items() if isinstance(item, ModuleWidget)]

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
        from src.gui.modules.output.output import OutputModule

        for module in self.get_modules():
            if isinstance(module, OutputModule):
                return module
        return None

    def clear_all(self):
        """Clear all modules and cables from the canvas."""
        self.scene.clear()
        self.dragging_cable = None
        self.drag_start_port = None
