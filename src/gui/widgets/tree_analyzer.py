"""Tree analyzer widget for visualizing patch structure."""

import logging
from typing import Any

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem

logger = logging.getLogger(__name__)


class TreeAnalyzer(QWidget):
    """Widget to display patch structure as a hierarchical tree."""

    def __init__(self, parent=None):
        """Initialize the tree analyzer."""
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Tree widget
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["Module", "Type"])
        self.tree_widget.setColumnWidth(0, 200)
        self.tree_widget.setMinimumHeight(150)
        self.tree_widget.setAlternatingRowColors(True)
        layout.addWidget(self.tree_widget)

        # Set minimum size for the widget
        self.setMinimumHeight(180)

    def update_tree(self, tree_data: dict[str, Any]):
        """Update the tree display with new patch data.

        Args:
            tree_data: Hierarchical tree structure from PatchCompiler
        """
        logger.debug(f"Updating tree with data: {tree_data}")
        self.tree_widget.clear()

        if not tree_data:
            logger.warning("Empty tree data received")
            return

        if "error" in tree_data:
            error_item = QTreeWidgetItem([tree_data["error"], ""])
            error_item.setForeground(0, QColor("red"))
            self.tree_widget.addTopLevelItem(error_item)
            logger.warning(f"Tree error: {tree_data['error']}")
            return

        # Build tree recursively
        root_item = self._build_tree_item(tree_data)
        if root_item:
            self.tree_widget.addTopLevelItem(root_item)
            self.tree_widget.expandAll()
            logger.info("Tree updated successfully")
        else:
            logger.error("Failed to build tree item")

    def _build_tree_item(self, node: dict[str, Any]) -> QTreeWidgetItem:
        """Build a tree widget item from a node.

        Args:
            node: Node data from tree structure

        Returns:
            QTreeWidgetItem for display
        """
        # Validate node
        if not node or "name" not in node:
            logger.error(f"Invalid node data: {node}")
            return QTreeWidgetItem(["<invalid>", ""])

        # Create item
        item = QTreeWidgetItem([node["name"], node["name"]])

        # Mark cycles
        if node.get("cycle", False):
            item.setText(0, f"{node['name']} (cycle detected)")
            item.setForeground(0, QColor("gray"))
            return item

        # Add input connections directly (no "Inputs" header)
        if node.get("inputs"):
            for input_data in node["inputs"]:
                if "node" in input_data:
                    child_node = self._build_tree_item(input_data["node"])
                    if child_node:
                        item.addChild(child_node)

        # Add modulation connections under a "Modulations" header
        if node.get("modulations"):
            mod_header = QTreeWidgetItem(["⚡ Modulations", ""])
            mod_header.setForeground(0, QColor("darkGreen"))
            item.addChild(mod_header)

            for mod_data in node["modulations"]:
                port_name = mod_data.get("port_name", "Unknown")

                if "node" in mod_data:
                    child_node = self._build_tree_item(mod_data["node"])
                    if child_node:
                        # Add port name as prefix to the module name
                        child_node.setText(0, f"{port_name} → {child_node.text(0)}")
                        child_node.setForeground(0, QColor("green"))
                        mod_header.addChild(child_node)

        return item

    def clear(self):
        """Clear the tree display."""
        self.tree_widget.clear()
