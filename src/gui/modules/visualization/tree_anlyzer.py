"""Tree analyzer module for visualizing patch structure."""

import logging
from typing import Any

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QTreeWidget, QTreeWidgetItem

from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.module_registry import register_module
from src.gui.widgets.module_widget import ModuleWidget

logger = logging.getLogger(__name__)


@register_module()
class TreeAnalyzerModule(ModuleWidget):
    """Tree analyzer module for visualizing patch structure.

    This is a pure visualization module - it displays the hierarchical
    structure of the audio patch. It has no input/output ports and doesn't
    affect the audio signal.

    **Usage:**
    - Add to your patch (no connections needed)
    - Shows the tree structure of all connected modules
    - Updates automatically when patch changes
    - Great for understanding signal flow and debugging
    """

    metadata = ModuleMetadata(
        title="Tree",
        category=ModuleCategory.VISUALIZATION,
        description="Patch structure tree view (shows signal flow hierarchy)",
    )

    def __init__(self):
        """Initialize tree analyzer module."""
        super().__init__(
            width=450,
            height=300,
            color=QColor(80, 100, 120),
        )

        # NO INPUT/OUTPUT PORTS - pure visualization

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Create tree widget
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["Module", "Type"])
        self.tree_widget.setColumnWidth(0, 250)
        self.tree_widget.setMinimumHeight(220)
        self.tree_widget.setAlternatingRowColors(True)
        self.tree_widget.setStyleSheet("""
            QTreeWidget {
                background-color: #1a1a1f;
                color: #e0e0e0;
                border: 1px solid #333;
            }
            QTreeWidget::item:selected {
                background-color: #3a5a7a;
            }
            QTreeWidget::item:hover {
                background-color: #2a3a4a;
            }
        """)
        layout.addWidget(self.tree_widget)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Patch compiler reference (will be set when found)
        self.patch_compiler = None
        self._compiler_connected = False

        # Use timer to find patch compiler
        self.connection_timer = QTimer()
        self.connection_timer.timeout.connect(self._try_connect_compiler)
        self.connection_timer.setInterval(100)  # Try every 100ms
        self.connection_timer.start()

    def _try_connect_compiler(self):
        """Try to find the patch compiler via scene/parent chain."""
        if self._compiler_connected:
            self.connection_timer.stop()
            return

        try:
            # Navigate: Module → Scene → View (PatchCanvas) → Window (MainWindow)
            scene = self.scene()
            if scene is None:
                return

            views = scene.views()
            if not views:
                return

            view = views[0]
            main_window = view.window()

            if not hasattr(main_window, 'patch_compiler'):
                return

            # Found patch compiler!
            self.patch_compiler = main_window.patch_compiler

            # Request initial update
            self._update_tree()

            self._compiler_connected = True
            self.connection_timer.stop()

            logger.info("✓ Tree analyzer connected to patch compiler")

            # Set up periodic updates (every 500ms when playing)
            self.update_timer = QTimer()
            self.update_timer.timeout.connect(self._update_tree)
            self.update_timer.setInterval(500)
            self.update_timer.start()

        except Exception as e:
            logger.debug(f"Waiting for patch compiler: {e}")

    def _update_tree(self):
        """Update the tree display from patch compiler."""
        if not self.patch_compiler:
            return

        try:
            # Get tree data from compiler
            if hasattr(self.patch_compiler, 'get_tree_structure'):
                tree_data = self.patch_compiler.get_tree_structure()
                self._display_tree(tree_data)
            elif hasattr(self.patch_compiler, 'compiled_patch'):
                # Fallback: build simple tree from compiled patch
                if self.patch_compiler.compiled_patch:
                    tree_data = {
                        "name": type(self.patch_compiler.compiled_patch).__name__,
                        "inputs": []
                    }
                    self._display_tree(tree_data)
                else:
                    self._display_no_patch()
            else:
                self._display_no_patch()

        except Exception as e:
            logger.debug(f"Error updating tree: {e}")

    def _display_tree(self, tree_data: dict[str, Any]):
        """Display tree structure.

        Args:
            tree_data: Hierarchical tree structure
        """
        self.tree_widget.clear()

        if not tree_data:
            self._display_no_patch()
            return

        if "error" in tree_data:
            error_item = QTreeWidgetItem([tree_data["error"], ""])
            error_item.setForeground(0, QColor("red"))
            self.tree_widget.addTopLevelItem(error_item)
            return

        # Build tree recursively
        root_item = self._build_tree_item(tree_data)
        if root_item:
            self.tree_widget.addTopLevelItem(root_item)
            self.tree_widget.expandAll()

    def _build_tree_item(self, node: dict[str, Any]) -> QTreeWidgetItem | None:
        """Build a tree widget item from a node.

        Args:
            node: Node data from tree structure

        Returns:
            QTreeWidgetItem for display
        """
        if not node or "name" not in node:
            return None

        # Create item
        item = QTreeWidgetItem([node["name"], node.get("type", node["name"])])

        # Mark cycles
        if node.get("cycle", False):
            item.setText(0, f"{node['name']} (cycle detected)")
            item.setForeground(0, QColor("gray"))
            return item

        # Add input connections
        if node.get("inputs"):
            for input_data in node["inputs"]:
                if "node" in input_data:
                    child_node = self._build_tree_item(input_data["node"])
                    if child_node:
                        item.addChild(child_node)

        # Add modulation connections
        if node.get("modulations"):
            mod_header = QTreeWidgetItem(["⚡ Modulations", ""])
            mod_header.setForeground(0, QColor("#4ecdc4"))
            item.addChild(mod_header)

            for mod_data in node["modulations"]:
                port_name = mod_data.get("port_name", "Unknown")

                if "node" in mod_data:
                    child_node = self._build_tree_item(mod_data["node"])
                    if child_node:
                        child_node.setText(0, f"{port_name} → {child_node.text(0)}")
                        child_node.setForeground(0, QColor("#2ecc71"))
                        mod_header.addChild(child_node)

        return item

    def _display_no_patch(self):
        """Display message when no patch is available."""
        self.tree_widget.clear()
        item = QTreeWidgetItem(["No patch compiled", ""])
        item.setForeground(0, QColor("gray"))
        self.tree_widget.addTopLevelItem(item)

    # AudioModuleInterface implementation
    def get_required_inputs(self) -> list[str]:
        """Tree analyzer has no inputs."""
        return []

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Tree analyzer has no audio component.

        Args:
            input_components: Not used
            modulation_components: Not used

        Returns:
            None (visualization only)
        """
        return None

    def cleanup(self):
        """Clean up resources when module is removed."""
        self.connection_timer.stop()
        if hasattr(self, 'update_timer'):
            self.update_timer.stop()
        self.tree_widget.clear()

