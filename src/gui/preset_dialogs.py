"""Preset browser dialog for managing presets."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QLineEdit, QTextEdit, QGroupBox, QComboBox,
    QFileDialog, QMessageBox, QSplitter, QWidget
)
from PyQt6.QtCore import Qt, pyqtSignal

from .preset_manager import PresetManager

logger = logging.getLogger(__name__)


class PresetBrowserDialog(QDialog):
    """Dialog for browsing and managing presets.

    Allows users to:
    - Browse available presets
    - Load presets
    - Delete presets
    - View preset details
    - Filter by category
    """

    preset_selected = pyqtSignal(dict)  # Emits preset data when loaded

    def __init__(self, preset_manager: PresetManager, parent=None):
        """Initialize the preset browser dialog.

        Args:
            preset_manager: PresetManager instance
            parent: Parent widget
        """
        super().__init__(parent)

        self.preset_manager = preset_manager
        self.current_presets = []

        self.setWindowTitle("Preset Browser")
        self.setMinimumSize(700, 500)

        self._setup_ui()
        self._load_presets()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)

        # Category filter
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Category:"))

        self.category_combo = QComboBox()
        self.category_combo.addItem("All Categories", None)
        self.category_combo.currentIndexChanged.connect(self._on_category_changed)
        filter_layout.addWidget(self.category_combo, 1)

        layout.addLayout(filter_layout)

        # Splitter for preset list and details
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Preset list
        list_widget = QWidget()
        list_layout = QVBoxLayout(list_widget)
        list_layout.setContentsMargins(0, 0, 0, 0)

        list_layout.addWidget(QLabel("Available Presets:"))

        self.preset_list = QListWidget()
        self.preset_list.currentItemChanged.connect(self._on_preset_selected)
        self.preset_list.itemDoubleClicked.connect(self._on_load_clicked)
        list_layout.addWidget(self.preset_list)

        splitter.addWidget(list_widget)

        # Right: Preset details
        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)
        details_layout.setContentsMargins(0, 0, 0, 0)

        details_layout.addWidget(QLabel("Preset Details:"))

        self.details_group = QGroupBox()
        details_form = QVBoxLayout()

        # Name
        self.name_label = QLabel("<b>Name:</b> -")
        self.name_label.setWordWrap(True)
        details_form.addWidget(self.name_label)

        # Author
        self.author_label = QLabel("<b>Author:</b> -")
        self.author_label.setWordWrap(True)
        details_form.addWidget(self.author_label)

        # Category
        self.category_label = QLabel("<b>Category:</b> -")
        details_form.addWidget(self.category_label)

        # Tags
        self.tags_label = QLabel("<b>Tags:</b> -")
        self.tags_label.setWordWrap(True)
        details_form.addWidget(self.tags_label)

        # Description
        details_form.addWidget(QLabel("<b>Description:</b>"))
        self.description_text = QTextEdit()
        self.description_text.setReadOnly(True)
        self.description_text.setMaximumHeight(100)
        details_form.addWidget(self.description_text)

        # Created date
        self.created_label = QLabel("<b>Created:</b> -")
        details_form.addWidget(self.created_label)

        self.details_group.setLayout(details_form)
        details_layout.addWidget(self.details_group)
        details_layout.addStretch()

        splitter.addWidget(details_widget)

        # Set splitter sizes
        splitter.setSizes([300, 400])

        layout.addWidget(splitter, 1)

        # Buttons
        button_layout = QHBoxLayout()

        self.load_btn = QPushButton("Load Preset")
        self.load_btn.setEnabled(False)
        self.load_btn.clicked.connect(self._on_load_clicked)
        button_layout.addWidget(self.load_btn)

        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setEnabled(False)
        self.delete_btn.clicked.connect(self._on_delete_clicked)
        button_layout.addWidget(self.delete_btn)

        button_layout.addStretch()

        self.import_btn = QPushButton("Import...")
        self.import_btn.clicked.connect(self._on_import_clicked)
        button_layout.addWidget(self.import_btn)

        self.export_btn = QPushButton("Export...")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._on_export_clicked)
        button_layout.addWidget(self.export_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

    def _load_presets(self):
        """Load and display available presets."""
        # Get current category filter
        category = self.category_combo.currentData()

        # Load presets
        self.current_presets = self.preset_manager.list_presets(category)

        # Update category combo if needed
        if self.category_combo.count() == 1:  # Only "All Categories"
            categories = self.preset_manager.get_categories()
            for cat in categories:
                self.category_combo.addItem(cat, cat)

        # Clear and populate list
        self.preset_list.clear()

        for preset in self.current_presets:
            name = preset.get("name", "Unnamed")
            category = preset.get("category", "User")
            item = QListWidgetItem(f"{name} ({category})")
            item.setData(Qt.ItemDataRole.UserRole, preset)
            self.preset_list.addItem(item)

        # Update UI
        if not self.current_presets:
            self._clear_details()

    def _on_category_changed(self):
        """Handle category filter change."""
        self._load_presets()

    def _on_preset_selected(self, current: QListWidgetItem, previous: QListWidgetItem):
        """Handle preset selection.

        Args:
            current: Currently selected item
            previous: Previously selected item
        """
        if current:
            preset = current.data(Qt.ItemDataRole.UserRole)
            self._show_details(preset)
            self.load_btn.setEnabled(True)
            self.delete_btn.setEnabled(True)
            self.export_btn.setEnabled(True)
        else:
            self._clear_details()
            self.load_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            self.export_btn.setEnabled(False)

    def _show_details(self, preset: Dict[str, Any]):
        """Show preset details.

        Args:
            preset: Preset metadata dictionary
        """
        self.name_label.setText(f"<b>Name:</b> {preset.get('name', '-')}")
        self.author_label.setText(f"<b>Author:</b> {preset.get('author', '-')}")
        self.category_label.setText(f"<b>Category:</b> {preset.get('category', '-')}")

        tags = preset.get('tags', [])
        tags_str = ", ".join(tags) if tags else "-"
        self.tags_label.setText(f"<b>Tags:</b> {tags_str}")

        self.description_text.setText(preset.get('description', ''))

        created = preset.get('created', '-')
        if created != '-':
            # Format date nicely
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(created)
                created = dt.strftime("%Y-%m-%d %H:%M")
            except:
                pass
        self.created_label.setText(f"<b>Created:</b> {created}")

    def _clear_details(self):
        """Clear preset details display."""
        self.name_label.setText("<b>Name:</b> -")
        self.author_label.setText("<b>Author:</b> -")
        self.category_label.setText("<b>Category:</b> -")
        self.tags_label.setText("<b>Tags:</b> -")
        self.description_text.clear()
        self.created_label.setText("<b>Created:</b> -")

    def _on_load_clicked(self):
        """Handle load button click."""
        current_item = self.preset_list.currentItem()
        if not current_item:
            return

        preset_meta = current_item.data(Qt.ItemDataRole.UserRole)
        filepath = Path(preset_meta['filepath'])

        # Load full preset data
        preset_data = self.preset_manager.load_preset(filepath)

        if preset_data:
            self.preset_selected.emit(preset_data)
            self.accept()
        else:
            QMessageBox.critical(
                self,
                "Load Error",
                "Failed to load preset file."
            )

    def _on_delete_clicked(self):
        """Handle delete button click."""
        current_item = self.preset_list.currentItem()
        if not current_item:
            return

        preset_meta = current_item.data(Qt.ItemDataRole.UserRole)
        name = preset_meta.get('name', 'Unnamed')

        reply = QMessageBox.question(
            self,
            "Delete Preset",
            f"Are you sure you want to delete '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            filepath = Path(preset_meta['filepath'])
            if self.preset_manager.delete_preset(filepath):
                self._load_presets()
                QMessageBox.information(self, "Success", "Preset deleted.")
            else:
                QMessageBox.critical(self, "Error", "Failed to delete preset.")

    def _on_import_clicked(self):
        """Handle import button click."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Import Preset",
            "",
            "JSON Files (*.json);;All Files (*)"
        )

        if filepath:
            result = self.preset_manager.import_preset(Path(filepath))
            if result:
                self._load_presets()
                QMessageBox.information(self, "Success", "Preset imported.")
            else:
                QMessageBox.critical(self, "Error", "Failed to import preset.")

    def _on_export_clicked(self):
        """Handle export button click."""
        current_item = self.preset_list.currentItem()
        if not current_item:
            return

        preset_meta = current_item.data(Qt.ItemDataRole.UserRole)
        name = preset_meta.get('name', 'preset')

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Preset",
            f"{name}.json",
            "JSON Files (*.json);;All Files (*)"
        )

        if filepath:
            source = Path(preset_meta['filepath'])
            if self.preset_manager.export_preset(source, Path(filepath)):
                QMessageBox.information(self, "Success", "Preset exported.")
            else:
                QMessageBox.critical(self, "Error", "Failed to export preset.")


class SavePresetDialog(QDialog):
    """Dialog for saving a preset with metadata."""

    def __init__(self, parent=None):
        """Initialize the save preset dialog.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        self.setWindowTitle("Save Preset")
        self.setMinimumWidth(400)

        self._setup_ui()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)

        # Name
        layout.addWidget(QLabel("Preset Name: *"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Enter preset name...")
        layout.addWidget(self.name_edit)

        # Author
        layout.addWidget(QLabel("Author:"))
        self.author_edit = QLineEdit()
        self.author_edit.setPlaceholderText("Your name (optional)")
        layout.addWidget(self.author_edit)

        # Category
        layout.addWidget(QLabel("Category:"))
        self.category_combo = QComboBox()
        self.category_combo.setEditable(True)
        self.category_combo.addItems(["User", "Bass", "Lead", "Pad", "FX", "Drums", "Other"])
        layout.addWidget(self.category_combo)

        # Tags
        layout.addWidget(QLabel("Tags (comma-separated):"))
        self.tags_edit = QLineEdit()
        self.tags_edit.setPlaceholderText("synth, wobble, dark (optional)")
        layout.addWidget(self.tags_edit)

        # Description
        layout.addWidget(QLabel("Description:"))
        self.description_edit = QTextEdit()
        self.description_edit.setPlaceholderText("Describe your preset... (optional)")
        self.description_edit.setMaximumHeight(100)
        layout.addWidget(self.description_edit)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        save_btn = QPushButton("Save")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._on_save_clicked)
        button_layout.addWidget(save_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)

    def _on_save_clicked(self):
        """Handle save button click."""
        if not self.name_edit.text().strip():
            QMessageBox.warning(
                self,
                "Missing Name",
                "Please enter a preset name."
            )
            self.name_edit.setFocus()
            return

        self.accept()

    def get_metadata(self) -> Dict[str, Any]:
        """Get the entered metadata.

        Returns:
            Dictionary with preset metadata
        """
        tags_text = self.tags_edit.text().strip()
        tags = [t.strip() for t in tags_text.split(",")] if tags_text else []

        return {
            "name": self.name_edit.text().strip(),
            "author": self.author_edit.text().strip(),
            "category": self.category_combo.currentText().strip(),
            "tags": tags,
            "description": self.description_edit.toPlainText().strip()
        }

