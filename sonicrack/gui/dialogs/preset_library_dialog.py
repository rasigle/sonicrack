"""Preset browser dialog for managing presets."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict, cast

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from sonicrack.gui.core.preset_manager import PresetManager

logger = logging.getLogger(__name__)


class PresetMetadata(TypedDict, total=False):
    """Metadata shape returned by ``PresetManager.list_presets()``."""

    name: str
    author: str
    category: str
    tags: list[str]
    description: str
    created: str
    filepath: str


DEFAULT_PRESET_CATEGORIES = ("User", "Bass", "Lead", "Pad", "FX", "Drums", "Other")


class LibraryPresetBrowserDialog(QDialog):
    """Dialog for browsing and managing library presets.

    Allows users to:
    - Browse available presets
    - Load presets
    - Delete presets
    - View preset details
    - Filter by category
    """

    preset_selected = pyqtSignal(dict)  # Emits preset data when loaded

    def __init__(self, preset_manager: PresetManager, parent: QWidget | None = None):
        """Initialize the preset browser dialog.

        Args:
            preset_manager: PresetManager instance
            parent: Parent widget
        """
        super().__init__(parent)

        self.preset_manager = preset_manager
        self.current_presets: list[PresetMetadata] = []

        self.setWindowTitle("Preset Browser")
        self.setMinimumSize(700, 500)

        self._setup_ui()
        self._load_presets()

    def _setup_ui(self) -> None:
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

    def _load_presets(self) -> None:
        """Load and display available presets."""
        category = self._selected_category()

        self.current_presets = [
            cast(PresetMetadata, preset)
            for preset in self.preset_manager.list_presets(category)
        ]

        if self.category_combo.count() == 1:  # Only "All Categories"
            self._populate_categories()

        self.preset_list.clear()

        for preset in self.current_presets:
            item = QListWidgetItem(self._preset_list_label(preset))
            item.setData(Qt.ItemDataRole.UserRole, preset)
            self.preset_list.addItem(item)

        if not self.current_presets:
            self._clear_details()
            self._set_selection_actions_enabled(False)

    def _populate_categories(self) -> None:
        for category in self.preset_manager.get_categories():
            self.category_combo.addItem(category, category)

    def _selected_category(self) -> str | None:
        return cast(str | None, self.category_combo.currentData())

    @staticmethod
    def _preset_list_label(preset: PresetMetadata) -> str:
        name = preset.get("name") or "Unnamed"
        category = preset.get("category") or "User"
        return f"{name} ({category})"

    def _on_category_changed(self) -> None:
        """Handle category filter change."""
        self._load_presets()

    def _on_preset_selected(
        self,
        current: QListWidgetItem | None,
        previous: QListWidgetItem | None,
    ) -> None:
        """Handle preset selection.

        Args:
            current: Currently selected item
            previous: Previously selected item
        """
        del previous
        if current:
            preset = self._preset_from_item(current)
            self._show_details(preset)
            self._set_selection_actions_enabled(True)
        else:
            self._clear_details()
            self._set_selection_actions_enabled(False)

    def _selected_preset(self) -> PresetMetadata | None:
        current_item = self.preset_list.currentItem()
        if current_item is None:
            return None
        return self._preset_from_item(current_item)

    @staticmethod
    def _preset_from_item(item: QListWidgetItem) -> PresetMetadata:
        return cast(PresetMetadata, item.data(Qt.ItemDataRole.UserRole))

    def _set_selection_actions_enabled(self, enabled: bool) -> None:
        self.load_btn.setEnabled(enabled)
        self.delete_btn.setEnabled(enabled)
        self.export_btn.setEnabled(enabled)

    def _show_details(self, preset: PresetMetadata) -> None:
        """Show preset details.

        Args:
            preset: Preset metadata dictionary
        """
        self.name_label.setText(f"<b>Name:</b> {preset.get('name') or '-'}")
        self.author_label.setText(f"<b>Author:</b> {preset.get('author') or '-'}")
        self.category_label.setText(f"<b>Category:</b> {preset.get('category') or '-'}")

        tags = preset.get("tags", [])
        tags_str = ", ".join(tags) if tags else "-"
        self.tags_label.setText(f"<b>Tags:</b> {tags_str}")

        self.description_text.setText(preset.get("description") or "")
        self.created_label.setText(
            f"<b>Created:</b> {self._format_created(preset.get('created'))}"
        )

    @staticmethod
    def _format_created(created: str | None) -> str:
        if not created:
            return "-"
        try:
            return datetime.fromisoformat(created).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            return created

    def _clear_details(self) -> None:
        """Clear preset details display."""
        self.name_label.setText("<b>Name:</b> -")
        self.author_label.setText("<b>Author:</b> -")
        self.category_label.setText("<b>Category:</b> -")
        self.tags_label.setText("<b>Tags:</b> -")
        self.description_text.clear()
        self.created_label.setText("<b>Created:</b> -")

    def _on_load_clicked(self) -> None:
        """Handle load button click."""
        preset_meta = self._selected_preset()
        if preset_meta is None:
            return

        filepath = self._preset_file(preset_meta)
        if filepath is None:
            QMessageBox.critical(
                self, "Load Error", "Preset metadata has no file path."
            )
            return

        preset_data = self.preset_manager.load_preset(filepath)

        if preset_data:
            self.preset_selected.emit(preset_data)
            self.accept()
        else:
            QMessageBox.critical(self, "Load Error", "Failed to load preset file.")

    @staticmethod
    def _preset_file(preset: PresetMetadata) -> Path | None:
        filepath = preset.get("filepath")
        return Path(filepath) if filepath else None

    def _on_delete_clicked(self) -> None:
        """Handle delete button click."""
        preset_meta = self._selected_preset()
        if preset_meta is None:
            return

        name = preset_meta.get("name", "Unnamed")

        reply = QMessageBox.question(
            self,
            "Delete Preset",
            f"Are you sure you want to delete '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            filepath = self._preset_file(preset_meta)
            if filepath is None:
                QMessageBox.critical(self, "Error", "Preset metadata has no file path.")
                return
            if self.preset_manager.delete_preset(filepath):
                self._load_presets()
                QMessageBox.information(self, "Success", "Preset deleted.")
            else:
                QMessageBox.critical(self, "Error", "Failed to delete preset.")

    def _on_import_clicked(self) -> None:
        """Handle import button click."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Import Preset", "", "JSON Files (*.json);;All Files (*)"
        )

        if filepath:
            result = self.preset_manager.import_preset(Path(filepath))
            if result:
                self._load_presets()
                QMessageBox.information(self, "Success", "Preset imported.")
            else:
                QMessageBox.critical(self, "Error", "Failed to import preset.")

    def _on_export_clicked(self) -> None:
        """Handle export button click."""
        preset_meta = self._selected_preset()
        if preset_meta is None:
            return

        name = preset_meta.get("name", "preset")

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Preset", f"{name}.json", "JSON Files (*.json);;All Files (*)"
        )

        if filepath:
            source = self._preset_file(preset_meta)
            if source is None:
                QMessageBox.critical(self, "Error", "Preset metadata has no file path.")
                return
            if self.preset_manager.export_preset(source, Path(filepath)):
                QMessageBox.information(self, "Success", "Preset exported.")
            else:
                QMessageBox.critical(self, "Error", "Failed to export preset.")


class SaveLibraryPresetDialog(QDialog):
    """Dialog for saving a preset with metadata."""

    def __init__(self, parent: QWidget | None = None):
        """Initialize the save preset dialog.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        self.setWindowTitle("Save Preset")
        self.setMinimumWidth(400)

        self._setup_ui()

    def _setup_ui(self) -> None:
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
        self.category_combo.addItems(DEFAULT_PRESET_CATEGORIES)
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

    def _on_save_clicked(self) -> None:
        """Handle save button click."""
        if not self.name_edit.text().strip():
            QMessageBox.warning(self, "Missing Name", "Please enter a preset name.")
            self.name_edit.setFocus()
            return

        self.accept()

    def get_metadata(self) -> dict[str, Any]:
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
            "description": self.description_edit.toPlainText().strip(),
        }
