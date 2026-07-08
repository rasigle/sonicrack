"""Preset browser dialog for managing presets."""

import logging

from PyQt6 import QtWidgets
from PyQt6.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QVBoxLayout,
)

from sonicrack.patching.module import AudioModule

logger = logging.getLogger(__name__)


class ModuleInfoDialog(QDialog):
    """Dialog showing the basic properties of the selected module."""

    def __init__(self, module: AudioModule, parent=None):
        """Initialize the preset browser dialog.

        Args:
            module: Audio module to show info for
            parent: Parent widget
        """
        super().__init__(parent)

        self.module = module

        self.setWindowTitle("Audio Module Information")
        self.setMinimumSize(700, 500)

        self._setup_ui()
        self._populate_details()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)

        # Preset details
        layout.addWidget(QLabel("Preset Details:"))

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

        # Version
        self.version_label = QLabel("<b>Version:</b> -")
        details_form.addWidget(self.version_label)

        # Description
        details_form.addWidget(QLabel("<b>Description:</b>"))
        self.description_text = QTextEdit()
        self.description_text.setReadOnly(True)
        self.description_text.setMaximumHeight(100)
        details_form.addWidget(self.description_text)

        self.details_group.setLayout(details_form)
        layout.addWidget(self.details_group, 1)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

    def _populate_details(self):
        """Populate preset details from the module."""
        if self.module is None:
            return

        metadata = self.module.metadata
        self.name_label.setText(f"<b>Name:</b> {metadata.title}")
        self.category_label.setText(f"<b>Category:</b> {metadata.category}")
        # self.tags_label.setText(f"<b>Tags:</b>
        # {', '.join(metadata.tags) if metadata.tags else '-'}")
        self.author_label.setText(f"<b>Author:</b> {metadata.author}")
        self.version_label.setText(f"<b>Version:</b> {metadata.version}")
        self.description_text.setHtml(metadata.description)
