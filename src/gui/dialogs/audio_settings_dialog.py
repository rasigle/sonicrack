"""Application settings dialog for global audio configuration."""

from __future__ import annotations

import logging

from PyQt6 import QtWidgets

from src.gui.app_settings import app_settings
from src.gui.audio_config import audio_config
from src.gui.ui_constants import DEFAULT_BUFFER_SIZES, DEFAULT_SAMPLE_RATES

logger = logging.getLogger(__name__)


class AudioSettingsDialog(QtWidgets.QDialog):
    """Simple modal dialog to edit global audio parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Application Settings")
        self.setModal(True)
        self.setMinimumWidth(320)

        self._build_ui()
        self._load_current_settings()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        form_layout = QtWidgets.QFormLayout()
        layout.addLayout(form_layout)

        # Sample rate combo
        self.sample_rate_combo = QtWidgets.QComboBox()
        for rate in DEFAULT_SAMPLE_RATES:
            self.sample_rate_combo.addItem(f"{rate} Hz", rate)
        form_layout.addRow("Sample rate", self.sample_rate_combo)

        # Buffer size combo
        self.buffer_size_combo = QtWidgets.QComboBox()
        for size in DEFAULT_BUFFER_SIZES:
            self.buffer_size_combo.addItem(f"{size} samples", size)
        form_layout.addRow("Buffer size", self.buffer_size_combo)

        self.restore_last_patch_checkbox = QtWidgets.QCheckBox(
            "Restore the last patch on startup"
        )
        layout.addWidget(self.restore_last_patch_checkbox)

        # Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._apply_and_close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _load_current_settings(self):
        # Set initial indices to current global config
        sr_index = self.sample_rate_combo.findData(audio_config.sample_rate)
        if sr_index != -1:
            self.sample_rate_combo.setCurrentIndex(sr_index)

        buf_index = self.buffer_size_combo.findData(audio_config.buffer_size)
        if buf_index != -1:
            self.buffer_size_combo.setCurrentIndex(buf_index)

        self.restore_last_patch_checkbox.setChecked(app_settings.restore_last_patch)

    def _apply_and_close(self):
        new_sample_rate = self.sample_rate_combo.currentData()
        new_buffer_size = self.buffer_size_combo.currentData()
        restore_last_patch = self.restore_last_patch_checkbox.isChecked()

        logger.info(
            "Applying settings: sample_rate=%s, buffer_size=%s, "
            "restore_last_patch=%s",
            new_sample_rate,
            new_buffer_size,
            restore_last_patch,
        )

        audio_config.sample_rate = int(new_sample_rate)
        audio_config.buffer_size = int(new_buffer_size)
        app_settings.restore_last_patch = restore_last_patch
        app_settings.save()

        self.accept()
