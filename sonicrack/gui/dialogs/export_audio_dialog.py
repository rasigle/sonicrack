"""Dialog for offline audio export."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from sonicrack.config.app_settings import app_settings


class ExportAudioDialog(QDialog):
    """Choose duration and destination for an offline WAV bounce."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Audio")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.25, 300.0)
        self.duration_spin.setDecimals(2)
        self.duration_spin.setSingleStep(0.5)
        self.duration_spin.setValue(8.0)
        self.duration_spin.setSuffix(" s")
        form.addRow("Duration", self.duration_spin)

        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        start_dir = Path(app_settings.file_dialog_start_dir())
        default_path = start_dir / "sonicrack_export.wav"
        self.path_edit.setText(str(default_path))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        path_row.addWidget(self.path_edit)
        path_row.addWidget(browse)
        form.addRow("File", path_row)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        start = self.path_edit.text() or app_settings.file_dialog_start_dir()
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "Export Audio",
            start,
            "WAV files (*.wav)",
        )
        if path:
            if not path.lower().endswith(".wav"):
                path += ".wav"
            self.path_edit.setText(path)

    def duration_seconds(self) -> float:
        return float(self.duration_spin.value())

    def file_path(self) -> str:
        return self.path_edit.text().strip()
