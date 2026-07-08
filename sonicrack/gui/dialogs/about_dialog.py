from PyQt6.QtWidgets import QMessageBox, QWidget

from sonicrack.constants import APP_TITLE


def show_about(parent: QWidget, version: str = ""):
    """Show about dialog."""

    version_text = f"<b>Version:</b> {version}" if version else ""
    about_text = (
        f"<h2>SonicRack Modular Synthesizer</h2>"
        "<p>A flexible audio synthesis environment.</p>"
        "<p><b>Features:</b></p>"
        "<ul>"
        "<li>Visual modular patching</li>"
        "<li>Real-time audio synthesis</li>"
        "<li>Waveform and spectrum visualization</li>"
        "<li>High-performance audio engine</li>"
        "</ul>"
        f"{version_text}"
        "</ul>"
        "<p><b>Author:</b></p>"
        "<li>Rainer Sigle (rainer.sigle@live.de)</li>"
    )

    QMessageBox.about(parent, APP_TITLE, about_text)
