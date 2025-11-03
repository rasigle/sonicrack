from PyQt6.QtWidgets import QMessageBox, QWidget


def show_about(parent: QWidget):
    """Show about dialog."""
    QMessageBox.about(
        parent,
        "About AudioPlayground",
        "<h2>AudioPlayground Modular Synthesizer</h2>"
        "<p>A full-featured modular synthesis environment.</p>"
        "<p><b>Features:</b></p>"
        "<ul>"
        "<li>Visual modular patching</li>"
        "<li>Real-time audio synthesis</li>"
        "<li>Waveform and spectrum visualization</li>"
        "<li>High-performance audio engine</li>"
        "</ul>"
        "<p>Version 0.1.0</p>"
        "</ul>"
        "<p><b>Author:</b></p>"
        "<p>Rainer Sigle (rainer.sigle@live.de)</p>"
    )
