"""Modular Synthesizer GUI Application

A full-featured modular synthesizer with visual patching, real-time audio,
and comprehensive visualization.

Usage:
    python modular_synth_app.py
"""

import logging
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from src.gui.main_window import ModularSynthWindow

RESOURCES_DIR = Path(__file__).parent.parent / 'resources'


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def activate_ui_exception_logging():
    """Activate exception logging for the UI thread."""
    sys._excepthook = sys.excepthook

    def exception_hook(exctype, value, traceback):
        print(exctype, value, traceback)
        sys._excepthook(exctype, value, traceback)
        sys.exit(1)

    sys.excepthook = exception_hook


def main():
    """Main entry point for the modular synthesizer application."""
    setup_logging()
    activate_ui_exception_logging()

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("AudioPlayground Modular Synth")
    app.setOrganizationName("AudioPlayground")

    # Show splash screen
    from PyQt6.QtWidgets import QSplashScreen
    from PyQt6.QtGui import QPixmap
    from PyQt6.QtCore import Qt

    splash_path = Path(RESOURCES_DIR) / 'splash' / 'splash.png'
    if splash_path.exists():
        splash_pixmap = QPixmap(str(splash_path))
        splash = QSplashScreen(splash_pixmap, Qt.WindowType.WindowStaysOnTopHint)
        splash.show()
        app.processEvents()

        # Show loading message
        splash.showMessage(
            "Loading modules...",
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
            Qt.GlobalColor.white
        )
        app.processEvents()
    else:
        splash = None

    # Create and show main window
    window = ModularSynthWindow()

    # Close splash screen when main window is ready
    if splash:
        splash.finish(window)

    window.show()

    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
