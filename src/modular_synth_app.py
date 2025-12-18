"""Modular Synthesizer GUI Application

A full-featured modular synthesizer with visual patching, real-time audio,
and comprehensive visualization.

Usage:
    python modular_synth_app.py
"""

import sys

from PyQt6.QtWidgets import QApplication

from src.constants import SPLASH_PATH, LOG_FILENAME
from src.gui.main_window import ModularSynthWindow
from src.utils import setup_logging


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
    setup_logging(log_file=LOG_FILENAME, console_output=True, detailed=False)
    activate_ui_exception_logging()

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("AudioPlayground Modular Synth")
    app.setOrganizationName("AudioPlayground")

    # Show splash screen
    from PyQt6.QtWidgets import QSplashScreen
    from PyQt6.QtGui import QPixmap
    from PyQt6.QtCore import Qt

    if SPLASH_PATH.exists():
        splash_pixmap = QPixmap(str(SPLASH_PATH))
        splash = QSplashScreen(splash_pixmap, Qt.WindowType.WindowStaysOnTopHint)
        splash.show()
        app.processEvents()

        # Show loading message
        splash.showMessage(
            "Loading modules...",
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
            Qt.GlobalColor.white,
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
