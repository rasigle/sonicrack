"""Modular Synthesizer GUI Application

A full-featured modular synthesizer with visual patching, real-time audio,
and comprehensive visualization.

Usage:
    python modular_synth_app.py
"""

import logging
import sys

from PyQt6.QtWidgets import QApplication

from src.gui.main_window import ModularSynthWindow


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

    # Create and show main window
    window = ModularSynthWindow()
    window.show()

    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
