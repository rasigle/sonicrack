"""Packaged entry point for the SonicRack modular synthesizer.

Usage:
    python -m sonicrack.modular_synth_app
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from typing import Any, cast

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QSplashScreen

from sonicrack.constants import (
    CRASH_TRACE_FILENAME,
    LOG_FILENAME,
    SPLASH_RESOURCE,
    resource,
    resource_path,
)
from sonicrack.gui.main_window import ModularSynthWindow
from sonicrack.utils.crash_diagnostics import activate_crash_diagnostics
from sonicrack.utils.logging_config import setup_logging

APP_NAME = "SonicRack Modular Synth"
ORG_NAME = "SonicRack"

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser for packaged and development launches."""
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
        help="Console logging level.",
    )
    parser.add_argument(
        "--detailed-log",
        action="store_true",
        help="Include source file and line number in log messages.",
    )
    parser.add_argument(
        "--no-console-log",
        action="store_true",
        help="Disable console logging and write only to the log file.",
    )
    parser.add_argument(
        "--no-splash",
        action="store_true",
        help="Start without showing the splash screen.",
    )
    return parser


def _configure_logging(args: argparse.Namespace) -> None:
    """Configure app logging from parsed command-line arguments."""
    setup_logging(
        level=getattr(logging, args.log_level),
        log_file=LOG_FILENAME,
        console_output=not args.no_console_log,
        detailed=args.detailed_log,
    )


def _create_application(qt_args: Sequence[str]) -> QApplication:
    """Create or reuse the QApplication instance."""
    existing_app = QApplication.instance()
    if existing_app is not None:
        app = existing_app
    else:
        app = QApplication([sys.argv[0], *qt_args])

    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    return cast(QApplication, app)


def _create_splash(app: QApplication, enabled: bool) -> QSplashScreen | None:
    """Create and display the splash screen if the resource is available."""
    if not enabled:
        return None

    splash_resource = resource(*SPLASH_RESOURCE)
    if not splash_resource.is_file():
        logger.info("Splash image not found: %s", splash_resource)
        return None

    with resource_path(*SPLASH_RESOURCE) as splash_path:
        splash_pixmap = QPixmap(str(splash_path))
    if splash_pixmap.isNull():
        logger.warning("Splash image could not be loaded: %s", splash_resource)
        return None

    splash = QSplashScreen(splash_pixmap, Qt.WindowType.WindowStaysOnTopHint)
    splash.show()
    app.processEvents()

    splash.showMessage(
        "Loading modules...",
        Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
        Qt.GlobalColor.white,
    )
    app.processEvents()
    return splash


def _shutdown_window(
    get_window: Callable[[], ModularSynthWindow | None],
    *,
    graceful: bool,
) -> None:
    """Shut down the main window if it has been created."""
    window = get_window()
    if window is None:
        return

    try:
        window.shutdown(graceful=graceful)
    except Exception:
        logger.exception("Error during application shutdown")


def activate_ui_exception_logging(
    app: QApplication,
    get_window: Callable[[], ModularSynthWindow | None],
) -> None:
    """Log uncaught UI exceptions and request a controlled application exit."""
    activate_crash_diagnostics(
        app,
        cast(Callable[[], Any | None], get_window),
        trace_file=CRASH_TRACE_FILENAME,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the modular synthesizer application and return its exit code."""
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args, qt_args = _build_arg_parser().parse_known_args(raw_args)

    _configure_logging(args)
    logger.info("Starting %s", APP_NAME)

    app = _create_application(qt_args)
    window: ModularSynthWindow | None = None

    def get_window() -> ModularSynthWindow | None:
        return window

    activate_ui_exception_logging(app, get_window)
    app.aboutToQuit.connect(lambda: _shutdown_window(get_window, graceful=True))

    splash = _create_splash(app, enabled=not args.no_splash)

    try:
        window = ModularSynthWindow()

        if splash is not None:
            splash.finish(window)

        window.show()
        return app.exec()
    except Exception:
        logger.critical("Failed to start application", exc_info=True)
        if splash is not None:
            splash.close()
        _shutdown_window(get_window, graceful=False)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
