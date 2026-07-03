"""Crash diagnostics for GUI application failures."""

from __future__ import annotations

import faulthandler
import logging
import sys
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO

from PyQt6.QtCore import QtMsgType, qInstallMessageHandler
from PyQt6.QtWidgets import QApplication

from src.constants import LOG_DIRECTORY

logger = logging.getLogger(__name__)

_fault_trace_file: TextIO | None = None


def activate_crash_diagnostics(
    app: QApplication,
    get_window: Callable[[], Any | None],
    *,
    trace_file: str | Path,
) -> None:
    """Install crash hooks that leave useful traces on failure.

    Args:
        app: QApplication to request a controlled exit from on fatal Python errors.
        get_window: Callback returning the main window when available.
        trace_file: Path to the file receiving faulthandler stack dumps.
    """
    trace_path = Path(trace_file)
    if not trace_path.is_absolute():
        trace_path = LOG_DIRECTORY / trace_path
    _enable_fault_trace_file(trace_path)

    previous_excepthook = sys.excepthook
    previous_threading_excepthook = threading.excepthook
    previous_unraisablehook = sys.unraisablehook

    def exception_hook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: TracebackType | None,
    ) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            previous_excepthook(exc_type, exc_value, exc_traceback)
            app.exit(130)
            return

        logger.critical(
            "Unhandled exception in UI thread",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        dump_current_tracebacks("Unhandled exception in UI thread")
        _shutdown_window(get_window, graceful=False)
        app.exit(1)

    def threading_exception_hook(args: threading.ExceptHookArgs) -> None:
        logger.critical(
            "Unhandled exception in thread %s",
            getattr(args.thread, "name", "<unknown>"),
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
        dump_current_tracebacks("Unhandled exception in worker thread")
        previous_threading_excepthook(args)

    def unraisable_hook(unraisable: sys.UnraisableHookArgs) -> None:
        logger.critical(
            "Unraisable exception from %r: %s",
            unraisable.object,
            unraisable.err_msg,
            exc_info=(
                type(unraisable.exc_value),
                unraisable.exc_value,
                unraisable.exc_traceback,
            ),
        )
        dump_current_tracebacks("Unraisable exception")
        previous_unraisablehook(unraisable)

    sys.excepthook = exception_hook
    threading.excepthook = threading_exception_hook
    sys.unraisablehook = unraisable_hook
    qInstallMessageHandler(_qt_message_handler)

    logger.info("Crash diagnostics enabled: trace_file=%s", trace_path)


def dump_current_tracebacks(reason: str) -> None:
    """Write stack traces for all Python threads to the crash trace file."""
    if _fault_trace_file is None:
        logger.warning("Cannot dump tracebacks for %s: trace file is not open", reason)
        return

    try:
        _fault_trace_file.write(f"\n=== {reason} ===\n")
        _fault_trace_file.flush()
        faulthandler.dump_traceback(file=_fault_trace_file, all_threads=True)
        _fault_trace_file.flush()
    except Exception:
        logger.exception("Failed to dump crash tracebacks")


def _enable_fault_trace_file(trace_path: Path) -> None:
    """Open the crash trace file and enable faulthandler native fault dumps."""
    global _fault_trace_file

    trace_path.parent.mkdir(parents=True, exist_ok=True)

    if _fault_trace_file is not None:
        try:
            faulthandler.disable()
            _fault_trace_file.close()
        except Exception:
            logger.exception("Failed to close previous crash trace file")

    _fault_trace_file = open(trace_path, "a", encoding="utf-8")
    _fault_trace_file.write(
        "\n=== AudioPlayground crash diagnostics session "
        f"{datetime.now().isoformat(timespec='seconds')} ===\n"
    )
    _fault_trace_file.flush()
    faulthandler.enable(file=_fault_trace_file, all_threads=True)


def _qt_message_handler(
    msg_type: QtMsgType,
    context: Any,
    message: str,
) -> None:
    """Forward Qt diagnostic messages into the Python log."""
    source = getattr(context, "file", None) or "<unknown>"
    line = getattr(context, "line", 0)
    function = getattr(context, "function", None) or "<unknown>"
    formatted = "Qt: %s (%s:%s, %s)" % (message, source, line, function)

    if msg_type == QtMsgType.QtDebugMsg:
        logger.debug(formatted)
    elif msg_type == QtMsgType.QtInfoMsg:
        logger.info(formatted)
    elif msg_type == QtMsgType.QtWarningMsg:
        logger.warning(formatted)
    elif msg_type == QtMsgType.QtCriticalMsg:
        logger.error(formatted)
    elif msg_type == QtMsgType.QtFatalMsg:
        logger.critical(formatted)
        dump_current_tracebacks("Qt fatal message")
    else:
        logger.warning(formatted)


def _shutdown_window(
    get_window: Callable[[], Any | None],
    *,
    graceful: bool,
) -> None:
    """Best-effort window shutdown from a crash hook."""
    window = get_window()
    if window is None:
        return

    try:
        shutdown = getattr(window, "shutdown", None)
        if callable(shutdown):
            shutdown(graceful=graceful)
    except Exception:
        logger.exception("Error during crash shutdown")
