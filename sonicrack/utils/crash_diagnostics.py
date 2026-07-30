"""Crash diagnostics for GUI application failures."""

from __future__ import annotations

import contextlib
import faulthandler
import logging
import sys
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO

from PyQt6.QtCore import QMessageLogContext, QtMsgType, qInstallMessageHandler
from PyQt6.QtWidgets import QApplication

from sonicrack.constants import LOG_DIRECTORY

logger = logging.getLogger(__name__)

_fault_trace_file: TextIO | None = None
_hooks_installed = False

_original_excepthook = sys.excepthook
_original_threading_excepthook = threading.excepthook
_original_unraisablehook = sys.unraisablehook
_previous_qt_message_handler: Callable[..., None] | None = None


def _exc_info(
    exc_type: type[BaseException],
    exc_value: BaseException | None,
    exc_traceback: TracebackType | None,
) -> (
    tuple[type[BaseException], BaseException, TracebackType | None]
    | tuple[None, None, None]
):
    if exc_value is None:
        return (None, None, None)
    return (exc_type, exc_value, exc_traceback)


def activate_crash_diagnostics(
    app: QApplication,
    get_window: Callable[[], Any | None],
    *,
    trace_file: str | Path,
    exit_on_thread_exception: bool = False,
) -> None:
    """Install crash hooks that leave useful traces on failure.

    Args:
        app: QApplication to request a controlled exit from on fatal Python errors.
        get_window: Callback returning the main window when available.
        trace_file: Path to the file receiving faulthandler stack dumps.
        exit_on_thread_exception: Whether unhandled worker-thread exceptions should
            request application exit.
    """
    global _hooks_installed, _previous_qt_message_handler

    trace_path = _resolve_trace_path(trace_file)
    _enable_fault_trace_file(trace_path)

    def exception_hook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: TracebackType | None,
    ) -> None:
        try:
            if issubclass(exc_type, KeyboardInterrupt):
                _original_excepthook(exc_type, exc_value, exc_traceback)
                app.exit(130)
                return

            logger.critical(
                "Unhandled exception in UI thread",
                exc_info=(exc_type, exc_value, exc_traceback),
            )
            dump_current_tracebacks("Unhandled exception in UI thread")
            _shutdown_window(get_window, graceful=False)
            _flush_logging()
            app.exit(1)
        except Exception:
            _last_resort_log("Exception inside sys.excepthook")
            with contextlib.suppress(Exception):
                app.exit(1)

    def threading_exception_hook(args: threading.ExceptHookArgs) -> None:
        try:
            logger.critical(
                "Unhandled exception in thread %s",
                getattr(args.thread, "name", "<unknown>"),
                exc_info=_exc_info(args.exc_type, args.exc_value, args.exc_traceback),
            )
            dump_current_tracebacks("Unhandled exception in worker thread")

            if exit_on_thread_exception:
                _shutdown_window(get_window, graceful=False)
                _flush_logging()
                app.exit(1)
                return

            _original_threading_excepthook(args)
        except Exception:
            _last_resort_log("Exception inside threading.excepthook")

    def unraisable_hook(unraisable: sys.UnraisableHookArgs) -> None:
        try:
            exc_value = unraisable.exc_value
            exc_type = type(exc_value) if exc_value is not None else RuntimeError

            logger.critical(
                "Unraisable exception from %r: %s",
                unraisable.object,
                unraisable.err_msg,
                exc_info=_exc_info(exc_type, exc_value, unraisable.exc_traceback),
            )
            dump_current_tracebacks("Unraisable exception")
            _original_unraisablehook(unraisable)
        except Exception:
            _last_resort_log("Exception inside sys.unraisablehook")

    sys.excepthook = exception_hook
    threading.excepthook = threading_exception_hook
    sys.unraisablehook = unraisable_hook

    if not _hooks_installed:
        _previous_qt_message_handler = qInstallMessageHandler(_qt_message_handler)
        _hooks_installed = True

    logger.info("Crash diagnostics enabled: trace_file=%s", trace_path)


def deactivate_crash_diagnostics() -> None:
    """Restore Python crash hooks and close the faulthandler trace file."""
    global _fault_trace_file, _hooks_installed, _previous_qt_message_handler

    sys.excepthook = _original_excepthook
    threading.excepthook = _original_threading_excepthook
    sys.unraisablehook = _original_unraisablehook

    if _hooks_installed:
        qInstallMessageHandler(_previous_qt_message_handler)
        _previous_qt_message_handler = None
        _hooks_installed = False

    try:
        faulthandler.disable()
    except Exception:
        logger.exception("Failed to disable faulthandler")

    if _fault_trace_file is not None:
        try:
            _fault_trace_file.close()
        except Exception:
            logger.exception("Failed to close crash trace file")
        finally:
            _fault_trace_file = None


def dump_current_tracebacks(reason: str) -> None:
    """Write stack traces for all Python threads to the crash trace file."""
    trace_file = _fault_trace_file
    if trace_file is None:
        logger.warning("Cannot dump tracebacks for %s: trace file is not open", reason)
        return

    try:
        trace_file.write(f"\n=== {reason} ===\n")
        trace_file.flush()
        faulthandler.dump_traceback(file=trace_file, all_threads=True)
        trace_file.flush()
    except Exception:
        logger.exception("Failed to dump crash tracebacks")


def _resolve_trace_path(trace_file: str | Path) -> Path:
    trace_path = Path(trace_file)
    if not trace_path.is_absolute():
        trace_path = LOG_DIRECTORY / trace_path
    return trace_path


def _enable_fault_trace_file(trace_path: Path) -> None:
    """Open the crash trace file and enable faulthandler native fault dumps."""
    global _fault_trace_file

    trace_path.parent.mkdir(parents=True, exist_ok=True)

    if _fault_trace_file is not None:
        try:
            faulthandler.disable()
        except Exception:
            logger.exception("Failed to disable previous faulthandler")

        try:
            _fault_trace_file.close()
        except Exception:
            logger.exception("Failed to close previous crash trace file")

    # File must remain open until deactivate_crash_diagnostics() is called
    _fault_trace_file = open(  # noqa: SIM115
        trace_path,
        "a",
        encoding="utf-8",
        buffering=1,
    )
    _fault_trace_file.write(
        "\n=== SonicRack crash diagnostics session "
        f"{datetime.now().isoformat(timespec='seconds')} ===\n"
    )
    _fault_trace_file.flush()

    faulthandler.enable(file=_fault_trace_file, all_threads=True)


def _qt_message_handler(
    msg_type: QtMsgType,
    context: QMessageLogContext,
    message: str | None,
) -> None:
    """Forward Qt diagnostic messages into the Python log."""
    try:
        source = getattr(context, "file", None) or "<unknown>"
        line = getattr(context, "line", 0)
        function = getattr(context, "function", None) or "<unknown>"
        formatted = f"Qt: {message or ''} ({source}:{line}, {function})"

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
            _flush_logging()
        else:
            logger.warning(formatted)

        if _previous_qt_message_handler is not None:
            _previous_qt_message_handler(msg_type, context, message)

    except Exception:
        _last_resort_log("Exception inside Qt message handler")


def _shutdown_window(
    get_window: Callable[[], Any | None],
    *,
    graceful: bool,
) -> None:
    """Best-effort window shutdown from a crash hook."""
    try:
        window = get_window()
    except Exception:
        logger.exception("Could not retrieve window during crash shutdown")
        return

    if window is None:
        return

    try:
        shutdown = getattr(window, "shutdown", None)
        if callable(shutdown):
            shutdown(graceful=graceful)
    except Exception:
        logger.exception("Error during crash shutdown")


def _flush_logging() -> None:
    """Best-effort flush of all logging handlers."""
    with contextlib.suppress(Exception):
        for handler in logging.getLogger().handlers:
            handler.flush()


def _last_resort_log(message: str) -> None:
    """Write diagnostics when normal logging may be broken."""
    try:
        stderr = sys.__stderr__
        if stderr is not None:
            stderr.write(f"{message}\n")
            stderr.flush()
    except Exception:
        pass
