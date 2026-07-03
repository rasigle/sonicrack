"""Tests for application crash diagnostics."""

import faulthandler
import sys
import threading

from PyQt6.QtCore import qInstallMessageHandler

from src.utils import crash_diagnostics
from src.utils.crash_diagnostics import activate_crash_diagnostics


class _FakeApp:
    def __init__(self):
        self.exit_code = None

    def exit(self, code: int = 0):
        self.exit_code = code


class _FakeWindow:
    def __init__(self):
        self.shutdown_calls = []

    def shutdown(self, *, graceful: bool):
        self.shutdown_calls.append(graceful)


def test_ui_exception_hook_logs_tracebacks_and_shuts_down(tmp_path):
    """Unhandled UI exceptions should dump all-thread traces before exiting."""
    original_excepthook = sys.excepthook
    original_threading_excepthook = threading.excepthook
    original_unraisablehook = sys.unraisablehook
    trace_file = tmp_path / "fault_trace.log"
    app = _FakeApp()
    window = _FakeWindow()

    try:
        activate_crash_diagnostics(
            app,
            lambda: window,
            trace_file=trace_file,
        )

        try:
            raise RuntimeError("boom")
        except RuntimeError as exc:
            sys.excepthook(type(exc), exc, exc.__traceback__)

        trace_text = trace_file.read_text(encoding="utf-8")

        assert "Unhandled exception in UI thread" in trace_text
        assert "Current thread" in trace_text
        assert window.shutdown_calls == [False]
        assert app.exit_code == 1
    finally:
        sys.excepthook = original_excepthook
        threading.excepthook = original_threading_excepthook
        sys.unraisablehook = original_unraisablehook
        qInstallMessageHandler(None)
        faulthandler.disable()
        if crash_diagnostics._fault_trace_file is not None:
            crash_diagnostics._fault_trace_file.close()
            crash_diagnostics._fault_trace_file = None
