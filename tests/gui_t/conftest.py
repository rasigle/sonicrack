"""Shared pytest fixtures for GUI tests."""

import contextlib
import os

import pytest
from PyQt6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(name="qapp", scope="session")
def fixture_qapp():
    """Create a single QApplication for the full GUI test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture(name="app")
def fixture_app(qapp):
    """Backward-compatible alias used by existing GUI tests."""
    return qapp


@pytest.fixture(autouse=True)
def cleanup_qt_widgets(qapp):
    """Close and delete top-level widgets after each test.

    Reusing a single QApplication is much more stable on Windows, but tests also
    need deterministic widget cleanup so later cases do not interact with stale
    QWidget instances.
    """
    yield

    for widget in list(qapp.topLevelWidgets()):
        with contextlib.suppress(RuntimeError):
            widget.close()
            widget.deleteLater()

    qapp.processEvents()
