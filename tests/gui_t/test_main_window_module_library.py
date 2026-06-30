"""Tests for the main-window module library panel."""

from PyQt6 import QtWidgets

from src.gui.main_window import ModularSynthWindow
from src.gui.module_registry import ModuleRegistry
from src.gui.widgets.module_widget import ModuleWidget


def _make_module(title: str, category: str) -> type[ModuleWidget]:
    class TestModule(ModuleWidget):
        class metadata:
            title = ""
            category = ""
            description = "Test"
            version = "1.0.0"
            author = "Test"

    TestModule.metadata.title = title
    TestModule.metadata.category = category
    return TestModule


def _library_widgets(window: ModularSynthWindow) -> list[QtWidgets.QWidget]:
    layout = window.module_library_scroll_layout
    assert layout is not None
    return [
        widget
        for index in range(layout.count())
        if (widget := layout.itemAt(index).widget()) is not None
    ]


def test_module_library_can_hide_and_show():
    """The module library can be hidden and restored from the view action."""
    window = ModularSynthWindow()

    assert window.module_library_panel is not None
    assert not window.module_library_panel.isHidden()

    window._set_module_library_visible(False)
    assert window.module_library_panel.isHidden()
    assert window.module_library_toggle_action is not None
    assert not window.module_library_toggle_action.isChecked()

    window.module_library_toggle_action.trigger()
    assert not window.module_library_panel.isHidden()
    assert window.module_library_toggle_action.isChecked()


def test_module_library_sorts_full_list_alphabetically():
    """Ungrouped module buttons are sorted alphabetically."""
    window = ModularSynthWindow()
    registry = ModuleRegistry()
    registry.register(_make_module("Zeta", "Beta"))
    registry.register(_make_module("alpha", "Alpha"))
    registry.register(_make_module("Middle", "Beta"))
    window.registry = registry
    assert window.module_library_group_checkbox is not None
    window.module_library_group_checkbox.setChecked(False)

    window._populate_module_library()

    button_texts = [
        widget.text()
        for widget in _library_widgets(window)
        if isinstance(widget, QtWidgets.QPushButton)
    ]
    assert button_texts == ["+ alpha", "+ Middle", "+ Zeta"]


def test_module_library_groups_by_sorted_categories_with_sorted_modules():
    """Grouped mode sorts categories and the module buttons inside each category."""
    window = ModularSynthWindow()
    assert window.module_library_group_checkbox is not None
    assert window.module_library_group_checkbox.isChecked()
    registry = ModuleRegistry()
    registry.register(_make_module("Zeta", "Beta"))
    registry.register(_make_module("alpha", "Alpha"))
    registry.register(_make_module("Middle", "Beta"))
    window.registry = registry
    window._populate_module_library()

    visible_texts = [
        widget.text()
        for widget in _library_widgets(window)
        if isinstance(widget, QtWidgets.QLabel | QtWidgets.QPushButton)
    ]
    assert visible_texts == ["Alpha", "+ alpha", "Beta", "+ Middle", "+ Zeta"]
