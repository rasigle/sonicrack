"""Main window for the modular synthesizer."""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from sonicrack import __version__
from sonicrack.constants import PRESET_FILE_EXTENSION, resource, resource_path
from sonicrack.gui.app_settings import app_settings
from sonicrack.gui.audio_engine import AudioEngine
from sonicrack.gui.core.module import ModuleCategory
from sonicrack.gui.core.preset_manager import PresetManager
from sonicrack.gui.dialogs.about_dialog import show_about
from sonicrack.gui.dialogs.preset_library_dialog import (
    LibraryPresetBrowserDialog,
    SaveLibraryPresetDialog,
)
from sonicrack.gui.module_registry import initialize_module_registry
from sonicrack.gui.ui_constants import (
    APP_ICON_RESOURCE,
    APP_TITLE,
    DEFAULT_AUTOSAVE_PATCH,
)
from sonicrack.gui.widgets.patch_canvas import PatchCanvas

if TYPE_CHECKING:
    from sonicrack.gui.module_registry import ModuleRegistry
    from sonicrack.gui.modules.output.output import OutputModule
    from sonicrack.gui.widgets.port_widget import PortWidget

logger = logging.getLogger(__name__)


def _target_port_names_for_restore(port_name: str | None) -> tuple[str, ...]:
    """Return saved target port names plus current aliases for patch restore."""
    if not isinstance(port_name, str):
        return ()

    aliases = {
        "L/Mono": ("L/Mono", "Left/Mono", "In"),
        "Left/Mono": ("Left/Mono", "L/Mono", "In"),
        "R": ("R", "Right", "In"),
        "Right": ("Right", "R", "In"),
        "Input": ("Input", "In"),
        "in": ("in", "In"),
    }
    return aliases.get(port_name, (port_name,))


class ModularSynthWindow(QMainWindow):
    """Main window for the modular synthesizer application.

    Provides a modular audio synthesis environment with:
    - Patch canvas for visual module patching and editing
    - Module library
    - Real-time audio playback
    - Preset management
    """

    def __init__(
        self,
        *,
        restore_last_patch: bool | None = None,
        autosave_patch_path: str | Path | None = None,
    ):
        """Initialize the main window."""
        super().__init__()

        # Patch file tracking
        self.current_patch_path = None  # Path to currently loaded patch file
        self.patch_modified: bool = False  # Track if patch has unsaved changes
        self._is_shutting_down = False
        self._autosave_patch_path = Path(autosave_patch_path or DEFAULT_AUTOSAVE_PATCH)

        # Initialize core components
        logger.debug("Initializing ModularSynthWindow core components")
        self.audio_engine: AudioEngine = AudioEngine()
        self.registry: ModuleRegistry = initialize_module_registry()
        self.preset_manager: PresetManager = PresetManager()

        # UI elements
        self.patch_canvas: PatchCanvas | None = None
        self.statusbar: QStatusBar | None = None
        self.main_splitter: QSplitter | None = None
        self.module_library_panel: QWidget | None = None
        self.module_library_scroll_layout: QVBoxLayout | None = None
        self.module_library_group_checkbox: QtWidgets.QCheckBox | None = None
        self.module_library_toggle_action: QtGui.QAction | None = None
        self.full_screen_action: QtGui.QAction | None = None
        self._window_state_before_full_screen: Qt.WindowState | None = None

        # UI setup
        logger.debug("Initializing UI components")
        self._setup_ui()
        self._setup_app_icon()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_statusbar()
        self._connect_signals()
        should_restore_last_patch = (
            app_settings.restore_last_patch
            if restore_last_patch is None
            else restore_last_patch
        )
        if should_restore_last_patch:
            self._restore_last_patch()

        logger.debug("Modular Synth Window initialized")

    def _setup_ui(self):
        """Setup the user interface."""
        # Central widget with splitter
        self.setWindowTitle(APP_TITLE)
        self.setGeometry(100, 100, 1400, 900)

        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        # Main splitter (horizontal)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter = main_splitter

        # Left panel - Module Library
        module_library = self._create_module_library_panel()
        self.module_library_panel = module_library
        main_splitter.addWidget(module_library)

        # Center - Patch Canvas
        canvas_widget = QWidget()
        canvas_layout = QVBoxLayout(canvas_widget)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        # Patch canvas
        self.patch_canvas = PatchCanvas()
        canvas_layout.addWidget(self.patch_canvas)

        main_splitter.addWidget(canvas_widget)

        # Set splitter sizes
        main_splitter.setSizes([220, 1180])

        layout.addWidget(main_splitter)

    def _setup_app_icon(self):
        """Set up the application icon in the title bar."""
        icon_resource = resource(*APP_ICON_RESOURCE)
        if not icon_resource.is_file():
            logger.warning("App icon not found at %s", icon_resource)
            return

        with resource_path(*APP_ICON_RESOURCE) as icon_path:
            icon_pixmap = QtGui.QPixmap(str(icon_path))
        if icon_pixmap.isNull():
            logger.warning("App icon could not be loaded: %s", icon_resource)
            return

        self.setWindowIcon(QtGui.QIcon(icon_pixmap))

    def _create_module_library_panel(self) -> QWidget:
        """Create the module library panel.

        Returns:
            Widget containing the module library
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Title
        header_layout = QtWidgets.QHBoxLayout()
        title = QLabel("Module Library")
        title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        header_layout.addWidget(title)

        hide_button = QtWidgets.QToolButton()
        hide_button.setText("Hide")
        hide_button.setToolTip("Hide module library")
        hide_button.clicked.connect(lambda: self._set_module_library_visible(False))
        header_layout.addWidget(hide_button)
        layout.addLayout(header_layout)

        group_checkbox = QtWidgets.QCheckBox("Group by category")
        group_checkbox.setToolTip("Group module buttons by their registered category")
        group_checkbox.setChecked(True)
        group_checkbox.toggled.connect(self._populate_module_library)
        self.module_library_group_checkbox = group_checkbox
        layout.addWidget(group_checkbox)

        # Module buttons
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(180)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        scroll_layout.addStretch()
        self.module_library_scroll_layout = scroll_layout
        self._populate_module_library()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        return panel

    def _populate_module_library(self):
        """Populate module library buttons in the selected grouping mode."""
        scroll_layout = self.module_library_scroll_layout
        if scroll_layout is None:
            return

        while scroll_layout.count():
            item = scroll_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if self._is_module_library_grouped():
            for category in self._sorted_module_categories():
                category_label = QLabel(str(category))
                category_label.setStyleSheet(
                    "font-weight: bold; padding: 8px 4px 2px 4px;"
                )
                scroll_layout.addWidget(category_label)
                for module_name in self._sorted_module_names(category=category):
                    scroll_layout.addWidget(
                        self._create_module_library_button(module_name)
                    )
        else:
            for module_name in self._sorted_module_names():
                scroll_layout.addWidget(self._create_module_library_button(module_name))

        scroll_layout.addStretch()

    def _is_module_library_grouped(self) -> bool:
        """Return whether the module library should group modules by category."""
        return bool(
            self.module_library_group_checkbox
            and self.module_library_group_checkbox.isChecked()
        )

    def _sorted_module_names(self, category: str | None = None) -> list[str]:
        """Return module names alphabetically, optionally filtered by category."""
        if category is None:
            module_names = self.registry.list_modules()
        else:
            module_names = list(self.registry.get_by_category(category).keys())
        return sorted(module_names, key=str.casefold)

    def _sorted_module_categories(self) -> list[str]:
        """Return non-empty module categories alphabetically."""
        return sorted(
            (
                category
                for category in self.registry.get_categories()
                if self.registry.get_by_category(category)
            ),
            key=str.casefold,
        )

    def _create_module_library_button(self, module_name: str) -> QtWidgets.QPushButton:
        """Create a button that adds a module to the canvas."""
        btn = QtWidgets.QPushButton(f"+ {module_name}")
        btn.setMinimumHeight(35)
        btn.clicked.connect(lambda checked, name=module_name: self._add_module(name))
        return btn

    def _set_module_library_visible(self, visible: bool):
        """Show or hide the module library panel."""
        if self.module_library_panel is not None:
            self.module_library_panel.setVisible(visible)

        if self.module_library_toggle_action is not None:
            with contextlib.suppress(RuntimeError):
                self.module_library_toggle_action.setChecked(visible)

    def _set_full_screen(self, enabled: bool):
        """Enter or leave full-screen mode."""
        if enabled == self.isFullScreen():
            if self.full_screen_action is not None:
                with contextlib.suppress(RuntimeError):
                    self.full_screen_action.setChecked(enabled)
            return

        if enabled:
            self._window_state_before_full_screen = (
                self.windowState() & ~Qt.WindowState.WindowFullScreen
            )
            self.showFullScreen()
        else:
            previous_state = self._window_state_before_full_screen
            self._window_state_before_full_screen = None
            self.showNormal()
            if (
                previous_state is not None
                and previous_state & Qt.WindowState.WindowMaximized
            ):
                self.showMaximized()

        if self.full_screen_action is not None:
            with contextlib.suppress(RuntimeError):
                self.full_screen_action.setChecked(enabled)

    def _exit_full_screen(self):
        """Exit full-screen mode when active."""
        if self.isFullScreen():
            self._set_full_screen(False)

    def _setup_menu(self):
        """Set up the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        new_action = QtGui.QAction("&New Patch", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self._new_patch)
        file_menu.addAction(new_action)

        open_patch_action = QtGui.QAction("&Open Patch...", self)
        open_patch_action.setShortcut("Ctrl+O")
        open_patch_action.triggered.connect(self._open_patch)
        file_menu.addAction(open_patch_action)

        file_menu.addSeparator()

        save_patch_action = QtGui.QAction("&Save Patch", self)
        save_patch_action.setShortcut("Ctrl+S")
        save_patch_action.triggered.connect(self._save_patch)
        file_menu.addAction(save_patch_action)

        save_patch_as_action = QtGui.QAction("Save Patch &As...", self)
        save_patch_as_action.setShortcut("Ctrl+Shift+S")
        save_patch_as_action.triggered.connect(self._save_patch_as)
        file_menu.addAction(save_patch_as_action)

        file_menu.addSeparator()

        save_preset_action = QtGui.QAction("Save as Preset...", self)
        save_preset_action.triggered.connect(self._save_as_library_preset)
        file_menu.addAction(save_preset_action)

        load_preset_action = QtGui.QAction("&Load Preset...", self)
        load_preset_action.setShortcut("Ctrl+Shift+O")
        load_preset_action.triggered.connect(self._load_preset)
        file_menu.addAction(load_preset_action)

        file_menu.addSeparator()

        exit_action = QtGui.QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        clear_action = QtGui.QAction("&Clear Canvas", self)
        clear_action.triggered.connect(self._clear_canvas)
        edit_menu.addAction(clear_action)

        # View menu
        view_menu = menubar.addMenu("&View")
        self.module_library_toggle_action = QtGui.QAction("Show Module Library", self)
        self.module_library_toggle_action.setCheckable(True)
        self.module_library_toggle_action.setChecked(True)
        self.module_library_toggle_action.setShortcut("Ctrl+L")
        self.module_library_toggle_action.toggled.connect(
            self._set_module_library_visible
        )
        view_menu.addAction(self.module_library_toggle_action)

        full_screen_action = QtGui.QAction("&Full Screen", self)
        full_screen_action.setCheckable(True)
        full_screen_action.setShortcut("F11")
        full_screen_action.toggled.connect(self._set_full_screen)
        self.full_screen_action = full_screen_action
        view_menu.addAction(full_screen_action)

        exit_full_screen_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Esc"), self)
        exit_full_screen_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        exit_full_screen_shortcut.activated.connect(self._exit_full_screen)

        # Settings action
        settings_action = QtGui.QAction("&Settings", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self._open_settings_dialog)
        menubar.addAction(settings_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")
        about_action = QtGui.QAction("&About", self)
        about_action.triggered.connect(lambda: show_about(self, version=__version__))
        help_menu.addAction(about_action)

    def _setup_toolbar(self):
        """Setup the toolbar."""
        return

    def _setup_statusbar(self):
        """Setup the status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Ready")

    def _connect_signals(self):
        """Connect signals."""

        # Patch canvas signals
        patch_canvas = self._require_patch_canvas()
        patch_canvas.cable_connected.connect(self._on_cable_connected)
        patch_canvas.cable_disconnected.connect(self._on_cable_disconnected)
        patch_canvas.module_deleted.connect(self._on_module_deleted)

    def _require_patch_canvas(self) -> PatchCanvas:
        """Return the initialized patch canvas."""
        if self.patch_canvas is None:
            raise RuntimeError("Patch canvas has not been initialized")
        return self.patch_canvas

    def _require_statusbar(self) -> QStatusBar:
        """Return the initialized status bar."""
        if self.statusbar is None:
            raise RuntimeError("Status bar has not been initialized")
        return self.statusbar

    def _add_module(self, module_name: str):
        """Add a module to the canvas.

        Args:
            module_name: Name of the module type to add
        """
        module_class = self.registry.get(module_name, strict=True)
        assert module_class is not None
        patch_canvas = self._require_patch_canvas()
        statusbar = self._require_statusbar()

        # Check if trying to add an Output module when one already exists
        if module_class.metadata.category == ModuleCategory.OUTPUT:
            # Check if an Output module already exists
            for module in patch_canvas.get_modules():
                if module.metadata.category == ModuleCategory.OUTPUT:
                    QMessageBox.warning(
                        self,
                        "Output Already Exists",
                        "Only one Output module is allowed per patch.\n\n"
                        "The Output module represents your audio device "
                        "(speakers/DAC). Multiple outputs would cause conflicts.\n\n"
                        "Connect multiple audio sources to the existing Output module "
                        "instead.",
                    )
                    statusbar.showMessage("Cannot add multiple Output modules")
                    logger.warning("Attempted to add multiple Output modules")
                    return

        # Initialize the module to add with default parameters
        module_instance = module_class()

        patch_canvas.add_module(module_instance)
        statusbar.showMessage(f"Added {module_name}")
        logger.info(f"Added module: {module_name}")

        # Mark patch as modified
        self._mark_patch_modified()

        self.audio_engine.add_module(module_instance)

        # If this is an Output module, give it a reference to the audio engine
        if module_class.metadata.category == ModuleCategory.OUTPUT:
            from sonicrack.gui.modules.output.output import OutputModule

            output_module = cast(OutputModule, module_instance)
            output_module.audio_engine = self.audio_engine
            logger.debug("Set audio_engine reference on Output module")

    def _on_audio_error(self, error: str):
        """Handle audio error."""
        QMessageBox.critical(self, "Audio Error", f"Audio error occurred:\n{error}")
        self._require_statusbar().showMessage(f"Error: {error}")

    def _on_cable_connected(self, start_port: PortWidget, target_port: PortWidget):
        """Handle cable connection."""
        try:
            start_port.port.connect(target_port.port)
        except ValueError as exc:
            logger.warning("Rejected cable connection: %s", exc)
            return
        self.audio_engine.mark_graph_changed()
        self._mark_patch_modified()

        # Notify modules about connection change
        start_module = start_port.parent_module
        target_module = target_port.parent_module

        # Notify target module (input port) about connection
        if hasattr(target_module, "on_port_connection_changed"):
            target_module.on_port_connection_changed(target_port.port_name, True)

        # Notify source module (output port) about connection if needed
        if hasattr(start_module, "on_port_connection_changed"):
            start_module.on_port_connection_changed(start_port.port_name, True)

        # Check if this connection involves an Output module
        from sonicrack.gui.modules.output.output import OutputModule

        # If connecting to/from Output module, start playback
        if isinstance(start_module, OutputModule) or isinstance(
            target_module, OutputModule
        ):
            logger.info("Connection to/from Output module detected - starting playback")
            self._start_output_playback()

    def _on_cable_disconnected(self, start_port: PortWidget, target_port: PortWidget):
        """Handle cable disconnection."""
        # Safety check: ports might be None during bulk module deletion
        if start_port is None or target_port is None:
            # Bulk deletion occurred, just trigger playback check
            logger.info("Bulk disconnection detected - checking playback state")
            self.audio_engine.mark_graph_changed()
            self._start_output_playback()
            self._mark_patch_modified()
            return

        start_port.port.disconnect(target_port.port)
        self.audio_engine.mark_graph_changed()
        self._mark_patch_modified()

        # Notify modules about disconnection
        start_module = start_port.parent_module
        target_module = target_port.parent_module

        # Notify target module (input port) about disconnection
        if hasattr(target_module, "on_port_connection_changed"):
            target_module.on_port_connection_changed(target_port.port_name, False)

        # Notify source module (output port) about disconnection if needed
        if hasattr(start_module, "on_port_connection_changed"):
            start_module.on_port_connection_changed(start_port.port_name, False)

        # Check if this disconnection involves an Output module
        from sonicrack.gui.modules.output.output import OutputModule

        # If disconnecting from Output module, check if we should stop playback
        if isinstance(start_module, OutputModule) or isinstance(
            target_module, OutputModule
        ):
            logger.info(
                "Disconnection from Output module detected - checking playback state"
            )
            self._start_output_playback()  # Will check connections and stop if none

    def _start_output_playback(self):
        """Start playback on the Output module through the shared render graph."""
        from sonicrack.gui.modules.output.output import OutputModule

        logger.info("=== Starting output playback ===")
        patch_canvas = self._require_patch_canvas()
        statusbar = self._require_statusbar()

        # Find the Output module
        output_module = None
        for module in patch_canvas.get_modules():
            if isinstance(module, OutputModule):
                output_module = module
                break

        if output_module is None:
            logger.debug("No Output module found")
            return

        logger.info(f"Found Output module: {output_module.metadata.title}")

        try:
            started = output_module.start_playback()
            if started:
                statusbar.showMessage("Playback started")
                logger.info("Output playback started")
            else:
                statusbar.showMessage(output_module.status_label.text())
                logger.info("Output playback remains stopped")

        except Exception as e:
            logger.error(f"Failed to start playback: {e}", exc_info=True)
            statusbar.showMessage(f"Playback error: {e}")

    def _restart_output_playback(self):
        """Restart playback to refresh audio callback with current connections.

        This is called when modules are deleted to ensure stale connections
        don't continue generating audio.
        """
        from sonicrack.gui.modules.output.output import OutputModule

        logger.info("=== Restarting playback (refreshing audio callback) ===")
        patch_canvas = self._require_patch_canvas()

        # Find the Output module
        output_module = None
        for module in patch_canvas.get_modules():
            if isinstance(module, OutputModule):
                output_module = module
                break

        if not output_module:
            logger.debug("No Output module found")
            return

        output_module = cast(OutputModule, output_module)

        # Stop playback first (if running)
        if output_module.audio_output.is_playing:
            logger.info("Stopping playback to refresh audio callback")
            output_module.stop_playback()

        # Small delay to ensure clean stop
        QTimer.singleShot(100, lambda: self._delayed_start_playback(output_module))

    def _delayed_start_playback(self, output_module: OutputModule):
        """Start playback after a short delay.

        Args:
            output_module: The OutputModule to start playback on
        """
        if not output_module.is_active:
            logger.info("Skipping delayed playback restart for inactive Output module")
            return

        try:
            logger.info("Restarting playback with refreshed connections")
            output_module.start_playback()
            self._require_statusbar().showMessage("Playback restarted (refreshed)")
        except Exception as e:
            logger.error(f"Failed to restart playback: {e}", exc_info=True)
            self._require_statusbar().showMessage(f"Playback error: {e}")

    def _on_module_deleted(self, module):
        """Handle module deletion.

        Args:
            module: The module that was deleted
        """
        logger.info(f"Module deleted: {module.metadata.title}")
        self._mark_patch_modified()

        shutdown = getattr(module, "shutdown", None)
        if callable(shutdown):
            shutdown(graceful=True)

        with contextlib.suppress(ValueError):
            self.audio_engine.modules.remove(module)
        self.audio_engine.mark_graph_changed()

        from sonicrack.gui.modules.output.output import OutputModule

        if isinstance(module, OutputModule):
            return

        # Check if we need to stop/update playback
        # (This will stop playback if Output module has no more connections)
        self._start_output_playback()

    def _on_parameter_changed(self, param_name: str, value):
        """Handle module parameter change using hot-swapping (no recompile).

        This method implements Strategy 1 (Parameter Hot-Swapping) to eliminate
        clicking when adjusting parameters. Instead of recompiling the entire
        patch, it updates the parameter directly in the compiled component.

        Args:
            param_name: Name of the changed parameter
            value: New value
        """
        logger.debug(f"Parameter changed: {param_name} = {value}")

        # Mark patch as modified
        self._mark_patch_modified()

    def _update_window_title(self):
        """Update window title to show current patch name and modified status."""
        base_title = APP_TITLE

        if self.current_patch_path:
            patch_name = Path(self.current_patch_path).stem
            title = f"{patch_name} - {base_title}"
        else:
            title = f"Untitled - {base_title}"

        # Add asterisk if modified
        if self.patch_modified:
            title = f"*{title}"

        self.setWindowTitle(title)

    def _open_settings_dialog(self):
        """Open global application settings dialog (audio, etc.)."""
        from sonicrack.gui.dialogs.audio_settings_dialog import AudioSettingsDialog

        dialog = AudioSettingsDialog(self)
        dialog.exec()

    def _mark_patch_modified(self):
        """Mark the patch as modified (has unsaved changes)."""
        if not self.patch_modified:
            self.patch_modified = True
            self._update_window_title()

    def _current_patch_data(self) -> dict[str, Any]:
        """Return the current canvas serialized as patch data."""
        patch_canvas = self._require_patch_canvas()
        metadata: dict[str, Any] = {
            "name": (
                Path(self.current_patch_path).stem
                if self.current_patch_path
                else "Last Session"
            )
        }
        patch_data = self.preset_manager._serialize_patch(
            patch_canvas.get_modules(),
            patch_canvas.get_connections(),
            metadata,
        )
        patch_data["autosave"] = {
            "current_patch_path": self.current_patch_path,
            "patch_modified": self.patch_modified,
        }
        return patch_data

    def _save_last_patch(self) -> bool:
        """Persist the current in-memory patch for restoration on next launch."""
        try:
            self._autosave_patch_path.parent.mkdir(parents=True, exist_ok=True)
            patch_data = self._current_patch_data()
            with open(self._autosave_patch_path, "w", encoding="utf-8") as f:
                json.dump(patch_data, f, indent=2)
            logger.info("Autosaved last patch to %s", self._autosave_patch_path)
            return True
        except Exception as exc:
            logger.warning(
                "Failed to autosave last patch to %s: %s",
                self._autosave_patch_path,
                exc,
                exc_info=True,
            )
            return False

    def _restore_last_patch(self) -> bool:
        """Restore the last autosaved patch, if one exists."""
        if not self._autosave_patch_path.is_file():
            return False

        try:
            with open(self._autosave_patch_path, encoding="utf-8") as f:
                patch_data = json.load(f)

            self._apply_preset(patch_data)

            autosave_data = patch_data.get("autosave", {})
            current_patch_path = autosave_data.get("current_patch_path")
            self.current_patch_path = (
                current_patch_path if isinstance(current_patch_path, str) else None
            )
            self.patch_modified = bool(autosave_data.get("patch_modified", False))
            self._update_window_title()

            self._require_statusbar().showMessage("Last patch restored")
            logger.info("Restored last patch from %s", self._autosave_patch_path)
            return True
        except Exception as exc:
            logger.warning(
                "Failed to restore last patch from %s: %s",
                self._autosave_patch_path,
                exc,
                exc_info=True,
            )
            return False

    def _save_patch(self):
        """Save the current patch into a file.

        If no path exists, prompts for Save As."""
        if self.current_patch_path:
            # Save to existing file
            self._do_save_patch(self.current_patch_path)
        else:
            # No path yet - do Save As
            self._save_patch_as()

    def _save_patch_as(self):
        """Save the current patch with a new name."""
        from PyQt6.QtWidgets import QFileDialog

        modules = self.patch_canvas.get_modules()
        if not modules:
            QMessageBox.warning(
                self,
                "No Modules",
                "Add some modules to the canvas before saving.",
            )
            return

        # Default to Documents directory
        default_dir = str(Path.home() / "Documents")

        # Show file dialog with proper filter format
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Patch As",
            default_dir,
            f"Patch Files (*{PRESET_FILE_EXTENSION});;All Files (*.*)",
        )

        # Check if user cancelled
        if not file_path:
            return

        # Ensure proper extension
        if not file_path.endswith(PRESET_FILE_EXTENSION):
            file_path += PRESET_FILE_EXTENSION

        self._do_save_patch(file_path)

    def _do_save_patch(self, file_path: str):
        """Actually save the patch to a file.

        Args:
            file_path: Path to save the patch to
        """
        patch_canvas = self._require_patch_canvas()

        # Validate file path
        if not file_path:
            logger.warning("Save cancelled - no file path provided")
            return

        modules = patch_canvas.get_modules()
        connections = patch_canvas.get_connections()
        metadata: dict[str, Any] = {}

        success = self.preset_manager.save_preset(
            modules, connections, metadata, file_path
        )

        if not success:
            QMessageBox.critical(
                self, "Save Error", f"Failed to save patch to:\n{file_path}"
            )
            logger.error(f"Failed to save patch to {file_path}")
            return

        # Update state
        self.current_patch_path = file_path
        self.patch_modified = False
        self._update_window_title()

        self._require_statusbar().showMessage(f"Patch saved: {Path(file_path).name}")
        logger.info(f"Patch saved to {file_path}")

    def _open_patch(self):
        """Open a patch file via user-dialog."""

        # Check for unsaved changes
        if self.patch_modified:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save before opening?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
            )

            if reply == QMessageBox.StandardButton.Yes:
                self._save_patch()
            elif reply == QMessageBox.StandardButton.Cancel:
                return

        # Default to Documents directory
        default_dir = str(Path.home() / "Documents")

        # Show file dialog with proper filter format
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Patch",
            default_dir,
            f"Patch Files (*{PRESET_FILE_EXTENSION});;All Files (*.*)",
        )

        if file_path:
            self._do_load_patch(file_path)

    def _do_load_patch(self, file_path: str):
        """Actually open a patch from a file.

        Args:
            file_path: Path to the patch file
        """
        import json
        from pathlib import Path

        try:
            # Load patch data
            with open(file_path) as f:
                patch_data = json.load(f)

            # Clear current patch
            self._clear_canvas()

            # Apply the patch
            self._apply_preset(patch_data)

            # Update state
            self.current_patch_path = file_path
            self.patch_modified = False
            self._update_window_title()

            self._require_statusbar().showMessage(
                f"Patch loaded: {Path(file_path).name}"
            )
            logger.info(f"Patch loaded from {file_path}")

        except Exception as e:
            QMessageBox.critical(
                self,
                "Open Error",
                f"Failed to open patch:\n{str(e)}",
            )
            logger.error(f"Failed to open patch: {e}")

    def _new_patch(self):
        """Create a new patch."""
        # Check for unsaved changes
        if self.patch_modified:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save before creating a new "
                "patch?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
            )

            if reply == QMessageBox.StandardButton.Yes:
                self._save_patch()
            elif reply == QMessageBox.StandardButton.Cancel:
                return

        # Clear and reset
        self._clear_canvas()
        self.current_patch_path = None
        self.patch_modified = False
        self._update_window_title()

    def _clear_canvas(self):
        """Clear the patch canvas."""
        if self.patch_canvas is not None:
            for module in list(self.patch_canvas.get_modules()):
                shutdown = getattr(module, "shutdown", None)
                if callable(shutdown):
                    shutdown(graceful=True)

        # Clear the canvas
        patch_canvas = self._require_patch_canvas()
        patch_canvas.clear_all()
        self.audio_engine.modules.clear()
        self.audio_engine.mark_graph_changed()
        self._require_statusbar().showMessage("Canvas cleared")

    def _stop_all_output_modules(self, graceful: bool = True):
        """Stop playback on all Output modules in the current patch.

        This ensures that when loading a new patch or clearing the canvas,
        audio from the old patch doesn't continue playing.

        Args:
            graceful: If True, allow Output modules to fade out before stopping.
        """
        if self.patch_canvas is None:
            return

        output_module = self.patch_canvas.get_output_module()
        if output_module is None:
            return

        if output_module.audio_output.is_playing:
            logger.info("Stopping Output module before clearing patch")
            output_module.stop_playback(graceful=graceful)

    def _save_as_library_preset(self):
        """Save the current patch as a preset."""
        patch_canvas = self._require_patch_canvas()
        modules = patch_canvas.get_modules()
        connections = patch_canvas.get_connections()

        if not modules:
            QMessageBox.warning(
                self,
                "No Modules",
                "Add some modules to the canvas before saving a preset.",
            )
            return

        # Show save preset dialog
        dialog = SaveLibraryPresetDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            metadata = dialog.get_metadata()

            # Save preset
            filepath = self.preset_manager.save_preset(
                modules=modules,
                connections=connections,
                metadata=metadata,
                save_as_library_preset=True,
            )

            if filepath:
                QMessageBox.information(
                    self, "Success", f"Preset saved successfully!\n\n{filepath.name}"
                )
                self.statusbar.showMessage(f"Preset saved: {metadata['name']}")
            else:
                QMessageBox.critical(self, "Error", "Failed to save preset.")

    def _load_preset(self):
        """Load a preset and rebuild the patch."""
        # Show preset browser dialog
        dialog = LibraryPresetBrowserDialog(self.preset_manager, self)
        dialog.preset_selected.connect(self._apply_preset)
        dialog.exec()

    def _apply_preset(self, preset_data: dict):
        """Apply a loaded preset to the canvas.

        Args:
            preset_data: Dictionary containing preset data with 'modules' and
                'connections'
        """
        patch_canvas = self._require_patch_canvas()

        for module in list(patch_canvas.get_modules()):
            shutdown = getattr(module, "shutdown", None)
            if callable(shutdown):
                shutdown(graceful=True)

        # Clear current patch
        patch_canvas.clear_all()
        self.audio_engine.modules.clear()
        self.audio_engine.mark_graph_changed()

        # Rebuild modules
        module_map = {}  # Maps old module IDs to new module instances

        for module_data in preset_data.get("modules", []):
            module_type = module_data.get("type")
            module_id = module_data.get("id")
            position = module_data.get("position", {"x": 0, "y": 0})
            parameters = module_data.get("parameters", {})

            # Get module class from registry
            module_class = self.registry.get(module_type)
            if not module_class:
                logger.warning(f"Unknown module type: {module_type}")
                continue

            # Create module instance
            module_instance = module_class()

            # Set parameters
            try:
                module_instance.set_parameters(parameters)
            except Exception as e:
                logger.warning(
                    "Failed to set parameters for %s: %s",
                    module_type,
                    e,
                    exc_info=True,
                )

            custom_name = module_data.get("custom_name")
            if isinstance(custom_name, str) and custom_name:
                module_instance.set_custom_name(custom_name)

            # Add to canvas
            patch_canvas.add_module(module_instance)
            self.audio_engine.add_module(module_instance)

            if module_instance.metadata.category == ModuleCategory.OUTPUT:
                from sonicrack.gui.modules.output.output import OutputModule

                output_module = cast(OutputModule, module_instance)
                output_module.audio_engine = self.audio_engine

            # Set position
            module_instance.setPos(position["x"], position["y"])

            # Store in map
            module_map[module_id] = module_instance

        # Rebuild connections
        for connection_data in preset_data.get("connections", []):
            source_id = connection_data.get("source_module")
            source_port = connection_data.get("source_port")
            target_id = connection_data.get("target_module")
            target_port = connection_data.get("target_port")

            source_module = module_map.get(source_id)
            target_module = module_map.get(target_id)

            if source_module and target_module:
                # Find the ports
                source_port_obj = None
                target_port_obj = None

                for port in source_module.output_ports:
                    if port.port_name == source_port:
                        source_port_obj = port
                        break

                target_port_names = _target_port_names_for_restore(target_port)
                for port in target_module.input_ports:
                    if port.port_name in target_port_names:
                        target_port_obj = port
                        break

                if source_port_obj and target_port_obj:
                    # Create cable connection
                    patch_canvas.create_connection(source_port_obj, target_port_obj)
                else:
                    logger.warning(
                        f"Could not find ports: {source_port} or {target_port}"
                    )

        # Compile the loaded patch
        self._require_statusbar().showMessage("Patch loaded successfully")

    def shutdown(self, graceful: bool = True) -> None:
        """Release app-owned resources before quitting.

        Args:
            graceful: If True, allow audio modules to fade out. If False, stop
                streams and worker resources as quickly as possible.
        """
        if self._is_shutting_down:
            return

        self._is_shutting_down = True
        logger.info("Shutting down ModularSynthWindow")

        modules = []
        if self.patch_canvas is not None:
            modules = list(self.patch_canvas.get_modules())

        for module in modules:
            try:
                shutdown = getattr(module, "shutdown", None)
                if callable(shutdown):
                    shutdown(graceful=graceful)
                    continue

                stop_playback = getattr(module, "stop_playback", None)
                if callable(stop_playback):
                    stop_playback(graceful=graceful)

                timer = getattr(module, "_viz_timer", None)
                if timer is not None and hasattr(timer, "stop"):
                    timer.stop()
            except Exception as exc:
                logger.warning(
                    "Error while shutting down module %s: %s",
                    type(module).__name__,
                    exc,
                    exc_info=True,
                )

    def closeEvent(self, event):
        """Handle window close event."""
        self._save_last_patch()
        self.shutdown(graceful=True)
        event.accept()
