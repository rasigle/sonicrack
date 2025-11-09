"""Main window for the modular synthesizer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6 import QtGui, QtWidgets, QtCore
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QToolBar,
    QLabel,
    QStatusBar,
    QMessageBox,
    QGroupBox,
    QScrollArea,
    QSplitter,
    QDialog,
    QFileDialog,
)

from src import version
from src.constants import PRESET_FILE_EXTENSION
from src.gui.audio_engine import AudioEngine
from src.gui.audio_module_interface import ModuleCategory
from src.gui.dialogs.about_dialog import show_about
from src.gui.dialogs.preset_library_dialog import (
    LibraryPresetBrowserDialog,
    SaveLibraryPresetDialog,
)
from src.gui.module_registry import initialize_modules
from src.gui.modules.output.output import OutputModule
from src.gui.patch_canvas import PatchCanvas
from src.gui.patch_compiler import PatchCompiler
from src.gui.preset_manager import PresetManager
from src.gui.ui_constants import APP_TITLE, APP_ICON_PATH, DEBOUNCE_TIMER_DELAY_MS

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.engine import AudioComponent


class ModularSynthWindow(QMainWindow):
    """Main window for the modular synthesizer application.

    Provides a complete modular synthesis environment with:
    - Patch canvas for visual module patching
    - Module library
    - Real-time audio playback
    - Waveform and spectrum visualization
    - Preset management (future)
    """

    def __init__(self):
        """Initialize the main window."""
        super().__init__()

        # Initialize the module registry with all built-in modules
        print("Main")
        self.registry = initialize_modules()

        self.setWindowTitle(APP_TITLE)
        self.setGeometry(100, 100, 1400, 900)

        # Set application icon
        if not APP_ICON_PATH.exists():
            logger.warning(f"App icon not found at {APP_ICON_PATH}")
        else:
            self.setWindowIcon(QtGui.QIcon(str(APP_ICON_PATH)))

        # Core components
        self.audio_engine = AudioEngine()
        self.patch_compiler = PatchCompiler()
        self.preset_manager = PresetManager()

        # Patch file tracking
        self.current_patch_path = None  # Path to currently loaded patch file
        self.patch_modified = False  # Track if patch has unsaved changes

        # Debounce timer for parameter changes (avoid audio spikes)
        self.compile_debounce_timer = QtCore.QTimer()
        self.compile_debounce_timer.setSingleShot(True)
        self.compile_debounce_timer.timeout.connect(
            lambda: self._compile_patch()
        )

        # UI setup
        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_statusbar()
        self._connect_signals()

        # Visualization timer
        self.vis_timer = QtCore.QTimer()
        self.vis_timer.timeout.connect(self._update_visualizations)
        self.vis_timer.start(50)  # Update at 20 Hz

        logger.info("Modular Synth Window initialized")

    def _setup_ui(self):
        """Setup the user interface."""
        # Central widget with splitter
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        # Main splitter (horizontal)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - Module Library
        self.module_library = self._create_module_library()
        main_splitter.addWidget(self.module_library)

        # Center - Patch Canvas
        canvas_widget = QWidget()
        canvas_layout = QVBoxLayout(canvas_widget)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        # Patch canvas
        self.patch_canvas = PatchCanvas()
        canvas_layout.addWidget(self.patch_canvas)

        main_splitter.addWidget(canvas_widget)

        # Right panel - Visualizations
        viz_widget = self._create_visualization_panel()
        main_splitter.addWidget(viz_widget)

        # Set splitter sizes
        main_splitter.setSizes([200, 800, 400])

        layout.addWidget(main_splitter)

    def _create_module_library(self) -> QWidget:
        """Create the module library panel.

        Returns:
            Widget containing the module library
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Title
        title = QLabel("Module Library")
        title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        layout.addWidget(title)

        # Module buttons
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(180)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Add module buttons
        for module_name in self.registry.list_modules():
            btn = QtWidgets.QPushButton(f"+ {module_name}")
            btn.setMinimumHeight(35)
            btn.clicked.connect(
                lambda checked, name=module_name: self._add_module(name)
            )
            scroll_layout.addWidget(btn)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        return panel

    def _create_visualization_panel(self) -> QWidget:
        """Create the visualization panel.

        Returns:
            Widget containing visualizations
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)

        # Control panel
        controls_group = QGroupBox("Playback")
        controls_layout = QVBoxLayout()

        # Play/Stop buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("▶ Play")
        self.play_btn.setMinimumHeight(40)
        self.play_btn.clicked.connect(self._on_play_clicked)
        btn_layout.addWidget(self.play_btn)

        self.stop_btn = QtWidgets.QPushButton("⬛ Stop")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        btn_layout.addWidget(self.stop_btn)

        controls_layout.addLayout(btn_layout)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        layout.addStretch()

        panel.setMinimumWidth(350)
        return panel

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

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QtGui.QAction("&About", self)
        about_action.triggered.connect(lambda: show_about(self, version=version))
        help_menu.addAction(about_action)

    def _setup_toolbar(self):
        """Setup the toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Quick access buttons
        toolbar.addAction("New", self._new_patch)
        toolbar.addAction("Clear", self._clear_canvas)

    def _setup_statusbar(self):
        """Setup the status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Ready")

    def _connect_signals(self):
        """Connect signals."""
        # Audio engine signals
        self.audio_engine.playback_started.connect(self._on_playback_started)
        self.audio_engine.playback_stopped.connect(self._on_playback_stopped)
        self.audio_engine.error_occurred.connect(self._on_audio_error)

        # Patch canvas signals
        self.patch_canvas.cable_connected.connect(self._on_cable_connected)
        self.patch_canvas.cable_disconnected.connect(self._on_cable_disconnected)
        self.patch_canvas.module_deleted.connect(self._on_module_deleted)

    def _add_module(self, module_name: str):
        """Add a module to the canvas.

        Args:
            module_name: Name of the module type to add
        """
        module_class = self.registry.get(module_name, strict=True)

        # Check if trying to add an Output module when one already exists
        if module_class.metadata.category == ModuleCategory.OUTPUT:

            # Check if an Output module already exists
            for module in self.patch_canvas.get_modules():
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
                    self.statusbar.showMessage("Cannot add multiple Output modules")
                    logger.warning("Attempted to add multiple Output modules")
                    return

        # Initialize the module to add with default parameters
        module_instance = module_class()

        # Connect parameter change signal to auto-compile
        module_instance.parameter_changed.connect(self._on_parameter_changed)

        # Special handling for Output module - connect master volume directly
        if isinstance(module_instance, OutputModule):
            module_instance.master_volume_changed.connect(
                self.audio_engine.set_master_volume
            )
            # Emit initial volume to sync audio engine (use QTimer to ensure
            # connections are ready)
            QTimer.singleShot(0, module_instance._on_volume_changed)

        self.patch_canvas.add_module(module_instance)
        self.statusbar.showMessage(f"Added {module_name}")
        logger.info(f"Added module: {module_name}")

        # Mark patch as modified
        self._mark_patch_modified()

        # Auto-compile when module is added
        self._compile_patch()

    def _compile_patch(self, show_messages: bool = False) -> bool:
        """Compile the current patch automatically.

        Args:
            show_messages: If True, show message boxes for errors/success

        Returns:
            True if compilation succeeded, False otherwise
        """
        # Get all modules
        modules = self.patch_canvas.get_modules()
        if not modules:
            if show_messages:
                QMessageBox.warning(
                    self, "No Modules", "Add some modules to the canvas first!"
                )
            # Clear patch and stop playback
            self.audio_engine.clear_audiopath()
            return False

        # Get all connections
        connections = self.patch_canvas.get_connections()
        logger.debug(f"Canvas has {len(connections)} connections")
        if not connections:
            self.audio_engine.clear_audiopath()
            return False

        # Define the compile scope and check for errors
        self.patch_compiler.set_patch(modules, connections)
        errors = self.patch_compiler.get_prevalidation_errors()

        if errors:
            if show_messages:
                error_msg = "Patch has errors:\n\n" + "\n".join(
                    f"• {err}" for err in errors
                )
                QMessageBox.warning(self, "Compilation Errors", error_msg)
            logger.warning(f"Patch compilation errors: {errors}")

            # Clear patch and stop playback on error
            self.audio_engine.clear_audiopath()
            return False

        # Now we are read to compile
        audio_patch: AudioComponent = self.patch_compiler.compile()
        if audio_patch is None:

            if show_messages:
                QMessageBox.critical(
                    self, "Compilation Failed", "Failed to compile patch."
                )
            logger.error("Patch compilation failed")
            # Clear patch on failure
            self.audio_engine.clear_audiopath()
            return False

        self.audio_engine.set_audiopatch(audio_patch)

        # Set the master volume from output module
        output_module = self.patch_canvas.get_output_module()
        if output_module is not None:
            self.audio_engine.set_master_volume(output_module.get_master_volume())

        if show_messages:
            QMessageBox.information(self, "Success", "Patch compiled successfully!")
        self.statusbar.showMessage("Patch compiled and ready")
        logger.debug("Patch compiled successfully")
        return True

    def _on_play_clicked(self):
        """Handle play button click."""
        # Auto-compile before playing if patch not set
        if self.audio_engine.patch is None:
            if not self._compile_patch():
                # Compilation failed
                return

        self.audio_engine.start_playback()

    def _on_stop_clicked(self):
        """Handle stop button click."""
        self.audio_engine.stop_playback()

    def _on_playback_started(self):
        """Handle playback started."""
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.statusbar.showMessage("Playing...")

    def _on_playback_stopped(self):
        """Handle playback stopped."""
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.statusbar.showMessage("Stopped")

    def _on_audio_error(self, error: str):
        """Handle audio error."""
        QMessageBox.critical(self, "Audio Error", f"Audio error occurred:\n{error}")
        self.statusbar.showMessage(f"Error: {error}")

    def _on_cable_connected(self, start_port, end_port):
        """Handle cable connection."""
        logger.debug(f"Cable connected: {start_port.port_name} -> {end_port.port_name}")

        # Mark patch as modified
        self._mark_patch_modified()

        # Update UI state of affected modules BEFORE compilation
        affected_modules = set()
        if start_port and start_port.parent_module:
            affected_modules.add(start_port.parent_module)
        if end_port and end_port.parent_module:
            affected_modules.add(end_port.parent_module)

        for module in affected_modules:
            if hasattr(module, "update_knob_state"):
                try:
                    module.update_knob_state()
                except Exception as e:
                    logger.warning(
                        f"Failed to update knob state for {module.metadata.title}: {e}"
                    )

        # Auto-compile when connection changes
        self._compile_patch()

    def _on_cable_disconnected(self, start_port, end_port):
        """Handle cable disconnection."""
        logger.debug(
            f"Cable disconnected: {start_port.port_name} -> {end_port.port_name}"
        )

        # Mark patch as modified
        self._mark_patch_modified()

        # Update UI state of affected modules BEFORE compilation
        # This ensures knobs update even if compilation fails
        affected_modules = set()
        if start_port and start_port.parent_module:
            affected_modules.add(start_port.parent_module)
        if end_port and end_port.parent_module:
            affected_modules.add(end_port.parent_module)

        for module in affected_modules:
            if hasattr(module, "update_knob_state"):
                try:
                    module.update_knob_state()
                except Exception as e:
                    logger.warning(
                        f"Failed to update knob state for {module.metadata.title}: {e}"
                    )

        # Check if output module was disconnected
        output_module_class = self.registry.get("Output")
        if output_module_class and isinstance(
            end_port.parent_module, output_module_class
        ):
            # Output was disconnected - stop playback and clear patch
            self.audio_engine.stop_playback()
            self.audio_engine.clear_audiopath()
            logger.info("Output disconnected - playback stopped")

        # Automatically recompile patch to update audio chain
        # Even if compilation fails, UI was already updated above
        logger.info("Cable disconnected - auto-recompiling patch to update audio chain")

        try:
            self._compile_patch()
        except Exception as e:
            logger.error(f"Compilation failed after cable disconnect: {e}")
            # UI state was already updated, so this is acceptable

    def _on_module_deleted(self, module):
        """Handle module deletion.

        Args:
            module: The module that was deleted
        """
        logger.info(f"Module deleted: {module.metadata.title}")

        # Mark patch as modified
        self._mark_patch_modified()

        # If Output module was deleted, stop playback immediately
        if module.metadata.category == ModuleCategory.OUTPUT:
            self.audio_engine.stop_playback()
            self.audio_engine.clear_audiopath()
            logger.info("Output module deleted - playback stopped")
        else:
            # For other modules, check if we're playing
            if self.audio_engine.is_playing:
                # Recompile to update the patch
                # If compilation fails (e.g., no output), it will handle stopping
                self._compile_patch()

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

        # Find which module emitted this signal
        sender_module = self.sender()

        # Validate sender is an AudioModuleInterface
        from src.gui.audio_module_interface import AudioModule

        if isinstance(sender_module, AudioModule):
            # Check if module is actually in the compiled patch
            is_in_patch = sender_module in self.patch_compiler._module_to_component

            if not is_in_patch:
                # Module is not connected to the audio chain - ignore parameter change
                logger.debug(
                    f"Ignoring parameter change for unconnected module: "
                    f"{sender_module.metadata.title}"
                )
                return

            # Some parameters require full recompilation (can't be hot-swapped)
            # e.g., waveform type changes create a different oscillator class
            requires_recompile = param_name in ["waveform"]

            if self.audio_engine.is_playing and not requires_recompile:
                # Hot-swap the parameter without recompiling (prevents clicks!)
                success = self.patch_compiler.update_parameter(
                    sender_module, param_name, value
                )

                if success:
                    logger.debug(f"✓ Hot-swapped {param_name} (no recompile, no click)")
                else:
                    # Fallback: debounced recompile if hot-swap fails
                    logger.warning(
                        f"Hot-swap failed for {param_name}, falling back to recompile"
                    )
                    self.compile_debounce_timer.stop()
                    self.compile_debounce_timer.start(DEBOUNCE_TIMER_DELAY_MS)
            else:
                # Not playing OR requires recompile: use debounced recompile
                # (Debouncing is less critical when not playing)
                self.compile_debounce_timer.stop()
                self.compile_debounce_timer.start(DEBOUNCE_TIMER_DELAY_MS)

    def _update_visualizations(self):
        """Update waveform and spectrum displays."""
        if self.audio_engine.current_buffer is not None:
            self.waveform_display.set_samples(self.audio_engine.current_buffer)
            self.spectrum_analyzer.set_samples(self.audio_engine.current_buffer)

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

    def _mark_patch_modified(self):
        """Mark the patch as modified (has unsaved changes)."""
        if not self.patch_modified:
            self.patch_modified = True
            self._update_window_title()

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
        # Validate file path
        if not file_path:
            logger.warning("Save cancelled - no file path provided")
            return

        modules = self.patch_canvas.get_modules()
        connections = self.patch_canvas.get_connections()
        metadata = {}

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

        self.statusbar.showMessage(f"Patch saved: {Path(file_path).name}")
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
            with open(file_path, "r") as f:
                patch_data = json.load(f)

            # Clear current patch
            self._clear_canvas()

            # Apply the patch
            self._apply_preset(patch_data)

            # Update state
            self.current_patch_path = file_path
            self.patch_modified = False
            self._update_window_title()

            self.statusbar.showMessage(f"Patch loaded: {Path(file_path).name}")
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
        self.audio_engine.stop_playback()
        self.audio_engine.clear_audiopath()
        self.patch_canvas.clear_all()
        self.statusbar.showMessage("Canvas cleared")

    def _save_as_library_preset(self):
        """Save the current patch as a preset."""
        modules = self.patch_canvas.get_modules()
        connections = self.patch_canvas.get_connections()

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
        # Clear current patch
        self.audio_engine.stop_playback()
        self.patch_canvas.clear_all()

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
            for param_name, param_value in parameters.items():
                if hasattr(module_instance, "set_parameter"):
                    try:
                        module_instance.set_parameter(param_name, param_value)
                    except Exception as e:
                        logger.warning(f"Failed to set parameter {param_name}: {e}")

            # Connect parameter change signal
            module_instance.parameter_changed.connect(self._on_parameter_changed)

            # Special handling for Output module
            if isinstance(module_instance, OutputModule):
                module_instance.master_volume_changed.connect(
                    self.audio_engine.set_master_volume
                )
                QTimer.singleShot(0, module_instance._on_volume_changed)

            # Add to canvas
            self.patch_canvas.add_module(module_instance)

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

                for port in target_module.input_ports:
                    if port.port_name == target_port:
                        target_port_obj = port
                        break

                if source_port_obj and target_port_obj:
                    # Create cable connection
                    self.patch_canvas.create_connection(
                        source_port_obj, target_port_obj
                    )
                else:
                    logger.warning(
                        f"Could not find ports: {source_port} or {target_port}"
                    )

        # Compile the loaded patch
        self._compile_patch()
        self.statusbar.showMessage("Preset loaded successfully")

    def closeEvent(self, event):
        """Handle window close event."""
        self.audio_engine.cleanup()
        event.accept()
