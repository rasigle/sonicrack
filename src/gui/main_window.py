"""Main window for the modular synthesizer."""

import logging
from typing import Any

from PyQt6.QtCore import Qt, QTimer, QPointF
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QToolBar,
    QLabel,
    QStatusBar,
    QMessageBox,
    QGroupBox,
    QScrollArea,
    QSplitter,
    QDialog,
)

from src.gui.audio_engine import AudioEngine
from src.gui.dialogs.about_dialog import show_about
from src.gui.dialogs.preset_dialog import PresetBrowserDialog, SavePresetDialog
from src.gui.module_registry import initialize_modules
from src.gui.patch_canvas import PatchCanvas
from src.gui.patch_compiler import PatchCompiler
from src.gui.preset_manager import PresetManager
from src.gui.widgets.spectrum_analyzer import SpectrumAnalyzer
from src.gui.widgets.tree_analyzer import TreeAnalyzer
from src.gui.widgets.waveform_display import WaveformDisplay

logger = logging.getLogger(__name__)


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
        self.registry = initialize_modules()

        self.setWindowTitle("AudioPlayground - Modular Synthesizer")
        self.setGeometry(100, 100, 1400, 900)

        # Core components
        self.audio_engine = AudioEngine()
        self.patch_compiler = PatchCompiler()
        self.preset_manager = PresetManager()

        # Debounce timer for parameter changes (avoid audio spikes)
        self.compile_debounce_timer = QTimer()
        self.compile_debounce_timer.setSingleShot(True)
        self.compile_debounce_timer.timeout.connect(lambda: self._compile_patch())

        # UI setup
        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_statusbar()
        self._connect_signals()

        # Visualization timer
        self.vis_timer = QTimer()
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
            btn = QPushButton(f"+ {module_name}")
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

        # Title
        title = QLabel("Visualizations")
        title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        layout.addWidget(title)

        # Waveform display
        wave_group = QGroupBox("Waveform")
        wave_layout = QVBoxLayout()
        self.waveform_display = WaveformDisplay()
        wave_layout.addWidget(self.waveform_display)
        wave_group.setLayout(wave_layout)
        layout.addWidget(wave_group)

        # Spectrum analyzer
        spectrum_group = QGroupBox("Spectrum")
        spectrum_layout = QVBoxLayout()
        self.spectrum_analyzer = SpectrumAnalyzer()
        spectrum_layout.addWidget(self.spectrum_analyzer)
        spectrum_group.setLayout(spectrum_layout)
        layout.addWidget(spectrum_group)

        # Tree analyzer
        tree_group = QGroupBox("Patch Tree")
        tree_layout = QVBoxLayout()
        self.tree_analyzer = TreeAnalyzer()
        tree_layout.addWidget(self.tree_analyzer)
        tree_group.setLayout(tree_layout)
        layout.addWidget(tree_group)

        # Control panel
        controls_group = QGroupBox("Playback")
        controls_layout = QVBoxLayout()

        # Play/Stop buttons
        btn_layout = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setMinimumHeight(40)
        self.play_btn.clicked.connect(self._on_play_clicked)
        btn_layout.addWidget(self.play_btn)

        self.stop_btn = QPushButton("⬛ Stop")
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
        """Setup the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        new_action = QAction("&New Patch", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self._new_patch)
        file_menu.addAction(new_action)

        file_menu.addSeparator()

        save_preset_action = QAction("&Save Preset...", self)
        save_preset_action.setShortcut("Ctrl+S")
        save_preset_action.triggered.connect(self._save_preset)
        file_menu.addAction(save_preset_action)

        load_preset_action = QAction("&Load Preset...", self)
        load_preset_action.setShortcut("Ctrl+O")
        load_preset_action.triggered.connect(self._load_preset)
        file_menu.addAction(load_preset_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")

        clear_action = QAction("&Clear Canvas", self)
        clear_action.triggered.connect(self._clear_canvas)
        edit_menu.addAction(clear_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(show_about)
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

    def _add_module(self, module_name: str):
        """Add a module to the canvas.

        Args:
            module_name: Name of the module type to add
        """
        module_class = self.registry.get(module_name)
        if module_class:
            module = module_class()

            # Connect parameter change signal to auto-compile
            module.parameter_changed.connect(self._on_parameter_changed)

            self.patch_canvas.add_module(module)
            self.statusbar.showMessage(f"Added {module_name}")
            logger.info(f"Added module: {module_name}")

            # Auto-compile when module is added
            self._compile_patch()
        else:
            logger.error(f"Module not found: {module_name}")

    def _compile_patch(self, show_messages: bool = False) -> bool:
        """Compile the current patch automatically.

        Args:
            show_messages: If True, show message boxes for errors/success

        Returns:
            True if compilation succeeded, False otherwise
        """
        # Get all modules and connections
        modules = [
            item for item in self.patch_canvas.scene.items() if is_module_widget(item)
        ]
        connections = self.patch_canvas.get_connections()

        if not modules:
            if show_messages:
                QMessageBox.warning(
                    self, "No Modules", "Add some modules to the canvas first!"
                )
            # Clear patch and stop playback
            self.audio_engine.set_patch(None)
            self.tree_analyzer.clear()
            return False

        # Check for errors
        self.patch_compiler.set_patch(modules, connections)
        errors = self.patch_compiler.get_compilation_errors()

        if errors:
            if show_messages:
                error_msg = "Patch has errors:\n\n" + "\n".join(
                    f"• {err}" for err in errors
                )
                QMessageBox.warning(self, "Compilation Errors", error_msg)
            logger.warning(f"Patch compilation errors: {errors}")

            # Clear patch and stop playback on error
            self.audio_engine.set_patch(None)
            self.tree_analyzer.clear()
            return False

        # Compile
        patch = self.patch_compiler.compile()

        logging.info(patch)
        if patch:
            self.audio_engine.set_patch(patch)

            # Update tree analyzer with patch structure
            tree_data = self.patch_compiler.build_patch_tree()
            self.tree_analyzer.update_tree(tree_data)

            # Get master volume from output module
            output_module_class = self.registry.get("Output")
            if output_module_class:
                for module in modules:
                    if isinstance(module, output_module_class):
                        self.audio_engine.set_master_volume(module.get_master_volume())
                        break

            if show_messages:
                QMessageBox.information(self, "Success", "Patch compiled successfully!")
            self.statusbar.showMessage("Patch compiled and ready")
            logger.info("Patch compiled successfully")
            return True
        else:
            # Clear tree on compilation failure
            self.tree_analyzer.clear()

            if show_messages:
                QMessageBox.critical(
                    self, "Compilation Failed", "Failed to compile patch."
                )
            logger.error("Patch compilation failed")
            # Clear patch on failure
            self.audio_engine.set_patch(None)
            return False

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
        # Auto-compile when connection changes
        self._compile_patch()

    def _on_cable_disconnected(self, start_port, end_port):
        """Handle cable disconnection."""
        logger.debug(
            f"Cable disconnected: {start_port.port_name} -> {end_port.port_name}"
        )

        # Check if output module was disconnected
        output_module_class = self.registry.get("Output")
        if output_module_class and isinstance(
            end_port.parent_module, output_module_class
        ):
            # Output was disconnected - stop playback and clear patch
            self.audio_engine.stop_playback()
            self.audio_engine.set_patch(None)
            logger.info("Output disconnected - playback stopped")

        # Auto-compile when connection changes
        self._compile_patch()

    def _on_parameter_changed(self, param_name: str, value):
        """Handle module parameter change.

        Args:
            param_name: Name of the changed parameter
            value: New value
        """
        logger.debug(f"Parameter changed: {param_name} = {value}")
        # Debounce compilation to avoid audio spikes during knob rotation
        # Wait 100ms after last change before recompiling
        self.compile_debounce_timer.stop()
        self.compile_debounce_timer.start(100)  # 100ms delay

    def _update_visualizations(self):
        """Update waveform and spectrum displays."""
        if self.audio_engine.current_buffer is not None:
            self.waveform_display.set_samples(self.audio_engine.current_buffer)
            self.spectrum_analyzer.set_samples(self.audio_engine.current_buffer)

    def _new_patch(self):
        """Create a new patch."""
        reply = QMessageBox.question(
            self,
            "New Patch",
            "Clear the current patch?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._clear_canvas()
            self.statusbar.showMessage("New patch created")

    def _clear_canvas(self):
        """Clear the patch canvas."""
        self.audio_engine.stop_playback()
        self.audio_engine.set_patch(None)  # Clear compiled patch
        self.patch_canvas.clear_all()
        self.waveform_display.clear()
        self.spectrum_analyzer.clear()
        self.statusbar.showMessage("Canvas cleared")

    def _save_preset(self):
        """Save the current patch as a preset."""
        # Get all modules and connections
        modules = [
            item
            for item in self.patch_canvas.scene.items()
            if hasattr(item, "component_category")
        ]
        connections = self.patch_canvas.get_connections()

        if not modules:
            QMessageBox.warning(
                self,
                "No Modules",
                "Add some modules to the canvas before saving a preset.",
            )
            return

        # Show save preset dialog
        dialog = SavePresetDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            metadata = dialog.get_metadata()

            # Save preset
            filepath = self.preset_manager.save_preset(
                modules=modules,
                connections=connections,
                name=metadata["name"],
                author=metadata["author"],
                description=metadata["description"],
                tags=metadata["tags"],
                category=metadata["category"],
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
        dialog = PresetBrowserDialog(self.preset_manager, self)
        dialog.preset_selected.connect(self._apply_preset)
        dialog.exec()

    def _apply_preset(self, preset_data: dict):
        """Apply a loaded preset to the canvas.

        Args:
            preset_data: Preset data dictionary
        """
        try:
            # Stop playback first
            self.audio_engine.stop_playback()

            # Clear current patch
            self.patch_canvas.clear_all()

            # Create modules
            module_map = {}  # Maps preset module IDs to actual module instances

            for module_data in preset_data.get("modules", []):
                module_type = module_data["type"]
                module_id = module_data["id"]

                module_class = self.registry.get(module_type)
                if module_class:
                    # Create module
                    module = module_class()

                    # Connect parameter change signal
                    module.parameter_changed.connect(self._on_parameter_changed)

                    # Restore custom name if present
                    custom_name = module_data.get("custom_name", "")
                    if custom_name:
                        module.set_custom_name(custom_name)

                    # Set position
                    pos_data = module_data.get("position", {})
                    x = pos_data.get("x", 0)
                    y = pos_data.get("y", 0)

                    # Add to canvas
                    self.patch_canvas.add_module(module, QPointF(x, y))

                    # Set parameters
                    params = module_data.get("parameters", {})
                    module.set_parameters(params)

                    # Store in map
                    module_map[module_id] = module
                else:
                    logger.warning(f"Unknown module type: {module_type}")

            # Create connections
            for conn_data in preset_data.get("connections", []):
                from_module_id = conn_data["from_module"]
                from_port_idx = conn_data["from_port"]
                to_module_id = conn_data["to_module"]
                to_port_idx = conn_data["to_port"]

                # Get modules
                from_module = module_map.get(from_module_id)
                to_module = module_map.get(to_module_id)

                if from_module and to_module:
                    # Get ports
                    if from_port_idx < len(from_module.output_ports):
                        from_port = from_module.output_ports[from_port_idx]

                        if to_port_idx < len(to_module.input_ports):
                            to_port = to_module.input_ports[to_port_idx]

                            # Create cable
                            from src.gui.patch_canvas import Cable

                            cable = Cable(from_port, to_port)
                            self.patch_canvas.scene.addItem(cable)

            # Auto-compile the loaded patch
            self._compile_patch()

            # Show success message
            preset_name = preset_data.get("metadata", {}).get("name", "Unknown")
            self.statusbar.showMessage(f"Loaded preset: {preset_name}")
            logger.info(f"Preset loaded: {preset_name}")

        except Exception as e:
            logger.error(f"Failed to apply preset: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to load preset:\n{e}")

    def closeEvent(self, event):
        """Handle window close event."""
        self.audio_engine.cleanup()
        event.accept()


def is_module_widget(obj: Any) -> bool:
    """Check if an object is a ModuleWidget.

    Args:
        obj: Object to check

    Returns:
        True if obj is a ModuleWidget, False otherwise
    """
    from src.gui.audio_module_interface import AudioModuleInterface

    return isinstance(obj, AudioModuleInterface)
