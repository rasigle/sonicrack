"""MIDI Input module for the modular synthesizer GUI.

This module provides real-time MIDI input from connected controllers and keyboards.
It converts MIDI messages to control voltages (CV) that can control oscillators
and other synthesis parameters.

Outputs:
    - Freq: Frequency CV based on MIDI note (Hz)
    - Gate: Gate signal (1.0 = note on, 0.0 = note off)
    - Vel: Velocity CV (0.0 to 1.0)

Features:
    - Device selection from available MIDI inputs
    - Real-time message processing
    - Visual feedback (active note display)
    - Polyphonic support via last-note-priority
    - Pitch bend support

Usage:
    1. Add MIDI Input module to patch
    2. Select MIDI device from dropdown
    3. Connect outputs to oscillator/envelope inputs
    4. Play notes on your MIDI controller

Example Patch:
    MIDI Input (Freq) → Oscillator (Freq)
    MIDI Input (Gate) → ADSR Envelope → Volume (Mod)
    MIDI Input (Vel)  → [Future: velocity-sensitive parameter]
"""

import logging
from typing import Any, Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
)

from src.engine.io.midi import (
    MIDIToCV,
    MIDIMessage,
    NoteOnMessage,
    NoteOffMessage,
    CVFrequencyOutput,
    CVGateOutput,
    CVVelocityOutput,
)
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.module_registry import register_module
from src.gui.modules.input.midi_worker_thread import MIDIWorkerThread
from src.gui.widgets.module_widget import ModuleWidget

logger = logging.getLogger(__name__)


@register_module()
class MIDIInputModule(ModuleWidget):
    """MIDI Input module with device selection and CV outputs.

    This module receives MIDI messages from a connected controller/keyboard
    and converts them to control voltages for modulating synthesis parameters.

    The module runs a background thread to receive MIDI messages and updates
    the CV converter in real-time.
    """

    metadata = ModuleMetadata(
        title="MIDI Input",
        category=ModuleCategory.SOURCE,
        description="Real-time MIDI input from controllers and keyboards",
        version="1.0.0",
        author="AudioPlayground",
    )

    # Signal for thread-safe UI updates
    midi_message_received = pyqtSignal(object)  # MIDIMessage
    device_status_changed = pyqtSignal(str)  # Status message

    def __init__(self):
        """Initialize MIDI input module."""
        super().__init__(
            width=240,
            height=220,
            color=QColor(200, 100, 150),
        )

        # Add output ports
        self.freq_port = self.add_output("Freq")
        self.gate_port = self.add_output("Gate")
        self.vel_port = self.add_output("Vel")

        # MIDI components
        self.midi_worker: Optional[MIDIWorkerThread] = None
        self.cv_converter = MIDIToCV()

        # Create specialized output components for each port
        self.freq_output = CVFrequencyOutput(self.cv_converter)
        self.gate_output = CVGateOutput(self.cv_converter)
        self.vel_output = CVVelocityOutput(self.cv_converter)
        self._is_running = False

        # UI setup
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Device selection
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItem("(No Device)")
        device_layout.addWidget(self.device_combo)
        layout.addLayout(device_layout)

        # Refresh button
        btn_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_devices)
        btn_layout.addWidget(self.refresh_btn)

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self._toggle_midi)
        btn_layout.addWidget(self.start_btn)
        layout.addLayout(btn_layout)

        # Status display
        self.status_label = QLabel("Idle")
        self.status_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.status_label)

        # Current note display
        self.note_label = QLabel("--")
        self.note_label.setStyleSheet(
            "color: white; font-size: 14px; font-weight: bold;"
        )
        layout.addWidget(self.note_label)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Connect signals
        self.midi_message_received.connect(self._on_midi_message)
        self.device_status_changed.connect(self._on_status_changed)

        # Register parameters
        self.register_parameter(
            "device", self.device_combo, getter="currentText", setter="setCurrentText"
        )

        # Initial device refresh
        self._refresh_devices()

        logger.info("MIDI Input module initialized")

    def _refresh_devices(self):
        """Refresh the list of available MIDI devices."""
        try:
            # Import MIDIInput only for device listing
            from src.engine.io.midi.input import MIDO_AVAILABLE
            from src.engine.io.midi import MIDIInput

            if not MIDO_AVAILABLE:
                self.device_status_changed.emit("MIDI library not installed")
                logger.warning("mido library not available")
                return

            # Get devices (quick operation)
            devices = MIDIInput.list_devices()
            self.device_combo.clear()
            self.device_combo.addItem("(No Device)")
            self.device_combo.addItems(devices)

            if devices:
                self.device_status_changed.emit(f"Found {len(devices)} device(s)")
            else:
                self.device_status_changed.emit("No MIDI devices found")

            logger.info(f"Refreshed MIDI devices: {devices}")
        except ImportError:
            self.device_status_changed.emit("MIDI library not installed")
            logger.warning("Could not import MIDI modules")
        except Exception as e:
            self.device_status_changed.emit(f"Error: {str(e)}")
            logger.error(f"Failed to list MIDI devices: {e}", exc_info=True)

    def _toggle_midi(self):
        """Start or stop MIDI input."""
        if self._is_running:
            self._stop_midi()
        else:
            self._start_midi()

    def _start_midi(self):
        """Start receiving MIDI messages using worker thread."""
        logger.info("_start_midi called")
        device = self.device_combo.currentText()
        logger.info(f"Selected device: {device}")

        if device == "(No Device)":
            self.device_status_changed.emit("Please select a device")
            logger.warning("No device selected")
            return

        try:
            # Create worker thread (does NOT block UI)
            logger.info("Creating MIDI worker thread...")
            self.midi_worker = MIDIWorkerThread(device)

            # Connect worker signals to our slots (thread-safe communication)
            self.midi_worker.message_received.connect(self._on_midi_message)
            self.midi_worker.status_changed.connect(self._on_worker_status_changed)
            self.midi_worker.error_occurred.connect(self._on_worker_error)

            # Start the worker thread (non-blocking!)
            logger.info("Starting worker thread...")
            self.midi_worker.start()

            self._is_running = True
            self.start_btn.setText("Stop")
            self.device_status_changed.emit("Connecting...")

            logger.info("Worker thread started successfully")
        except Exception as e:
            self.device_status_changed.emit(f"Error: {str(e)}")
            logger.error(f"Failed to start MIDI worker: {e}", exc_info=True)

    def _stop_midi(self):
        """Stop receiving MIDI messages."""
        if self.midi_worker:
            try:
                logger.info("Stopping MIDI worker thread...")
                self.midi_worker.stop()
                # Wait for thread to finish (with timeout)
                self.midi_worker.wait(2000)  # 2 second timeout
                logger.info("MIDI worker thread stopped")
            except Exception as e:
                logger.error(f"Error stopping MIDI worker: {e}")
            finally:
                self.midi_worker = None

        self._is_running = False
        self.start_btn.setText("Start")
        self.note_label.setText("--")

        # Reset CV converter
        self.cv_converter.reset()

        logger.info("Stopped MIDI input")

    def _on_worker_status_changed(self, status: str):
        """Handle status change from worker thread (thread-safe).

        Args:
            status: Status message from worker
        """
        self.device_status_changed.emit(status)

    def _on_worker_error(self, error: str):
        """Handle error from worker thread (thread-safe).

        Args:
            error: Error message from worker
        """
        logger.error(f"Worker error: {error}")
        self.device_status_changed.emit(f"Error: {error}")
        # Auto-stop on error
        self._stop_midi()

    def _on_midi_message(self, msg: MIDIMessage):
        """Handle received MIDI message (runs in UI thread).

        Args:
            msg: Received MIDI message
        """
        # Update CV converter (no lock needed - we're in UI thread)
        self.cv_converter.process_message(msg)

        # Update UI
        if isinstance(msg, NoteOnMessage):
            if msg.velocity > 0:
                from src.engine.io.midi import midi_to_note_name

                note_name = midi_to_note_name(msg.note)
                self.note_label.setText(f"{note_name} ({msg.note})")
                self.note_label.setStyleSheet(
                    "color: #00ff00; font-size: 14px; font-weight: bold;"
                )
        elif isinstance(msg, NoteOffMessage):
            # Check if gate is off (no lock needed)
            if self.cv_converter.gate == 0.0:  # Only if no notes active
                self.note_label.setText("--")
                self.note_label.setStyleSheet(
                    "color: white; font-size: 14px; font-weight: bold;"
                )

    def _on_status_changed(self, status: str):
        """Update status label.

        Args:
            status: Status message to display
        """
        self.status_label.setText(status)

    def process(self, num_samples: int = 1):
        """Process MIDI input and write CV outputs to ports.

        The MIDI module generates control voltages based on the current
        MIDI state (active notes, velocity, etc.).

        Args:
            num_samples: Number of samples to generate
        """
        # Generate frequency CV
        if self.freq_port.is_connected:
            freq_samples = self.freq_output.get_samples(num_samples)
            self.freq_port.write(freq_samples)

        # Generate gate CV
        if self.gate_port.is_connected:
            gate_samples = self.gate_output.get_samples(num_samples)
            self.gate_port.write(gate_samples)

        # Generate velocity CV
        if self.vel_port.is_connected:
            vel_samples = self.vel_output.get_samples(num_samples)
            self.vel_port.write(vel_samples)

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Create the MIDI to CV component.

        Returns:
            MIDIToCV component that generates control voltages
        """
        return self.cv_converter

    def get_output_component(self, port_name: str) -> Any:
        """Get the component for a specific output port.

        The MIDI module has multiple outputs (Freq, Gate, Vel) each with
        a specialized adapter component.

        Args:
            port_name: Name of the output port

        Returns:
            The specialized CV output component for that port
        """
        if port_name == "Freq":
            return self.freq_output
        elif port_name == "Gate":
            return self.gate_output
        elif port_name == "Vel":
            return self.vel_output
        else:
            # Default to frequency output
            return self.freq_output

    def __del__(self):
        """Destructor - ensure MIDI input is stopped."""
        try:
            self._stop_midi()
        except Exception:
            pass
