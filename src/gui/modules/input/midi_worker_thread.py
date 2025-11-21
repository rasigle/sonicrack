"""Worker thread for MIDI input to prevent UI freezing.

This thread handles MIDI I/O in the background and communicates with
the UI thread via Qt signals (thread-safe).
"""

import logging
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

from src.engine.io.midi import MIDIInput

logger = logging.getLogger(__name__)


class MIDIWorkerThread(QThread):
    """Background thread for MIDI input operations.

    This prevents blocking the UI thread when opening devices,
    starting input, or polling for messages.

    Signals:
        message_received: Emitted when a MIDI message is received
        status_changed: Emitted when connection status changes
        error_occurred: Emitted when an error occurs
    """

    # Thread-safe signals
    message_received = pyqtSignal(object)  # MIDIMessage
    status_changed = pyqtSignal(str)  # Status message
    error_occurred = pyqtSignal(str)  # Error message

    def __init__(self, device_name: str):
        """Initialize MIDI worker thread.

        Args:
            device_name: Name of MIDI device to open
        """
        super().__init__()
        self.device_name = device_name
        self.midi_input: Optional[MIDIInput] = None
        self._running = False

    def run(self):
        """Main thread loop - runs in background thread."""
        try:
            # Open MIDI device (may block - but we're in background thread)
            logger.info(f"Worker thread: Opening MIDI device: {self.device_name}")
            self.midi_input = MIDIInput(self.device_name)
            self.midi_input.start()

            self._running = True
            self.status_changed.emit(f"Connected: {self.device_name}")
            logger.info("Worker thread: MIDI input started")

            # Main polling loop
            while self._running:
                try:
                    # Poll for messages (this is a blocking operation)
                    messages = self.midi_input.get_messages(
                        timeout=0.01
                    )  # 10ms timeout

                    # Emit each message via signal (thread-safe)
                    for msg in messages:
                        self.message_received.emit(msg)

                    # Small sleep to prevent busy-waiting
                    self.msleep(1)  # 1ms sleep

                except Exception as e:
                    if self._running:
                        logger.error(f"Worker thread: Error polling messages: {e}")
                        self.error_occurred.emit(f"Polling error: {e}")

        except Exception as e:
            logger.error(f"Worker thread: Failed to start MIDI: {e}", exc_info=True)
            self.error_occurred.emit(f"Failed to open device: {e}")

        finally:
            # Cleanup
            logger.info("Worker thread: Cleaning up...")
            if self.midi_input:
                try:
                    self.midi_input.stop()
                    self.midi_input.close()
                except Exception as e:
                    logger.error(f"Worker thread: Error during cleanup: {e}")

            self.status_changed.emit("Disconnected")
            logger.info("Worker thread: Exited")

    def stop(self):
        """Stop the MIDI worker thread."""
        logger.info("Worker thread: Stop requested")
        self._running = False
        # Thread will exit naturally from the run() loop
