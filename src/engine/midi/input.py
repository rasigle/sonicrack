"""MIDI input handler for real-time MIDI from controllers/keyboards.

This module provides real-time MIDI input functionality using the mido library.
It handles device enumeration, message reception, and callback-based processing.

Example:
    >>> from src.engine.midi.input import MIDIInput
    >>>
    >>> def on_message(msg):
    ...     print(f"Received: {msg}")
    >>>
    >>> midi = MIDIInput()
    >>> print(midi.list_devices())
    >>> midi.start(on_message)
    >>> # ... receive messages ...
    >>> midi.stop()
"""

import logging
from typing import Callable, Optional
from queue import Queue, Empty
import threading

try:
    import mido

    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    logging.warning("mido not installed. Install with: pip install mido python-rtmidi")

from src.engine.midi.messages import (
    MIDIMessage,
    NoteOnMessage,
    NoteOffMessage,
    ControlChangeMessage,
    PitchBendMessage,
    ProgramChangeMessage,
    AftertouchMessage,
)

logger = logging.getLogger(__name__)


class MIDIInput:
    """Handle real-time MIDI input from controllers and keyboards.

    This class manages MIDI input ports, converts raw MIDI messages to our
    internal message format, and provides both callback and polling interfaces.

    Attributes:
        device_name: Name of the currently opened MIDI device
        is_running: Whether the input is currently receiving messages

    Example:
        >>> # List available devices
        >>> devices = MIDIInput.list_devices()
        >>> print(devices)

        >>> # Open first device and print messages
        >>> midi = MIDIInput(devices[0])
        >>> midi.start(lambda msg: print(msg))
        >>> # ... messages printed as they arrive ...
        >>> midi.stop()
    """

    def __init__(self, device_name: Optional[str] = None):
        """Initialize MIDI input.

        Args:
            device_name: Name of MIDI device to open. If None, opens first available.

        Raises:
            RuntimeError: If mido is not installed
            IOError: If device cannot be opened
        """
        if not MIDO_AVAILABLE:
            raise RuntimeError(
                "mido library not installed. "
                "Install with: pip install mido python-rtmidi"
            )

        self.device_name = device_name
        self._port: Optional[mido.ports.BaseInput] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._callback: Optional[Callable[[MIDIMessage], None]] = None
        self._message_queue: Queue = Queue()
        self._current_time = 0.0

        logger.info(f"MIDI Input initialized for device: {device_name or 'auto'}")

    @staticmethod
    def list_devices() -> list[str]:
        """List all available MIDI input devices.

        Returns:
            List of device names

        Raises:
            RuntimeError: If mido is not installed

        Example:
            >>> devices = MIDIInput.list_devices()
            >>> for i, device in enumerate(devices):
            ...     print(f"{i}: {device}")
        """
        if not MIDO_AVAILABLE:
            raise RuntimeError("mido library not installed")

        return mido.get_input_names()

    def open(self, device_name: Optional[str] = None):
        """Open a MIDI input device.

        Args:
            device_name: Name of device to open. If None, uses constructor value or first available.

        Raises:
            IOError: If device cannot be opened
        """
        if self._port is not None:
            self.close()

        device = device_name or self.device_name

        if device is None:
            # Open first available device
            devices = self.list_devices()
            if not devices:
                raise IOError("No MIDI input devices found")
            device = devices[0]

        try:
            self._port = mido.open_input(device)
            self.device_name = device
            logger.info(f"Opened MIDI device: {device}")
        except Exception as e:
            raise IOError(f"Failed to open MIDI device '{device}': {e}")

    def close(self):
        """Close the MIDI input device."""
        if self._port is not None:
            self._port.close()
            self._port = None
            logger.info("Closed MIDI device")

    def start(self, callback: Optional[Callable[[MIDIMessage], None]] = None):
        """Start receiving MIDI messages.

        Args:
            callback: Function to call for each received message.
                     Signature: callback(msg: MIDIMessage) -> None

        Example:
            >>> def print_notes(msg):
            ...     if isinstance(msg, NoteOnMessage):
            ...         print(f"Note {msg.note} on, velocity {msg.velocity}")
            >>>
            >>> midi = MIDIInput()
            >>> midi.start(print_notes)
        """
        if self._running:
            logger.warning("MIDI input already running")
            return

        if self._port is None:
            self.open()

        self._callback = callback
        self._running = True
        self._current_time = 0.0

        # Start receiving thread
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()

        logger.info("Started MIDI input")

    def stop(self):
        """Stop receiving MIDI messages."""
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        logger.info("Stopped MIDI input")

    def get_messages(self, timeout: float = 0.0) -> list[MIDIMessage]:
        """Get queued messages (polling interface).

        Alternative to callback - retrieve messages that have been queued.

        Args:
            timeout: Maximum time to wait for messages (seconds)

        Returns:
            List of received messages (may be empty)

        Example:
            >>> midi = MIDIInput()
            >>> midi.start()  # No callback
            >>>
            >>> while True:
            ...     messages = midi.get_messages(timeout=0.1)
            ...     for msg in messages:
            ...         print(msg)
        """
        messages = []
        deadline = None

        if timeout > 0:
            import time

            deadline = time.time() + timeout

        while True:
            try:
                remaining = None
                if deadline is not None:
                    import time

                    remaining = max(0, deadline - time.time())
                    if remaining <= 0:
                        break

                msg = self._message_queue.get(timeout=remaining)
                messages.append(msg)
            except Empty:
                break

        return messages

    def _receive_loop(self):
        """Internal thread loop for receiving MIDI messages."""
        import time

        while self._running:
            try:
                # Use iter_pending() to get available messages without blocking
                for raw_msg in self._port.iter_pending():
                    # Convert to our message format
                    msg = self._convert_message(raw_msg)

                    if msg is not None:
                        # Queue message
                        self._message_queue.put(msg)

                        # Call callback if provided
                        if self._callback is not None:
                            try:
                                self._callback(msg)
                            except Exception as e:
                                logger.error(f"Error in MIDI callback: {e}")

                # Small sleep to avoid busy-waiting
                time.sleep(0.001)  # 1ms

            except Exception as e:
                if self._running:
                    logger.error(f"Error receiving MIDI message: {e}")
                time.sleep(0.01)  # Back off on error

    def _convert_message(self, raw_msg) -> Optional[MIDIMessage]:
        """Convert mido message to our internal format.

        Args:
            raw_msg: mido.Message object

        Returns:
            Converted MIDIMessage or None if message type not supported
        """
        # Update timestamp (relative time)
        self._current_time += raw_msg.time

        timestamp = self._current_time
        channel = getattr(raw_msg, "channel", 0)

        # Convert based on message type
        if raw_msg.type == "note_on":
            # Note: mido uses velocity=0 for note_on as note_off
            if raw_msg.velocity == 0:
                return NoteOffMessage(
                    timestamp=timestamp, channel=channel, note=raw_msg.note, velocity=0
                )
            else:
                return NoteOnMessage(
                    timestamp=timestamp,
                    channel=channel,
                    note=raw_msg.note,
                    velocity=raw_msg.velocity,
                )

        elif raw_msg.type == "note_off":
            return NoteOffMessage(
                timestamp=timestamp,
                channel=channel,
                note=raw_msg.note,
                velocity=raw_msg.velocity,
            )

        elif raw_msg.type == "control_change":
            return ControlChangeMessage(
                timestamp=timestamp,
                channel=channel,
                controller=raw_msg.control,
                value=raw_msg.value,
            )

        elif raw_msg.type == "pitchwheel":
            return PitchBendMessage(
                timestamp=timestamp, channel=channel, value=raw_msg.pitch
            )

        elif raw_msg.type == "program_change":
            return ProgramChangeMessage(
                timestamp=timestamp, channel=channel, program=raw_msg.program
            )

        elif raw_msg.type == "aftertouch":
            return AftertouchMessage(
                timestamp=timestamp, channel=channel, pressure=raw_msg.value
            )

        else:
            # Unsupported message type
            logger.debug(f"Unsupported MIDI message type: {raw_msg.type}")
            return None

    @property
    def is_running(self) -> bool:
        """Check if MIDI input is currently running."""
        return self._running

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        self.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
        self.close()
