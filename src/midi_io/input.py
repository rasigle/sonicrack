"""MIDI input handler for real-time MIDI from controllers/keyboards.

This module provides real-time MIDI input functionality using the mido library.
It handles device enumeration, message reception, and callback-based processing.

Example:
    >>> from src.midi_io import MIDIInput
    >>>
    >>> def on_message(msg):
    ...     print(f"Received: {msg}")
    >>>
    >>> midi = MIDIInput()
    >>> print(midi.list_devices())
    ['Steinberg UR22-1 0']
    >>> midi.start(on_message)
    >>> # ... receive messages ...
    >>>
    >>> midi.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from queue import Empty, Queue

try:
    import mido

    MIDO_AVAILABLE = True
except ImportError:
    mido = None
    MIDO_AVAILABLE = False
    logging.warning("mido not installed. Install with: pip install mido python-rtmidi")

from src.midi_io.messages import (
    AftertouchMessage,
    ControlChangeMessage,
    MIDIMessage,
    NoteOffMessage,
    NoteOnMessage,
    PitchBendMessage,
    ProgramChangeMessage,
)

logger = logging.getLogger(__name__)


class MIDIInput:
    """Handle real-time MIDI input from controllers and keyboards.

    This class manages MIDI input ports, converts raw MIDI messages to our
    internal message format, and provides both callback and polling interfaces.

    Attributes:
        device_name: Name of the currently opened MIDI device

    Example:
        >>> # List available devices
        >>> devices = MIDIInput.list_devices()
        >>> print(devices)
        ['Steinberg UR22-1 0']
        >>> # Open first device and print messages
        >>> midi = MIDIInput(devices[0])
        >>> midi.start(lambda msg: print(msg))
        >>> # ... messages printed as they arrive ...
        >>> midi.stop()
    """

    def __init__(self, device_name: str | None = None):
        """Initialize MIDI input.

        Args:
            device_name: Name of MIDI device to open. If None, opens first available.

        Raises:
            RuntimeError: If mido is not installed
            IOError: If device cannot be opened
        """

        self.device_name = device_name
        self._port: mido.ports.BaseInput | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._state_lock = threading.RLock()
        self._running = False
        self._callback: Callable[[MIDIMessage], None] | None = None
        self._message_queue: Queue[MIDIMessage] = Queue()
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
        return mido.get_input_names()

    def open(self, device_name: str | None = None) -> None:
        """Open a MIDI input device.

        Args:
            device_name: Name of device to open. If None, uses constructor value or
                first available.

        Raises:
            OSError: If device cannot be opened
            RuntimeError: If called while MIDI input is running
        """
        with self._state_lock:
            if self._running:
                raise RuntimeError("Cannot open MIDI device while input is running")

            if self._port is not None:
                self._close_port()

            device = device_name or self.device_name

            if device is None:
                # Open first available device
                devices = self.list_devices()
                if not devices:
                    raise OSError("No MIDI input devices found")
                device = devices[0]

            try:
                self._port = mido.open_input(device)
                self.device_name = device
                logger.info(f"Opened MIDI device: {device}")
            except (OSError, RuntimeError) as e:
                raise OSError(f"Failed to open MIDI device '{device}': {e}") from e

    def close(self) -> None:
        """Close the MIDI input device."""
        self.stop()

        with self._state_lock:
            self._close_port()

    def start(self, callback: Callable[[MIDIMessage], None] | None = None) -> None:
        """Start receiving MIDI messages.

        Args:
            callback: Function to call for each received message.
                The callback runs on the MIDI input background thread, so GUI code
                should use get_messages() or a UI thread-safe dispatcher instead.
                Signature: callback(msg: MIDIMessage) -> None

        Example:
            >>> def print_notes(msg):
            ...     if isinstance(msg, NoteOnMessage):
            ...         print(f"Note {msg.note} on, velocity {msg.velocity}")
            >>>
            >>> midi = MIDIInput()
            >>> midi.start(print_notes)
        """
        with self._state_lock:
            if self._running:
                logger.warning("MIDI input already running")
                return

            if self._port is None:
                self.open()

            self._callback = callback
            self._running = True
            self._stop_event.clear()
            self._current_time = 0.0

            self._thread = threading.Thread(
                target=self._receive_loop,
                name=f"MIDIInput[{self.device_name or 'auto'}]",
                daemon=True,
            )
            self._thread.start()

        logger.info("Started MIDI input")

    def stop(self) -> None:
        """Stop receiving MIDI messages."""
        with self._state_lock:
            thread = self._thread
            was_running = self._running
            if thread is None and not was_running:
                return

            self._running = False
            self._stop_event.set()

        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
            if thread.is_alive():
                logger.warning("MIDI input thread did not stop within 1 second")

        with self._state_lock:
            if self._thread is thread:
                self._thread = None

        logger.info("Stopped MIDI input")

    def get_messages(self, timeout: float = 0.0) -> list[MIDIMessage]:
        """Get queued messages (polling interface).

        Alternative to callback - retrieve messages that have been queued. With
        timeout=0, this drains currently queued messages without blocking.

        Args:
            timeout: Maximum time to wait for the first message, in seconds.
                Negative values are treated as zero.

        Returns:
            List of received messages (may be empty)

        Example:
            >>> midi = MIDIInput()
            >>> midi.start()  # No callback
            >>>
            >>> while True:
            ...     msgs = midi.get_messages(timeout=0.1)
            ...     for msg in msgs:
            ...         print(msg)
        """
        messages: list[MIDIMessage] = []

        if timeout <= 0:
            while True:
                try:
                    messages.append(self._message_queue.get_nowait())
                except Empty:
                    return messages

        deadline = time.monotonic() + timeout

        while True:
            try:
                remaining = max(0, deadline - time.monotonic())
                if not messages and remaining <= 0:
                    break
                msg = self._message_queue.get(timeout=remaining)
                messages.append(msg)
            except Empty:
                break

        return messages

    def _receive_loop(self):
        """Internal thread loop for receiving MIDI messages."""
        while not self._stop_event.is_set():
            try:
                # Use iter_pending() to get available messages without blocking
                with self._state_lock:
                    port = self._port

                if port is None:
                    break

                for raw_msg in port.iter_pending():
                    self._handle_raw_message(raw_msg)

                # Small sleep to avoid busy-waiting
                time.sleep(0.001)  # 1ms

            except (OSError, RuntimeError) as e:
                if self._running:
                    logger.error(f"Error receiving MIDI message: {e}")
                time.sleep(0.01)  # Back off on error

        with self._state_lock:
            self._running = False

    def _handle_raw_message(self, raw_msg) -> None:
        """Convert, queue, and optionally dispatch one raw MIDI message."""
        msg = self._convert_message(raw_msg)
        if msg is None:
            return

        self._message_queue.put(msg)

        callback = self._callback
        if callback is None:
            return

        try:
            callback(msg)
        except Exception:
            # User callbacks are an isolation boundary for the MIDI input thread.
            logger.exception("MIDI input callback failed")

    def _convert_message(self, raw_msg) -> MIDIMessage | None:
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
        with self._state_lock:
            return self._running

    def __enter__(self) -> MIDIInput:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Cleanup on deletion."""
        with suppress(Exception):
            self.close()

    def _close_port(self) -> None:
        """Close the current port. Caller must hold _state_lock."""
        if self._port is None:
            return

        port = self._port
        self._port = None
        port.close()
        logger.info("Closed MIDI device")
