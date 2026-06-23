"""Tests for MIDI input module."""

import time
from unittest.mock import Mock, patch

import pytest

import src.engine.io.midi.input as midi_input_module
from src.engine.io.midi import MIDIInput, NoteOffMessage, NoteOnMessage
from src.engine.io.midi.input import MIDO_AVAILABLE

# Skip all tests if mido not available
pytestmark = pytest.mark.skipif(not MIDO_AVAILABLE, reason="mido not installed")


class TestMIDIInput:
    """Test MIDIInput class."""

    def test_list_devices(self):
        """Test listing MIDI devices."""
        devices = MIDIInput.list_devices()
        assert isinstance(devices, list)
        # Can't test specific devices as it depends on system

    @patch.object(midi_input_module, "mido")
    def test_init(self, mock_mido):
        """Test initialization."""
        mock_port = Mock()
        mock_mido.open_input.return_value = mock_port

        midi = MIDIInput("TestDevice")
        assert midi.device_name == "TestDevice"
        assert midi._port is None  # Port not opened until open() called
        assert midi._running is False

    @patch.object(midi_input_module, "mido")
    def test_context_manager(self, mock_mido):
        """Test context manager protocol."""
        mock_port = Mock()
        mock_mido.open_input.return_value = mock_port

        with MIDIInput("TestDevice") as midi:
            # Open the port within context
            midi.open()
            assert midi._port == mock_port

        # Verify close was called
        mock_port.close.assert_called_once()

    @patch.object(midi_input_module, "mido")
    def test_start_stop(self, mock_mido):
        """Test starting and stopping."""
        mock_port = Mock()
        mock_port.iter_pending.return_value = []
        mock_mido.open_input.return_value = mock_port

        callback = Mock()
        midi = MIDIInput("TestDevice")

        # Start
        midi.start(callback)
        assert midi._running is True
        assert midi._callback == callback

        # Stop
        midi.stop()
        assert midi._running is False

    @patch.object(midi_input_module, "mido")
    def test_close_stops_input_before_closing_port(self, mock_mido):
        """Test that close() shuts down receiving before closing the port."""
        mock_port = Mock()
        mock_port.iter_pending.return_value = []
        mock_mido.open_input.return_value = mock_port

        midi = MIDIInput("TestDevice")
        midi.start()

        midi.close()

        assert midi.is_running is False
        mock_port.close.assert_called_once()
        assert midi._port is None

    @patch.object(midi_input_module, "mido")
    def test_open_while_running_raises(self, mock_mido):
        """Test that ports cannot be reopened while receive thread is active."""
        mock_port = Mock()
        mock_port.iter_pending.return_value = []
        mock_mido.open_input.return_value = mock_port

        midi = MIDIInput("TestDevice")
        midi.start()

        with pytest.raises(RuntimeError, match="while input is running"):
            midi.open("OtherDevice")

        midi.close()

    @patch.object(midi_input_module, "mido")
    def test_get_messages_default_is_non_blocking(self, mock_mido):
        """Test that default polling drains queued messages without blocking."""
        mock_mido.open_input.return_value = Mock()

        midi = MIDIInput("TestDevice")

        start = time.monotonic()
        messages = midi.get_messages()
        elapsed = time.monotonic() - start

        assert messages == []
        assert elapsed < 0.1

    @patch.object(midi_input_module, "mido")
    def test_get_messages_drains_queued_messages(self, mock_mido):
        """Test non-blocking polling returns all currently queued messages."""
        mock_mido.open_input.return_value = Mock()
        first = NoteOnMessage(timestamp=0.0, channel=0, note=60, velocity=100)
        second = NoteOffMessage(timestamp=0.1, channel=0, note=60, velocity=64)

        midi = MIDIInput("TestDevice")
        midi._message_queue.put(first)
        midi._message_queue.put(second)

        assert midi.get_messages() == [first, second]
        assert midi.get_messages() == []

    @patch.object(midi_input_module, "mido")
    def test_handle_raw_message_queues_and_dispatches_callback(self, mock_mido):
        """Test callback and polling receive the same converted message."""
        mock_mido.open_input.return_value = Mock()
        callback = Mock()
        midi = MIDIInput("TestDevice")
        midi._callback = callback

        raw_msg = Mock()
        raw_msg.type = "note_on"
        raw_msg.note = 60
        raw_msg.velocity = 100
        raw_msg.channel = 0
        raw_msg.time = 0.25

        midi._handle_raw_message(raw_msg)

        messages = midi.get_messages()
        assert len(messages) == 1
        callback.assert_called_once_with(messages[0])

    @patch.object(midi_input_module, "mido")
    def test_callback_exception_does_not_drop_queued_message(self, mock_mido, caplog):
        """Test callback failures are isolated from polling delivery."""
        mock_mido.open_input.return_value = Mock()
        callback = Mock(side_effect=RuntimeError("boom"))
        midi = MIDIInput("TestDevice")
        midi._callback = callback

        raw_msg = Mock()
        raw_msg.type = "note_on"
        raw_msg.note = 60
        raw_msg.velocity = 100
        raw_msg.channel = 0
        raw_msg.time = 0.25

        midi._handle_raw_message(raw_msg)

        assert len(midi.get_messages()) == 1
        assert "MIDI input callback failed" in caplog.text

    @patch.object(midi_input_module, "mido")
    def test_message_conversion_note_on(self, mock_mido):
        """Test converting note on messages."""
        mock_port = Mock()
        mock_mido.open_input.return_value = mock_port

        midi = MIDIInput("TestDevice")

        # Mock mido message
        mock_msg = Mock()
        mock_msg.type = "note_on"
        mock_msg.note = 60
        mock_msg.velocity = 100
        mock_msg.channel = 0
        mock_msg.is_meta = False
        mock_msg.time = 1.5  # Delta time in seconds

        converted = midi._convert_message(mock_msg)

        assert isinstance(converted, NoteOnMessage)
        assert converted.note == 60
        assert converted.velocity == 100
        assert converted.channel == 0
        # Timestamp accumulated from delta times
        assert converted.timestamp == 1.5

    @patch.object(midi_input_module, "mido")
    def test_message_conversion_note_off(self, mock_mido):
        """Test converting note off messages."""
        mock_port = Mock()
        mock_mido.open_input.return_value = mock_port

        midi = MIDIInput("TestDevice")

        # Mock mido message
        mock_msg = Mock()
        mock_msg.type = "note_off"
        mock_msg.note = 60
        mock_msg.velocity = 64
        mock_msg.channel = 0
        mock_msg.is_meta = False
        mock_msg.time = 2.0

        converted = midi._convert_message(mock_msg)

        assert isinstance(converted, NoteOffMessage)
        assert converted.note == 60
        assert converted.timestamp == 2.0

    @patch.object(midi_input_module, "mido")
    def test_note_on_velocity_zero_is_note_off(self, mock_mido):
        """Test that note on with velocity 0 converts to note off."""
        mock_port = Mock()
        mock_mido.open_input.return_value = mock_port

        midi = MIDIInput("TestDevice")

        # Mock mido message - note on with velocity 0
        mock_msg = Mock()
        mock_msg.type = "note_on"
        mock_msg.note = 60
        mock_msg.velocity = 0
        mock_msg.channel = 0
        mock_msg.is_meta = False
        mock_msg.time = 1.0

        converted = midi._convert_message(mock_msg)

        assert isinstance(converted, NoteOffMessage)
        assert converted.note == 60
