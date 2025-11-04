"""Tests for MIDI input module."""

import pytest
from unittest.mock import Mock, patch
from src.engine.midi.input import MIDIInput, MIDO_AVAILABLE
from src.engine.midi.messages import NoteOnMessage, NoteOffMessage


# Skip all tests if mido not available
pytestmark = pytest.mark.skipif(not MIDO_AVAILABLE, reason="mido not installed")


class TestMIDIInput:
    """Test MIDIInput class."""

    def test_list_devices(self):
        """Test listing MIDI devices."""
        devices = MIDIInput.list_devices()
        assert isinstance(devices, list)
        # Can't test specific devices as it depends on system

    @patch("src.engine.midi.input.mido")
    def test_init(self, mock_mido):
        """Test initialization."""
        mock_port = Mock()
        mock_mido.open_input.return_value = mock_port

        midi = MIDIInput("TestDevice")
        assert midi.device_name == "TestDevice"
        assert midi._port is None  # Port not opened until open() called
        assert midi._running is False

    @patch("src.engine.midi.input.mido")
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

    @patch("src.engine.midi.input.mido")
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

    @patch("src.engine.midi.input.mido")
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

    @patch("src.engine.midi.input.mido")
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

    @patch("src.engine.midi.input.mido")
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
