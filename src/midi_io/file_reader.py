"""MIDI file reader for playback and analysis.

This module provides MIDI file reading and playback functionality using the mido
library. It can parse MIDI files, extract notes and events, and provide timeline-based
playback.

Example:
    >>> from src.midi_io import MIDIFile
    >>>
    >>> # Load a MIDI file
    >>> midi_file = MIDIFile("song.mid")
    >>>
    >>> # Get basic info
    >>> print(f"Duration: {midi_file.get_duration()} seconds")
    >>> print(f"Tempo: {midi_file.get_tempo()} BPM")
    >>>
    >>> # Get notes in a time range
    >>> notes = midi_file.get_notes_in_range(0, 2.0)  # First 2 seconds
    >>> for note in notes:
    ...     print(f"{note.timestamp}s: Note {note.note}")
"""

import logging
from pathlib import Path

try:
    import mido

    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    logging.warning("mido not installed. Install with: pip install mido")

from src.midi_io.messages import (
    ControlChangeMessage,
    MIDIMessage,
    NoteOffMessage,
    NoteOnMessage,
    PitchBendMessage,
    ProgramChangeMessage,
)

logger = logging.getLogger(__name__)


class MIDIFile:
    """Read and parse MIDI files for playback.

    This class loads MIDI files, extracts all messages with timestamps,
    and provides methods for timeline-based access and playback.

    Attributes:
        filepath: Path to the MIDI file
        messages: List of all MIDI messages with timestamps
        duration: Total duration in seconds
        tempo: Current tempo in BPM
        ticks_per_beat: MIDI file ticks per beat (from header)

    Example:
        >>> midi = MIDIFile("example.mid")
        >>> print(f"Loaded {len(midi.messages)} messages")
        >>> print(f"Duration: {midi.duration:.2f} seconds")
        >>>
        >>> # Get first 5 seconds of notes
        >>> notes = midi.get_notes_in_range(0, 5.0)
        >>> for note in notes:
        ...     if isinstance(note, NoteOnMessage):
        ...         print(f"Note {note.note} at {note.timestamp:.2f}s")
    """

    def __init__(self, filepath: str | Path):
        """Initialize MIDI file reader.

        Args:
            filepath: Path to MIDI file (.mid or .midi)

        Raises:
            RuntimeError: If mido is not installed
            FileNotFoundError: If file doesn't exist
            IOError: If file cannot be read or is not a valid MIDI file
        """
        if not MIDO_AVAILABLE:
            raise RuntimeError(
                "mido library not installed. Install with: pip install mido"
            )

        self.filepath = Path(filepath)

        if not self.filepath.exists():
            raise FileNotFoundError(f"MIDI file not found: {filepath}")

        # Parse the MIDI file
        try:
            self._midi = mido.MidiFile(str(self.filepath))
        except (OSError, EOFError, ValueError) as e:
            raise OSError(f"Failed to read MIDI file '{filepath}': {e}") from e

        # Extract properties
        self.ticks_per_beat = self._midi.ticks_per_beat
        self._tempo = 500000  # Default: 120 BPM (500000 microseconds per beat)

        # Parse all messages with timestamps
        self.messages: list[MIDIMessage] = []
        self._parse_messages()

        # Calculate duration
        self.duration = self._calculate_duration()

        logger.info(
            f"Loaded MIDI file: {self.filepath.name} "
            f"({len(self.messages)} messages, {self.duration:.2f}s)"
        )

    def _parse_messages(self):
        """Parse all MIDI messages from the file with absolute timestamps."""
        current_time = 0.0
        tempo = self._tempo  # microseconds per beat

        # Merge all tracks and sort by time
        for msg in self._midi:
            # Convert delta time (ticks) to seconds
            if msg.time > 0:
                # time_in_seconds = (ticks * tempo) / (ticks_per_beat * 1_000_000)
                tick_duration = tempo / (self.ticks_per_beat * 1_000_000)
                current_time += msg.time * tick_duration

            # Update tempo if meta message
            if msg.type == "set_tempo":
                tempo = msg.tempo
                self._tempo = tempo
                continue

            # Skip other meta messages
            if msg.is_meta:
                continue

            # Convert to our message format
            converted = self._convert_message(msg, current_time)
            if converted is not None:
                self.messages.append(converted)

    def _convert_message(self, msg, timestamp: float) -> MIDIMessage | None:
        """Convert mido message to our internal format.

        Args:
            msg: mido.Message object
            timestamp: Absolute timestamp in seconds

        Returns:
            Converted MIDIMessage or None if message type not supported
        """
        channel = getattr(msg, "channel", 0)

        if msg.type == "note_on":
            # Note: velocity=0 is note_off
            if msg.velocity == 0:
                return NoteOffMessage(
                    timestamp=timestamp, channel=channel, note=msg.note, velocity=0
                )

            return NoteOnMessage(
                timestamp=timestamp,
                channel=channel,
                note=msg.note,
                velocity=msg.velocity,
            )

        elif msg.type == "note_off":
            return NoteOffMessage(
                timestamp=timestamp,
                channel=channel,
                note=msg.note,
                velocity=getattr(msg, "velocity", 64),
            )

        elif msg.type == "control_change":
            return ControlChangeMessage(
                timestamp=timestamp,
                channel=channel,
                controller=msg.control,
                value=msg.value,
            )

        elif msg.type == "pitchwheel":
            return PitchBendMessage(
                timestamp=timestamp, channel=channel, value=msg.pitch
            )

        elif msg.type == "program_change":
            return ProgramChangeMessage(
                timestamp=timestamp, channel=channel, program=msg.program
            )

        # Unsupported message type
        return None

    def _calculate_duration(self) -> float:
        """Calculate total duration of the MIDI file.

        Returns:
            Duration in seconds
        """
        if not self.messages:
            return 0.0

        return max(msg.timestamp for msg in self.messages)

    def get_notes_in_range(
        self, start_time: float, end_time: float, channel: int | None = None
    ) -> list[MIDIMessage]:
        """Get all MIDI messages in a time range.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            channel: Optional channel filter (0-15), None for all channels

        Returns:
            List of MIDI messages in the time range

        Example:
            >>> # Get first 10 seconds
            >>> midi = MIDIFile("example.mid")
            >>> msgs = midi.get_notes_in_range(0, 10.0)
            >>>
            >>> # Get messages on channel 1 only
            >>> ch1_messages = midi.get_notes_in_range(0, 10.0, channel=0)
        """
        messages = [
            msg for msg in self.messages if start_time <= msg.timestamp <= end_time
        ]

        if channel is not None:
            messages = [msg for msg in messages if msg.channel == channel]

        return messages

    def get_tempo(self) -> float:
        """Get the current tempo in BPM.

        Returns:
            Tempo in beats per minute
        """
        # Convert microseconds per beat to BPM
        return 60_000_000 / self._tempo

    def get_duration(self) -> float:
        """Get the total duration of the MIDI file.

        Returns:
            Duration in seconds
        """
        return self.duration

    def get_track_count(self) -> int:
        """Get the number of tracks in the MIDI file.

        Returns:
            Number of tracks
        """
        return len(self._midi.tracks)

    def get_note_count(self) -> int:
        """Get the total number of note on/off messages.

        Returns:
            Number of note messages
        """
        return sum(
            1
            for msg in self.messages
            if isinstance(msg, (NoteOnMessage, NoteOffMessage))
        )

    def get_channel_messages(self, channel: int) -> list[MIDIMessage]:
        """Get all messages for a specific channel.

        Args:
            channel: MIDI channel (0-15)

        Returns:
            List of messages on that channel
        """
        return [msg for msg in self.messages if msg.channel == channel]

    def count_per_message_type(self) -> dict[str, int]:
        """Get count of each message type.

        Returns:
            Dictionary mapping message type name to count

        Example:
            >>> midi = MIDIFile("example.mid")>>>
            >>> counts = midi.count_per_message_type()
            >>> print(counts)
            {'NoteOnMessage': 450, 'NoteOffMessage': 450, 'ControlChangeMessage': 23}
        """
        type_counts: dict[str, int] = {}
        for msg in self.messages:
            type_name = type(msg).__name__
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        return type_counts

    def get_used_channels(self) -> list[int]:
        """Get list of channels that have messages.

        Returns:
            Sorted list of channel numbers that contain messages
        """
        channels = set(msg.channel for msg in self.messages)
        return sorted(channels)

    def get_note_range(self) -> tuple[int, int]:
        """Get the range of notes used in the file.

        Returns:
            Tuple of (lowest_note, highest_note) or (0, 0) if no notes
        """
        notes = [
            msg.note
            for msg in self.messages
            if isinstance(msg, (NoteOnMessage, NoteOffMessage))
        ]

        if not notes:
            return 0, 0

        return min(notes), max(notes)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MIDIFile('{self.filepath.name}', "
            f"{len(self.messages)} messages, "
            f"{self.duration:.2f}s, "
            f"{self.get_tempo():.1f} BPM)"
        )

    def __len__(self) -> int:
        """Return number of messages."""
        return len(self.messages)
