"""MIDI support for AudioPlayground.

This package provides MIDI functionality including:
- MIDI message handling (Note On/Off, CC, Pitch Bend)
- Real-time MIDI input from controllers
- MIDI file reading and playback
- Voice management for polyphony
- Note-to-frequency conversion utilities

Example - Basic MIDI Input:
    >>> from src.midi_io import MIDIInput
    >>>
    >>> def on_message(msg):
    ...     print(f"Received: {msg}")
    >>>
    >>> midi = MIDIInput()
    >>> midi.start(on_message)

Example - MIDI File Playback:
    >>> from src.midi_io import MIDIFile
    >>>
    >>> midi_file = MIDIFile("song.mid")
    >>> notes = midi_file.get_notes_in_range(0, 1.0)  # First second

Example - Note Conversion:
    >>> from src.midi_io import midi_to_frequency, note_name_to_midi
    >>>
    >>> freq = midi_to_frequency(69)  # A4 = 440 Hz
    >>> note = note_name_to_midi("C4")  # Middle C = 60
"""

from src.midi_io.cv_outputs import (
    CVFrequencyOutput,
    CVGateOutput,
    CVVelocityOutput,
)
from src.midi_io.file_reader import MIDIFile
from src.midi_io.input import MIDIInput
from src.midi_io.messages import (
    AftertouchMessage,
    ControlChangeMessage,
    MIDIMessage,
    NoteOffMessage,
    NoteOnMessage,
    PitchBendMessage,
    ProgramChangeMessage,
)
from src.midi_io.midi_to_cv import MIDIToCV
from src.midi_io.monophonic_synth import MonophonicSynth
from src.midi_io.polyphonic_synth import PolyphonicSynth, Voice
from src.midi_io.utils import (
    frequency_to_midi,
    get_note_range,
    midi_to_frequency,
    midi_to_note_name,
    note_name_to_frequency,
    note_name_to_midi,
    transpose,
)

__all__ = [
    # Messages
    "MIDIMessage",
    "NoteOnMessage",
    "NoteOffMessage",
    "ControlChangeMessage",
    "PitchBendMessage",
    "ProgramChangeMessage",
    "AftertouchMessage",
    # Utilities
    "midi_to_frequency",
    "frequency_to_midi",
    "note_name_to_midi",
    "midi_to_note_name",
    "note_name_to_frequency",
    "get_note_range",
    "transpose",
    # Input
    "MIDIInput",
    # File
    "MIDIFile",
    # Synth
    "MonophonicSynth",
    "PolyphonicSynth",
    "Voice",
    # CV Converter
    "MIDIToCV",
    "CVFrequencyOutput",
    "CVGateOutput",
    "CVVelocityOutput",
]
