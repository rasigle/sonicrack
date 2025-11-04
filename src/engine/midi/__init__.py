"""MIDI support for AudioPlayground.

This package provides comprehensive MIDI functionality including:
- MIDI message handling (Note On/Off, CC, Pitch Bend)
- Real-time MIDI input from controllers
- MIDI file reading and playback
- Voice management for polyphony
- Note-to-frequency conversion utilities

Example - Basic MIDI Input:
    >>> from src.engine.midi import MIDIInput
    >>>
    >>> def on_message(msg):
    ...     print(f"Received: {msg}")
    >>>
    >>> midi = MIDIInput()
    >>> midi.start(on_message)

Example - MIDI File Playback:
    >>> from src.engine.midi import MIDIFile
    >>>
    >>> midi_file = MIDIFile("song.mid")
    >>> notes = midi_file.get_notes_in_range(0, 1.0)  # First second

Example - Note Conversion:
    >>> from src.engine.midi import midi_to_frequency, note_name_to_midi
    >>>
    >>> freq = midi_to_frequency(69)  # A4 = 440 Hz
    >>> note = note_name_to_midi("C4")  # Middle C = 60
"""

from src.engine.midi.messages import (
    MIDIMessage,
    NoteOnMessage,
    NoteOffMessage,
    ControlChangeMessage,
    PitchBendMessage,
    ProgramChangeMessage,
    AftertouchMessage,
)

from src.engine.midi.utils import (
    midi_to_frequency,
    frequency_to_midi,
    note_name_to_midi,
    midi_to_note_name,
    note_name_to_frequency,
    get_note_range,
    transpose,
)

from src.engine.midi.input import MIDIInput
from src.engine.midi.file_reader import MIDIFile
from src.engine.midi.monophonic_synth import MonophonicSynth
from src.engine.midi.polyphonic_synth import PolyphonicSynth
from src.engine.midi.midi_to_cv import MIDIToCV

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
    # CV Converter
    "MIDIToCV",
]
