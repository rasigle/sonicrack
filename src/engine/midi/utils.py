"""MIDI utility functions for note conversion and manipulation.

This module provides essential utilities for working with MIDI notes:
- Convert between MIDI note numbers and frequencies
- Parse note names ("C4", "A#5", etc.)
- Convert to scientific pitch notation

Standard Tuning:
    A4 = 440 Hz = MIDI note 69 (concert pitch)

Note Numbering:
    - C-1 = MIDI note 0 (8.176 Hz)
    - C0 = MIDI note 12
    - C4 (Middle C) = MIDI note 60 (261.626 Hz)
    - C5 = MIDI note 72
    - G9 = MIDI note 127 (12543.854 Hz)

Example:
    >>> freq = midi_to_frequency(60)  # Middle C
    >>> print(f"C4 = {freq:.2f} Hz")
    C4 = 261.63 Hz

    >>> note = note_name_to_midi("A4")
    >>> print(f"A4 = MIDI note {note}")
    A4 = MIDI note 69
"""

import re
from typing import Optional


# Note name to semitone mapping (C = 0)
NOTE_NAMES = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11,
}

# Reverse mapping for note number to name
SEMITONE_TO_NOTE = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
]


def midi_to_frequency(note: int, a4_tuning: float = 440.0) -> float:
    """Convert MIDI note number to frequency in Hz.

    Uses equal temperament tuning with A4 as reference.
    Formula: f = a4_tuning * 2^((note - 69) / 12)

    Args:
        note: MIDI note number (0-127)
        a4_tuning: Frequency of A4 in Hz (default 440 Hz, concert pitch)

    Returns:
        Frequency in Hz

    Raises:
        ValueError: If note is not in range 0-127

    Example:
        >>> midi_to_frequency(69)  # A4
        440.0
        >>> midi_to_frequency(60)  # C4 (Middle C)
        261.6255653005986
        >>> midi_to_frequency(72)  # C5
        523.2511306011972
    """
    if not 0 <= note <= 127:
        raise ValueError(f"MIDI note must be 0-127, got {note}")

    # Equal temperament formula
    return a4_tuning * (2 ** ((note - 69) / 12))


def frequency_to_midi(frequency: float, a4_tuning: float = 440.0) -> int:
    """Convert frequency in Hz to nearest MIDI note number.

    Args:
        frequency: Frequency in Hz (must be positive)
        a4_tuning: Frequency of A4 in Hz (default 440 Hz)

    Returns:
        Nearest MIDI note number (0-127)

    Raises:
        ValueError: If frequency is not positive

    Example:
        >>> frequency_to_midi(440.0)  # A4
        69
        >>> frequency_to_midi(261.63)  # Close to C4
        60
    """
    if frequency <= 0:
        raise ValueError(f"Frequency must be positive, got {frequency}")

    # Inverse of midi_to_frequency formula
    import math
    note = 69 + 12 * math.log2(frequency / a4_tuning)

    # Round to nearest integer and clamp to valid range
    note = max(0, min(127, round(note)))

    return int(note)


def note_name_to_midi(note_name: str) -> int:
    """Convert note name to MIDI note number.

    Supports standard note names with octave:
    - Letter: C, D, E, F, G, A, B
    - Accidentals: # (sharp), b (flat)
    - Octave: -1 to 9

    Args:
        note_name: Note name (e.g., "C4", "A#5", "Bb3")

    Returns:
        MIDI note number (0-127)

    Raises:
        ValueError: If note name is invalid or out of range

    Example:
        >>> note_name_to_midi("C4")  # Middle C
        60
        >>> note_name_to_midi("A4")  # Concert A
        69
        >>> note_name_to_midi("C#5")
        73
        >>> note_name_to_midi("Bb3")
        58
    """
    # Parse note name with regex: (Note)(Accidental?)(Octave)
    pattern = r'^([A-G])(#|b)?(-?\d+)$'
    match = re.match(pattern, note_name)

    if not match:
        raise ValueError(
            f"Invalid note name '{note_name}'. "
            f"Expected format: [A-G][#/b]?[-1 to 9]. Examples: C4, A#5, Bb3"
        )

    note, accidental, octave = match.groups()
    octave = int(octave)

    # Build full note string
    full_note = note + (accidental if accidental else "")

    # Look up semitone offset
    if full_note not in NOTE_NAMES:
        raise ValueError(f"Unknown note '{full_note}'")

    semitone = NOTE_NAMES[full_note]

    # Calculate MIDI note: (octave + 1) * 12 + semitone
    # C-1 = 0, C0 = 12, C1 = 24, ..., C4 = 60
    midi_note = (octave + 1) * 12 + semitone

    # Validate range
    if not 0 <= midi_note <= 127:
        raise ValueError(
            f"Note '{note_name}' = MIDI note {midi_note}, "
            f"which is out of range (0-127)"
        )

    return midi_note


def midi_to_note_name(note: int, use_sharps: bool = True) -> str:
    """Convert MIDI note number to note name.

    Args:
        note: MIDI note number (0-127)
        use_sharps: If True, use sharps (#), otherwise use flats (b)

    Returns:
        Note name (e.g., "C4", "A#5", "Bb3")

    Raises:
        ValueError: If note is not in range 0-127

    Example:
        >>> midi_to_note_name(60)
        'C4'
        >>> midi_to_note_name(69)
        'A4'
        >>> midi_to_note_name(61)
        'C#4'
        >>> midi_to_note_name(61, use_sharps=False)
        'Db4'
    """
    if not 0 <= note <= 127:
        raise ValueError(f"MIDI note must be 0-127, got {note}")

    # Calculate octave: note 12 = C0, note 60 = C4
    octave = (note // 12) - 1
    semitone = note % 12

    # Get note name
    note_letter = SEMITONE_TO_NOTE[semitone]

    # Convert sharps to flats if requested
    if not use_sharps and "#" in note_letter:
        # Sharp to flat conversion
        sharp_to_flat = {
            "C#": "Db", "D#": "Eb", "F#": "Gb",
            "G#": "Ab", "A#": "Bb"
        }
        note_letter = sharp_to_flat.get(note_letter, note_letter)

    return f"{note_letter}{octave}"


def note_name_to_frequency(note_name: str, a4_tuning: float = 440.0) -> float:
    """Convert note name directly to frequency.

    Convenience function combining note_name_to_midi and midi_to_frequency.

    Args:
        note_name: Note name (e.g., "C4", "A#5")
        a4_tuning: Frequency of A4 in Hz (default 440 Hz)

    Returns:
        Frequency in Hz

    Example:
        >>> note_name_to_frequency("A4")
        440.0
        >>> note_name_to_frequency("C4")
        261.6255653005986
    """
    midi_note = note_name_to_midi(note_name)
    return midi_to_frequency(midi_note, a4_tuning)


def get_note_range(start_note: str, end_note: str) -> list[int]:
    """Get all MIDI notes in a range (inclusive).

    Args:
        start_note: Starting note name (e.g., "C3")
        end_note: Ending note name (e.g., "C5")

    Returns:
        List of MIDI note numbers in range

    Example:
        >>> get_note_range("C4", "E4")
        [60, 61, 62, 63, 64]  # C4, C#4, D4, D#4, E4
    """
    start = note_name_to_midi(start_note)
    end = note_name_to_midi(end_note)

    if start > end:
        raise ValueError(f"Start note '{start_note}' is higher than end note '{end_note}'")

    return list(range(start, end + 1))


def transpose(note: int, semitones: int) -> int:
    """Transpose a MIDI note by semitones.

    Args:
        note: Original MIDI note (0-127)
        semitones: Number of semitones to transpose (can be negative)

    Returns:
        Transposed MIDI note, clamped to 0-127

    Example:
        >>> transpose(60, 12)  # C4 up one octave
        72  # C5
        >>> transpose(60, -12)  # C4 down one octave
        48  # C3
    """
    if not 0 <= note <= 127:
        raise ValueError(f"MIDI note must be 0-127, got {note}")

    transposed = note + semitones
    return max(0, min(127, transposed))

