"""Audio utility functions for file I/O, playback, and conversions.

This module provides essential utilities for working with audio data:
- File I/O: Reading and writing WAV files
- Playback: Playing audio through system audio devices
- Conversion: Converting between formats and bit depths
- Music theory: Note name to frequency conversion

These utilities are designed to work seamlessly with the audio engine
components and provide a consistent interface for audio operations.
"""

import re

import numpy as np
from scipy.io import wavfile

from src.constants import DEFAULT_SAMPLE_RATE


_NOTE_OFFSETS = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "FB": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
    "CB": 11,
}


def to_int16(audio: np.ndarray | list, amplitude: float = 1.0) -> np.ndarray:
    """Convert floating-point audio samples to 16-bit integer format.

    Converts normalized audio data (typically in range [-1.0, 1.0]) to
    16-bit signed integer format suitable for WAV file storage and playback.

    Args:
        audio: Audio samples as numpy array or list, expected in range [-1.0, 1.0].
        amplitude: Amplitude scaling factor (0.0 to 1.0). Values > 1.0 may cause
            clipping.

    Returns:
        np.ndarray: Audio samples as 16-bit signed integers.

    Example:
        >>> samples = np.array([0.0, 0.5, -0.5, 1.0])
        >>> int_samples = to_int16(samples, amplitude=0.8)
        >>> int_samples.dtype
        dtype('int16')

    Note:
        Amplitude scaling is applied before conversion. Use amplitude < 1.0
        to prevent clipping when combining multiple audio sources.
    """
    audio_array = np.asarray(audio, dtype=np.float64)
    # Scale and convert to 16-bit range
    scaled = audio_array * amplitude * (2**15 - 1)
    return np.clip(scaled, -(2**15), 2**15 - 1).astype(np.int16)


def play_wave(
    audio: np.ndarray | list,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    amplitude: float = 0.5,
    blocking: bool = True,
) -> None:
    """Play audio samples through the default audio output device.

    Args:
        audio: Audio samples to play (mono or stereo).
        sample_rate: Sample rate in Hz.
        amplitude: Playback amplitude (0.0 to 1.0). Default 0.5 for safety.
        blocking: If True, waits for playback to complete. If False, returns
            immediately.

    Example:
        >>> import numpy as np
        >>> # Generate a 1-second 440 Hz sine wave
        >>> t = np.linspace(0, 1, 44100)
        >>> samples = np.sin(2 * np.pi * 440 * t)
        >>> play_wave(samples)

    Note:
        Requires sounddevice package. This function converts to 16-bit
        format automatically and uses the specified amplitude for safety.
    """
    import sounddevice as sd

    audio_int16 = to_int16(audio, amplitude)
    sd.play(audio_int16, samplerate=sample_rate)
    if blocking:
        sd.wait()


def save_wave(
    audio: np.ndarray | list,
    audio_right: np.ndarray | list | None = None,
    filename: str = "temp.wav",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    amplitude: float = 0.1,
) -> str:
    """Save audio samples to a WAV file.

    Saves mono or stereo audio to a WAV file. Automatically adds .wav
    extension if not present and converts to 16-bit integer format.

    Args:
        audio: Audio samples for mono output or left channel.
        filename: Output filename. '.wav' extension added if not present.
        sample_rate: Sample rate in Hz.
        audio_right: Optional right channel audio for stereo output.
        amplitude: Amplitude scaling (0.0 to 1.0) to prevent clipping.

    Returns:
        str: The actual filename written (with .wav extension).

    Example:
        >>> # Save mono audio
        >>> samples = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> save_wave(samples,filename="tone.wav")
        'tone.wav'

        >>> # Save stereo audio
        >>> left = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> right = np.sin(2 * np.pi * 554 * np.linspace(0, 1, 44100))
        >>> save_wave(left,audio_right=right,filename="stereo.wav")
        'stereo.wav'

    Note:
        Default amplitude is 0.1 to prevent clipping when multiple
        sources are combined. Adjust as needed for your use case.
    """
    # Ensure .wav extension
    if not filename.endswith(".wav"):
        filename += ".wav"

    # Convert to 16-bit
    audio_int16 = to_int16(audio, amplitude)

    # Handle stereo
    if audio_right is not None:
        audio_right_int16 = to_int16(audio_right, amplitude)
        audio_int16 = np.stack([audio_int16, audio_right_int16], axis=-1)

    wavfile.write(filename, sample_rate, audio_int16)
    return filename


def load_wave(filename: str, mono: bool = True) -> tuple[int, np.ndarray]:
    """Load audio from a WAV file.

    Reads a WAV file and returns the sample rate and audio data as
    normalized floating-point values in the range [-1.0, 1.0].

    Args:
        filename: Path to the WAV file.
        mono: If True, converts stereo to mono by averaging channels.

    Returns:
        Tuple containing:
            - sample_rate (int): Sample rate of the audio file in Hz.
            - audio (np.ndarray): Audio samples as float32 in range [-1.0, 1.0].

    Example:
        >>> sr, audio_data = load_wave("recording.wav")
        >>> print(f"Loaded {len(audio_data)} samples at {sr} Hz")
        >>> print(f"Duration: {len(audio_data) / sr:.2f} seconds")

    Note:
        The returned audio is always normalized to [-1.0, 1.0] range
        regardless of the original bit depth.
    """
    sample_rate, audio = wavfile.read(filename)

    audio = np.asarray(audio)

    if np.issubdtype(audio.dtype, np.integer):
        if audio.dtype == np.uint8:
            audio = (audio.astype(np.float32) - 128.0) / 128.0
        else:
            info = np.iinfo(audio.dtype)
            scale = float(max(abs(info.min), info.max))
            audio = audio.astype(np.float32) / scale
    elif np.issubdtype(audio.dtype, np.floating):
        audio = audio.astype(np.float32)
    else:
        raise TypeError(f"Unsupported WAV data type: {audio.dtype}")

    # Convert stereo to mono if requested
    if mono and audio.ndim > 1:
        audio = audio.mean(axis=1)

    return sample_rate, audio.astype(np.float32)


def note_to_frequency(note: str) -> float:
    """Convert a musical note name to its frequency in Hz.

    Uses standard concert pitch (A4 = 440 Hz) and equal temperament tuning.
    Supports note names with octave numbers and accidentals.

    Args:
        note: Note name (e.g., 'A4', 'C#5', 'Bb3', 'F#4').

    Returns:
        float: Frequency in Hz.

    Example:
        >>> note_to_frequency('A4')
        440.0
        >>> note_to_frequency('C4')
        261.6255653005986
        >>> freq = note_to_frequency('E5')
        >>> print(f"{freq:.2f} Hz")
        659.25 Hz

    Note:
        Uses equal temperament with A4 = 440 Hz.
    """
    if not isinstance(note, str):
        raise TypeError(f"Note name must be a string, got {type(note).__name__}")

    match = re.fullmatch(r"\s*([A-Ga-g])([#bB]?)(-?\d+)\s*", note)
    if match is None:
        raise ValueError(f"Invalid note name: {note!r}")

    note_name = f"{match.group(1).upper()}{match.group(2).upper()}"
    octave = int(match.group(3))

    if note_name not in _NOTE_OFFSETS:
        raise ValueError(f"Invalid note name: {note!r}")

    midi_note = (octave + 1) * 12 + _NOTE_OFFSETS[note_name]
    return float(440.0 * (2.0 ** ((midi_note - 69) / 12.0)))


def mono_to_stereo(samples: np.ndarray) -> np.ndarray:
    """Convert mono samples to stereo by duplicating to both channels.

    Args:
        samples: Mono audio samples (any shape)

    Returns:
        Stereo samples as (N, 2) array where both channels are identical

    Example:
        >>> mono = np.array([0.1, 0.2, 0.3])
        >>> stereo = mono_to_stereo(mono)
        >>> stereo.shape
        (3, 2)
        >>> np.allclose(stereo[:, 0], stereo[:, 1])
        True
    """
    # Handle scalar
    if samples.ndim == 0:
        val = float(samples)
        return np.array([[val, val]])

    # Handle 1D array (most common case)
    if samples.ndim == 1:
        if samples.size == 1:
            val = float(samples[0])
            return np.array([[val, val]])
        else:
            # Duplicate to stereo: (N,) -> (N, 2)
            return np.column_stack((samples, samples))

    # Handle 2D array
    if samples.ndim == 2:
        rows, cols = samples.shape

        # Already stereo (N, 2)
        if cols == 2:
            return samples

        # Transposed stereo (2, N) -> (N, 2)
        if rows == 2 and cols != 2:
            return samples.T

        # Single column (N, 1) -> (N, 2)
        if cols == 1:
            return np.repeat(samples, 2, axis=1)

        # Single row (1, N) -> (N, 2)
        if rows == 1:
            return np.repeat(samples.T, 2, axis=1)

        # Multiple columns: average and duplicate
        mean_vals = samples.mean(axis=1)
        return np.column_stack((mean_vals, mean_vals))

    # Handle higher dimensions: flatten first
    flat = samples.reshape(-1)
    if flat.size == 1:
        val = float(flat[0])
        return np.array([[val, val]])
    else:
        return np.column_stack((flat, flat))


def combine_lr_to_stereo(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Combine left and right mono signals into stereo output.

    Args:
        left: Left channel samples (any shape)
        right: Right channel samples (any shape)

    Returns:
        Stereo samples as (N, 2) array with left in column 0, right in column 1

    Example:
        >>> left = np.array([0.1, 0.2, 0.3])
        >>> right = np.array([0.4, 0.5, 0.6])
        >>> stereo = combine_lr_to_stereo(left, right)
        >>> stereo.shape
        (3, 2)
        >>> np.allclose(stereo[:, 0], left)
        True
        >>> np.allclose(stereo[:, 1], right)
        True

    Note:
        If left and right have different lengths, the shorter one is
        zero-padded to match the longer one.
    """
    # Flatten both to 1D arrays
    left_flat = np.asarray(left).reshape(-1)
    right_flat = np.asarray(right).reshape(-1)

    # Handle empty arrays
    if left_flat.size == 0 and right_flat.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # Determine the length (use max, pad shorter one)
    max_len = max(left_flat.size, right_flat.size)

    # Pad shorter array with zeros if needed
    if left_flat.size < max_len:
        left_flat = np.pad(left_flat, (0, max_len - left_flat.size), mode="constant")
    if right_flat.size < max_len:
        right_flat = np.pad(right_flat, (0, max_len - right_flat.size), mode="constant")

    # Combine into stereo: (N, 2)
    return np.column_stack((left_flat, right_flat))
