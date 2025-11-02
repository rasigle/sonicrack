"""Audio utility functions for file I/O, playback, and conversions.

This module provides essential utilities for working with audio data:
- File I/O: Reading and writing WAV files
- Playback: Playing audio through system audio devices
- Conversion: Converting between formats and bit depths
- Music theory: Note name to frequency conversion

These utilities are designed to work seamlessly with the audio engine
components and provide a consistent interface for audio operations.
"""

import librosa
import numpy as np
from scipy.io import wavfile
import soundfile as sf

from src.constants import DEFAULT_SAMPLE_RATE


def to_int16(
    audio: np.ndarray | list, amplitude: float = 1.0
) -> np.ndarray:
    """Convert floating-point audio samples to 16-bit integer format.

    Converts normalized audio data (typically in range [-1.0, 1.0]) to
    16-bit signed integer format suitable for WAV file storage and playback.

    Args:
        audio: Audio samples as numpy array or list, expected in range [-1.0, 1.0].
        amplitude: Amplitude scaling factor (0.0 to 1.0). Values > 1.0 may cause clipping.

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
    return np.clip(scaled, -2**15, 2**15 - 1).astype(np.int16)


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
        blocking: If True, waits for playback to complete. If False, returns immediately.

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
        amplitude: float = 0.1) -> str:
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
        >>> sr, audio = load_wave("recording.wav")
        >>> print(f"Loaded {len(audio)} samples at {sr} Hz")
        >>> print(f"Duration: {len(audio) / sr:.2f} seconds")

    Note:
        The returned audio is always normalized to [-1.0, 1.0] range
        regardless of the original bit depth.
    """
    audio, sample_rate = sf.read(filename)

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
        This function uses librosa's note_to_hz conversion which follows
        standard Western music notation with equal temperament tuning.
    """
    return float(librosa.note_to_hz(note))
