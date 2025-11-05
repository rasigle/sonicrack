"""Constants used throughout the audio engine.

This module defines global constants for sample rates, buffer sizes,
and other audio-related values.
"""

from pathlib import Path

# Standard library paths
RESOURCES_PATH = Path(__file__).parent.parent / "resources"
SPLASH_PATH = Path(RESOURCES_PATH) / "splash" / "splash.png"

# Sample rate constants
DEFAULT_SAMPLE_RATE = 44100  # CD quality
SAMPLE_RATE_48K = 48000  # Professional audio
SAMPLE_RATE_96K = 96000  # High-resolution audio
SAMPLE_RATE_192K = 192000  # Ultra high-resolution

# Buffer size constants (in samples)
DEFAULT_BUFFER_SIZE = 512
BUFFER_SIZE_256 = 256  # Low latency
BUFFER_SIZE_1024 = 1024  # Higher latency, lower CPU
BUFFER_SIZE_2048 = 2048  # Maximum stability

# Bit depth
DEFAULT_BIT_DEPTH = 16
BIT_DEPTH_24 = 24
BIT_DEPTH_32 = 32

# Derived constants
NYQUIST_FREQUENCY = DEFAULT_SAMPLE_RATE / 2  # Maximum frequency we can represent

# Musical constants
A4_FREQUENCY = 440.0  # Concert pitch
SEMITONE_RATIO = 2 ** (1 / 12)  # Ratio between adjacent semitones

# Audio range
MIN_AMPLITUDE = -1.0
MAX_AMPLITUDE = 1.0

# Volume range in decibels
MIN_GAIN_DB = -96.0  # Near silence
MAX_GAIN_DB = 12.0  # Boost
DEFAULT_GAIN_DB = -20.0  # Default starting gain


# 🎵 Note mapping (keyboard keys → frequencies)
KEY_FREQUENCIES = {
    "a": 261.63,  # C4
    "w": 277.18,  # C#4
    "s": 293.66,  # D4
    "e": 311.13,  # D#4
    "d": 329.63,  # E4
    "f": 349.23,  # F4
    "t": 369.99,  # F#4
    "g": 392.00,  # G4
    "y": 415.30,  # G#4
    "h": 440.00,  # A4
    "u": 466.16,  # A#4
    "j": 493.88,  # B4
    "k": 523.25,  # C5
}
