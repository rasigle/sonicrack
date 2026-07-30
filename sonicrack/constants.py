"""Constants used throughout the audio engine.

This module defines global constants for sample rates, buffer sizes,
and other audio-related values.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from importlib.resources import as_file, files
from importlib.resources.abc import Traversable
from pathlib import Path

APP_TITLE: str = "SonicRack - Modular Synthesizer"
APP_ICON_NAME: str = "icon.png"
APP_ICON_RESOURCE: tuple[str, str] = ("icons", APP_ICON_NAME)
DEFAULT_PRESET_DIRECTORY = Path.home() / ".sonicrack" / "presets"
DEFAULT_AUTOSAVE_PATCH = Path.home() / ".sonicrack" / "last_session.apr"
DEFAULT_SETTINGS_FILE = Path.home() / ".sonicrack" / "settings.json"

# Logging
LOG_FILENAME = "sonicrack.log"
CRASH_TRACE_FILENAME = "sonicrack_fault_trace.log"
LOG_DIRECTORY: Path = Path(__file__).parent.parent

# Package resources
RESOURCE_PACKAGE = "sonicrack.resources"
RESOURCES = files(RESOURCE_PACKAGE)
SPLASH_RESOURCE = ("splash", "splash.png")
PRESET_FILE_EXTENSION = ".apr"  # Audio Preset file extension


def resource(*parts: str) -> Traversable:
    """Return a package resource without assuming a filesystem layout."""
    current = RESOURCES
    for part in parts:
        current = current.joinpath(part)
    return current


@contextmanager
def resource_path(*parts: str) -> Iterator[Path]:
    """Yield a filesystem path for a package resource."""
    with as_file(resource(*parts)) as path:
        yield path


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

# Auto-mode threshold: use vectorized generation for n >= this value
AUTO_MODE_VECTORIZE_THRESHOLD = 512

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
DEFAULT_GAIN_DB = 0.0  # Default starting gain


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

# Pulsewidth modulation limits
MIN_PW_PERCENTAGE_VALUE = 1
MAX_PW_PERCENTAGE_VALUE = 99
DEFAULT_PW_PERCENTAGE_VALUE = 50

# Audio-rate frequency knob anchors. Positions correspond roughly to a 270-degree
# hardware-style sweep from 7 o'clock to 5 o'clock:
# 7:00=11 Hz, 9:00=40 Hz, 12:00=282 Hz, 3:00=1715 Hz, 5:00=6000 Hz.

AUDIO_FREQUENCY_KNOB_CURVE: Sequence[tuple[float, float]] = (
    (0.0, 11.0),
    (0.2, 40.0),
    (0.5, 282.0),
    (0.8, 1715.0),
    (1.0, 6000.0),
)

DEFAULT_SAMPLE_RATES: Sequence[int] = (22050, 44100, 48000, 88200, 96000)
DEFAULT_BUFFER_SIZES: Sequence[int] = (128, 256, 512, 1024, 2048, 4096)
