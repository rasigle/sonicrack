"""App and shared audio constants for SonicRack.

Shared engine audio constants are imported from ``soniclab.constants`` so host
and engine stay aligned. App-only paths, resources, and UI curves remain here.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from importlib.resources import as_file, files
from importlib.resources.abc import Traversable
from pathlib import Path

# Shared audio engine constants (single source of truth in soniclab).
from soniclab.constants import (  # noqa: F401
    A4_FREQUENCY,
    AUTO_MODE_VECTORIZE_THRESHOLD,
    BIT_DEPTH_24,
    BIT_DEPTH_32,
    BUFFER_SIZE_256,
    BUFFER_SIZE_1024,
    BUFFER_SIZE_2048,
    DEFAULT_BIT_DEPTH,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_BUFFER_SIZES,
    DEFAULT_GAIN_DB,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SAMPLE_RATES,
    KEY_FREQUENCIES,
    MAX_AMPLITUDE,
    MAX_GAIN_DB,
    MIN_AMPLITUDE,
    MIN_GAIN_DB,
    NYQUIST_FREQUENCY,
    SAMPLE_RATE_48K,
    SAMPLE_RATE_96K,
    SAMPLE_RATE_192K,
    SEMITONE_RATIO,
)

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
