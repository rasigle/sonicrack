from collections.abc import Sequence
from pathlib import Path

from src.constants import RESOURCES_PATH

APP_TITLE: str = "AudioPlayground - Modular Synthesizer"
APP_ICON_NAME: str = "icon.png"
APP_ICON_PATH: Path = RESOURCES_PATH / "icons" / APP_ICON_NAME
DEFAULT_PRESET_DIRECTORY = Path.home() / ".audioplayground" / "presets"

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


# Default options (can be extended later or loaded from config file)
DEFAULT_SAMPLE_RATES: Sequence[int] = (22050, 44100, 48000, 88200, 96000)
DEFAULT_BUFFER_SIZES: Sequence[int] = (128, 256, 512, 1024, 2048, 4096)
