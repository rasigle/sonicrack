from pathlib import Path

from src.constants import RESOURCES_PATH

APP_TITLE: str = "AudioPlayground - Modular Synthesizer"
APP_ICON_NAME: str = "icon.png"
APP_ICON_PATH: Path = RESOURCES_PATH / "icons" / APP_ICON_NAME
DEFAULT_PRESET_DIRECTORY = Path.home() / ".audioplayground" / "presets"


# Debounce compilation to avoid audio spikes during knob rotation
# Wait an amount of ms after last change before recompiling
DEBOUNCE_TIMER_DELAY_MS = 50


MIN_PW_PERCENTAGE_VALUE = 1
MAX_PW_PERCENTAGE_VALUE = 99
DEFAULT_PW_PERCENTAGE_VALUE = 50
