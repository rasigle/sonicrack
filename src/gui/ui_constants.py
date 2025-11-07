
from src.constants import RESOURCES_PATH

APP_ICON_NAME = "icon.png"
APP_ICON_PATH = RESOURCES_PATH / "icons" / APP_ICON_NAME
APP_TITLE = "AudioPlayground - Modular Synthesizer"

# Debounce compilation to avoid audio spikes during knob rotation
# Wait an amount of ms after last change before recompiling
DEBOUNCE_TIMER_DELAY_MS = 50


MIN_PW_PERCENTAGE_VALUE = 1
MAX_PW_PERCENTAGE_VALUE = 99
DEFAULT_PW_PERCENTAGE_VALUE = 50
