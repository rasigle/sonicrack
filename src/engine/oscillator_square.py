"""Square wave generation strategies.

This module provides different algorithms for generating square waves,
from ideal (aliased) to bandlimited (antialiased) versions.

The strategy pattern allows easy switching between formulations while
maintaining clean, testable code.
"""
import logging
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

from src.constants import DEFAULT_GAIN_DB, DEFAULT_SAMPLE_RATE
from src.engine.oscillator import SineOscillator
from src.engine.audio_component_registry import (
    register_component,
    ComponentDescriptor,
    ComponentCategory,
)
from src.utils.utils import track_provided_args, filter_provided_args

