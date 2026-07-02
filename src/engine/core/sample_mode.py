"""Shared sample-generation mode contract for engine components."""

from typing import Literal

SampleMode = Literal["auto", "iterator", "vectorized"]
VALID_SAMPLE_MODES: tuple[SampleMode, ...] = ("auto", "iterator", "vectorized")
