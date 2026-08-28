"""Shared helpers for MIDI → CV GUI modules.

Thin adapters for soniclab ``MIDIToCV`` / ``PolyphonicMIDIToCV`` signals that
do not ship with dedicated output classes (mod wheel, expression, pitch bend).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from soniclab.core.component import AudioComponent
from soniclab.core.sample_mode import SampleMode
from soniclab.midi_io import MIDIToCV

from sonicrack.constants import DEFAULT_SAMPLE_RATE

NotePriority = Literal["last", "high", "low"]

PRIORITY_LABELS: dict[str, NotePriority] = {
    "Last": "last",
    "High": "high",
    "Low": "low",
}
PRIORITY_FROM_API: dict[NotePriority, str] = {
    "last": "Last",
    "high": "High",
    "low": "Low",
}


class ScalarCVOutput(AudioComponent):
    """Constant control-CV buffer driven by a zero-arg float getter."""

    def __init__(self, getter: Callable[[], float]):
        super().__init__()
        self._getter = getter

    def get_samples(
        self,
        n: int = DEFAULT_SAMPLE_RATE,
        reset: bool = False,
        mode: SampleMode = "auto",
    ) -> np.ndarray:
        del reset, mode
        return np.full(n, float(self._getter()), dtype=np.float32)

    def __iter__(self):
        return self

    def __next__(self) -> float:
        return float(self._getter())


def make_mod_wheel_output(cv: MIDIToCV) -> ScalarCVOutput:
    """CC#1 mod wheel, normalized 0–1."""
    return ScalarCVOutput(lambda: float(cv.mod_wheel))


def make_expression_output(cv: MIDIToCV) -> ScalarCVOutput:
    """CC#11 expression, normalized 0–1 (default 1.0)."""
    return ScalarCVOutput(lambda: float(cv.expression))


def make_pitch_bend_output(cv: MIDIToCV) -> ScalarCVOutput:
    """Pitch bend as bipolar CV in [-1, 1] relative to ``pitch_bend_range``."""

    def _bend() -> float:
        span = float(cv.pitch_bend_range) or 2.0
        return float(np.clip(cv.pitch_bend / span, -1.0, 1.0))

    return ScalarCVOutput(_bend)


def priority_from_label(label: str) -> NotePriority:
    """Map UI label to soniclab ``note_priority`` value."""
    return PRIORITY_LABELS.get(label, "last")


def label_from_priority(priority: str) -> str:
    """Map soniclab ``note_priority`` value to UI label."""
    if priority in PRIORITY_FROM_API:
        return PRIORITY_FROM_API[priority]  # type: ignore[index]
    return "Last"
