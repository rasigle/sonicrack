"""Polyphonic MIDI synthesizer with multiple voices."""

import logging
from typing import Callable, Any
from dataclasses import dataclass
import numpy as np

from src.engine.midi.messages import NoteOnMessage, NoteOffMessage
from src.engine.midi.utils import midi_to_frequency
from src.constants import DEFAULT_SAMPLE_RATE

logger = logging.getLogger(__name__)


@dataclass
class Voice:
    """Represents a single synthesizer voice."""

    note: int | None = None
    velocity: int = 0
    component: Any | None = None
    is_active: bool = False
    age: int = 0

    def is_free(self) -> bool:
        return self.note is None and not self.is_active

    def is_releasing(self) -> bool:
        return not self.is_active and self.note is not None

    def has_ended(self) -> bool:
        if self.component is None:
            return True
        if hasattr(self.component, "ended") and self.component.ended:
            return True
        if hasattr(self.component, "modifiers"):
            for modifier in self.component.modifiers:
                if hasattr(modifier, "modulator"):
                    if (
                        hasattr(modifier.modulator, "ended")
                        and modifier.modulator.ended
                    ):
                        return True
        return False

    def clear(self):
        self.note = None
        self.velocity = 0
        self.component = None
        self.is_active = False
        self.age = 0


class PolyphonicSynth:
    """Polyphonic MIDI synthesizer."""

    def __init__(
        self,
        voice_factory: Callable[[], Any],
        max_voices: int = 8,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ):
        self._voice_factory = voice_factory
        self._sample_rate = sample_rate
        self.max_voices = max_voices
        self.voices: list[Voice] = [Voice() for _ in range(max_voices)]
        logger.info(f"PolyphonicSynth initialized with {max_voices} voices")

    def note_on(self, note: int, velocity: int = 100):
        if not 0 <= note <= 127 or not 0 <= velocity <= 127:
            return
        if velocity == 0:
            self.note_off(note)
            return

        voice = self._allocate_voice()
        voice.component = self._voice_factory()
        voice.note = note
        voice.velocity = velocity
        voice.is_active = True
        voice.age = 0

        frequency = midi_to_frequency(note)
        self._set_voice_frequency(voice, frequency)
        self._trigger_voice_on(voice)

    def note_off(self, note: int):
        for voice in self.voices:
            if voice.note == note and voice.is_active:
                self._trigger_voice_off(voice)
                voice.is_active = False

    def process_message(self, message: NoteOnMessage | NoteOffMessage):
        if isinstance(message, NoteOnMessage):
            self.note_on(message.note, message.velocity)
        elif isinstance(message, NoteOffMessage):
            self.note_off(message.note)

    def get_samples(self, num_samples: int) -> np.ndarray:
        output = None
        active_voices = 0

        for voice in self.voices:
            if voice.is_free():
                continue

            voice.age += num_samples

            if voice.component is not None:
                try:
                    samples = voice.component.get_samples(num_samples)
                    if output is None:
                        output = np.zeros_like(samples, dtype=np.float32)
                    output += samples
                    active_voices += 1

                    if voice.has_ended():
                        voice.clear()
                except Exception as e:
                    logger.error(f"Error generating samples: {e}")
                    voice.clear()

        if output is None:
            return np.zeros(num_samples, dtype=np.float32)

        if active_voices > 1:
            output = output / np.sqrt(active_voices)

        return output

    def _allocate_voice(self) -> Voice:
        # Try free voice
        for voice in self.voices:
            if voice.is_free():
                return voice

        # Steal releasing voice
        releasing = [v for v in self.voices if v.is_releasing()]
        if releasing:
            oldest = max(releasing, key=lambda v: v.age)
            oldest.clear()
            return oldest

        # Steal active voice
        active = [v for v in self.voices if v.is_active]
        if active:
            oldest = max(active, key=lambda v: v.age)
            oldest.clear()
            return oldest

        self.voices[0].clear()
        return self.voices[0]

    def _set_voice_frequency(self, voice: Voice, frequency: float):
        comp = voice.component
        if comp is None:
            return
        if hasattr(comp, "oscillator") and hasattr(comp.oscillator, "frequency"):
            comp.oscillator.frequency = frequency
        elif hasattr(comp, "frequency"):
            comp.frequency = frequency

    def _trigger_voice_on(self, voice: Voice):
        comp = voice.component
        if comp is None:
            return
        if hasattr(comp, "trigger_note_on"):
            comp.trigger_note_on()
        elif hasattr(comp, "modifiers"):
            for mod in comp.modifiers:
                if hasattr(mod, "modulator") and hasattr(
                    mod.modulator, "trigger_note_on"
                ):
                    mod.modulator.trigger_note_on()
                    return

    def _trigger_voice_off(self, voice: Voice):
        comp = voice.component
        if comp is None:
            return
        if hasattr(comp, "trigger_note_off"):
            comp.trigger_note_off()
        elif hasattr(comp, "modifiers"):
            for mod in comp.modifiers:
                if hasattr(mod, "modulator") and hasattr(
                    mod.modulator, "trigger_note_off"
                ):
                    mod.modulator.trigger_note_off()
                    return

    def get_active_voice_count(self) -> int:
        return sum(1 for v in self.voices if not v.is_free())

    def get_free_voice_count(self) -> int:
        return sum(1 for v in self.voices if v.is_free())

    def reset(self):
        for voice in self.voices:
            voice.clear()

    def __repr__(self) -> str:
        active = self.get_active_voice_count()
        free = self.get_free_voice_count()
        return (
            f"PolyphonicSynth({active} active, {free} free / {self.max_voices} total)"
        )
