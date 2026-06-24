"""Monophonic MIDI-controlled synthesizer.

This module provides a simple monophonic (single-voice) synthesizer that responds
to MIDI note on/off messages. It plays one note at a time, with the most recent
note taking priority.

Example:
    >>> from src.midi_io import MonophonicSynth, NoteOnMessage, NoteOffMessage
    >>> from src.engine import SineOscillator, ADSREnvelope, Chain, ModulatedVolume
    >>>
    >>> # Create a simple synth
    >>> def voice_factory():
    ...     osc = SineOscillator(440)
    ...     env = ADSREnvelope(attack_duration=0.1, release_duration=0.2)
    ...     return Chain(osc, ModulatedVolume(env))
    >>>
    >>> synth = MonophonicSynth(voice_factory)
    >>>
    >>> # Trigger a note
    >>> synth.note_on(60, 100)  # Middle C, velocity 100
    >>> samples = synth.get_samples(1000)
    >>>
    >>> # Release the note
    >>> synth.note_off(60)
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from src.constants import DEFAULT_SAMPLE_RATE
from src.midi_io.messages import NoteOffMessage, NoteOnMessage
from src.midi_io.utils import midi_to_frequency

logger = logging.getLogger(__name__)


class MonophonicSynth:
    """Simple monophonic (single-voice) MIDI synthesizer.

    This synthesizer plays one note at a time. When a new note is triggered
    while another is playing, it immediately switches to the new note.

    The synthesizer requires a voice factory function that creates the audio
    generation component (oscillator + envelope + effects).

    Attributes:
        voice: Current voice component
        current_note: Currently playing MIDI note number (or None)
        is_playing: Whether a note is currently active

    Example:
        >>> # Create voice factory
        >>> from src.engine import SineOscillator, ADSREnvelope, Chain, ModulatedVolume
        >>>
        >>> def make_voice():
        ...     osc = SineOscillator(440)
        ...     env = ADSREnvelope(attack_duration=0.05, release_duration=0.1)
        ...     return Chain(osc, ModulatedVolume(env))
        >>>
        >>> # Create synth
        >>> synth = MonophonicSynth(make_voice)
        >>>
        >>> # Play notes
        >>> synth.note_on(60, 100)   # C4
        >>> audio1 = synth.get_samples(1000)
        >>> synth.note_off(60)
        >>> audio2 = synth.get_samples(1000)  # Release tail
    """

    def __init__(
        self, voice_factory: Callable[[], Any], sample_rate: int = DEFAULT_SAMPLE_RATE
    ):
        """Initialize monophonic synthesizer.

        Args:
            voice_factory: Function that creates a new voice component.
                          Should return an object with get_samples() method.
            sample_rate: Audio sample rate in Hz

        Example:
            >>> def voice_factory():
            ...     return SineOscillator(440)
            >>> synth = MonophonicSynth(voice_factory)
        """
        self._voice_factory = voice_factory
        self._sample_rate = sample_rate

        # Voice state
        self.voice: Any | None = None
        self.current_note: int | None = None
        self.current_velocity: int = 0
        self.is_playing: bool = False

        logger.debug("MonophonicSynth initialized")

    def note_on(self, note: int, velocity: int = 100):
        """Trigger a note.

        If a note is already playing, it will be immediately replaced by
        the new note (monophonic behavior).

        Args:
            note: MIDI note number (0-127)
            velocity: Note velocity (0-127), where 0 is silent

        Example:
            >>> synth.note_on(60, 100)  # Play middle C at velocity 100
        """
        if not 0 <= note <= 127:
            logger.warning(f"Invalid MIDI note: {note}")
            return

        if not 0 <= velocity <= 127:
            logger.warning(f"Invalid velocity: {velocity}")
            return

        # Velocity 0 is note off
        if velocity == 0:
            self.note_off(note)
            return

        # Create new voice and set frequency
        self.voice = self._voice_factory()
        frequency = midi_to_frequency(note)

        # Set oscillator frequency - try multiple paths
        # NOTE: Don't check hasattr(voice, 'frequency') first because Chain.__getattr__
        # proxies it but Chain.__setattr__ doesn't, so setting would create a new
        # attribute
        osc_found = False

        # 1. Chain/common: oscillator attribute (most common pattern)
        if hasattr(self.voice, "oscillator"):
            if hasattr(self.voice.oscillator, "frequency"):
                self.voice.oscillator.frequency = frequency
                osc_found = True
                logger.debug("Set frequency via voice.oscillator.frequency")

        # 2. Alternative: generator attribute
        elif hasattr(self.voice, "generator"):
            if hasattr(self.voice.generator, "frequency"):
                self.voice.generator.frequency = frequency
                osc_found = True
                logger.debug("Set frequency via voice.generator.frequency")

        # 3. Direct oscillator (no wrapper)
        elif hasattr(self.voice, "frequency"):
            self.voice.frequency = frequency
            osc_found = True
            logger.debug("Set frequency via voice.frequency")

        # 4. Search in components (for other structures)
        elif hasattr(self.voice, "components"):
            for comp in self.voice.components:
                if hasattr(comp, "frequency"):
                    comp.frequency = frequency
                    osc_found = True
                    logger.debug("Set frequency via component.frequency")
                    break

        if not osc_found:
            logger.warning("Could not find frequency attribute to set")

        # Trigger envelope if voice has it - try multiple paths
        envelope_triggered = False

        # 1. Direct trigger methods
        if hasattr(self.voice, "trigger_note_on"):
            self.voice.trigger_note_on()
            envelope_triggered = True
            logger.debug("Triggered note on via voice.trigger_note_on()")

        # 2. Direct envelope attribute
        elif hasattr(self.voice, "envelope"):
            if hasattr(self.voice.envelope, "trigger_note_on"):
                self.voice.envelope.trigger_note_on()
                envelope_triggered = True
                logger.debug("Triggered note on via voice.envelope.trigger_note_on()")

        # 3. Search in modifiers (for Chain)
        elif hasattr(self.voice, "modifiers"):
            for modifier in self.voice.modifiers:
                # Check if modifier has modulator (like ModulatedVolume)
                if hasattr(modifier, "modulator"):
                    if hasattr(modifier.modulator, "trigger_note_on"):
                        modifier.modulator.trigger_note_on()
                        envelope_triggered = True
                        logger.debug("Triggered note on via modifier.modulator")
                        break
                # Check if modifier itself can be triggered
                elif hasattr(modifier, "trigger_note_on"):
                    modifier.trigger_note_on()
                    envelope_triggered = True
                    logger.debug("Triggered note on via modifier")
                    break

        if not envelope_triggered:
            logger.debug("No envelope found to trigger")

        self.current_note = note
        self.current_velocity = velocity
        self.is_playing = True

        logger.debug(f"Note ON: {note} ({frequency:.2f} Hz, velocity {velocity})")

    def note_off(self, note: int):
        """Release a note.

        Only releases if the given note matches the currently playing note.

        Args:
            note: MIDI note number (0-127)

        Example:
            >>> synth.note_off(60)  # Release middle C
        """
        if self.current_note == note:
            # Trigger envelope release if voice has it - try multiple paths
            if self.voice is not None:
                envelope_released = False

                # 1. Direct trigger methods
                if hasattr(self.voice, "trigger_note_off"):
                    self.voice.trigger_note_off()
                    envelope_released = True
                    logger.debug("Triggered note off via voice.trigger_note_off()")

                # 2. Direct envelope attribute
                elif hasattr(self.voice, "envelope"):
                    if hasattr(self.voice.envelope, "trigger_note_off"):
                        self.voice.envelope.trigger_note_off()
                        envelope_released = True
                        logger.debug(
                            "Triggered note off via voice.envelope.trigger_note_off()"
                        )

                # 3. Search in modifiers (for Chain)
                elif hasattr(self.voice, "modifiers"):
                    for modifier in self.voice.modifiers:
                        # Check if modifier has modulator (like ModulatedVolume)
                        if hasattr(modifier, "modulator"):
                            if hasattr(modifier.modulator, "trigger_note_off"):
                                modifier.modulator.trigger_note_off()
                                envelope_released = True
                                logger.debug(
                                    "Triggered note off via modifier.modulator"
                                )
                                break
                        # Check if modifier itself can be triggered
                        elif hasattr(modifier, "trigger_note_off"):
                            modifier.trigger_note_off()
                            envelope_released = True
                            logger.debug("Triggered note off via modifier")
                            break

                if not envelope_released:
                    logger.debug("No envelope found to release")

            self.is_playing = False
            logger.debug(f"Note OFF: {note}")

    def process_message(self, message: NoteOnMessage | NoteOffMessage):
        """Process a MIDI message.

        Convenience method for processing MIDI messages directly.

        Args:
            message: NoteOnMessage or NoteOffMessage

        Example:
            >>> msg = NoteOnMessage(timestamp=0.0, note=60, velocity=100)
            >>> synth.process_message(msg)
        """
        if isinstance(message, NoteOnMessage):
            self.note_on(message.note, message.velocity)
        elif isinstance(message, NoteOffMessage):
            self.note_off(message.note)
        else:
            logger.warning(f"Unsupported message type: {type(message)}")

    def get_samples(self, num_samples: int) -> np.ndarray:
        """Generate audio samples.

        Args:
            num_samples: Number of samples to generate

        Returns:
            Array of audio samples (mono or stereo)

        Example:
            >>> samples = synth.get_samples(1024)
            >>> print(samples.shape)
            (1024,) or (1024, 2) for stereo
        """
        if self.voice is None:
            # Return silence if no voice
            return np.zeros(num_samples, dtype=np.float32)

        # Generate samples from voice
        samples = self.voice.get_samples(num_samples)

        # Check if envelope has ended (release complete) and note is not playing
        if not self.is_playing:
            # Check if the voice/envelope has ended
            ended = False

            # Check voice.ended
            if hasattr(self.voice, "ended") and self.voice.ended:
                ended = True
            # Check modifiers for ended state (ModulatedVolume)
            elif hasattr(self.voice, "modifiers"):
                for modifier in self.voice.modifiers:
                    if hasattr(modifier, "modulator"):
                        if (
                            hasattr(modifier.modulator, "ended")
                            and modifier.modulator.ended
                        ):
                            ended = True
                            break
                    elif hasattr(modifier, "ended") and modifier.ended:
                        ended = True
                        break

            # If ended, clear voice and return silence
            if ended:
                self.voice = None
                self.current_note = None
                logger.debug("Voice ended, clearing")
                return np.zeros(num_samples, dtype=np.float32)

        return samples

    def reset(self):
        """Reset the synthesizer state.

        Stops current note and clears the voice.
        """
        self.voice = None
        self.current_note = None
        self.current_velocity = 0
        self.is_playing = False
        logger.debug("MonophonicSynth reset")

    def __repr__(self) -> str:
        """String representation."""
        if self.is_playing:
            return (
                f"MonophonicSynth(note={self.current_note}, "
                f"velocity={self.current_velocity})"
            )
        else:
            return "MonophonicSynth(idle)"
