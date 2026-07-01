"""CV (Control Voltage) utilities for modular synth signal scaling.

Provides utilities to ensure CV signals are in the correct range for
different modulated components.
"""

import logging

import numpy as np

from src.engine.core.component import AudioComponent
from src.engine.utils.validation import validate_sample_count

logger = logging.getLogger(__name__)

PITCH_CV_REFERENCE_NOTE = 60
PITCH_CV_REFERENCE_FREQUENCY = 261.6255653005986
"""1V/oct reference: 0V is MIDI note 60, C4."""


def pitch_cv_to_frequency(volts: float | np.ndarray) -> float | np.ndarray:
    """Convert 1V/oct pitch CV to frequency in Hz.

    ``0V`` maps to C4, and each +1V doubles the resulting frequency.
    """
    values = np.asarray(volts, dtype=np.float64)
    frequency = PITCH_CV_REFERENCE_FREQUENCY * np.power(2.0, values)
    if np.isscalar(volts):
        return float(frequency)
    return frequency.astype(np.float32)


def frequency_to_pitch_cv(frequency_hz: float | np.ndarray) -> float | np.ndarray:
    """Convert frequency in Hz to 1V/oct pitch CV."""
    values = np.asarray(frequency_hz, dtype=np.float64)
    cv = np.log2(values / PITCH_CV_REFERENCE_FREQUENCY)
    if np.isscalar(frequency_hz):
        return float(cv)
    return cv.astype(np.float32)


def midi_note_to_pitch_cv(note: int | float | np.ndarray) -> float | np.ndarray:
    """Convert a MIDI note number to 1V/oct pitch CV."""
    values = np.asarray(note, dtype=np.float64)
    cv = (values - PITCH_CV_REFERENCE_NOTE) / 12.0
    if np.isscalar(note):
        return float(cv)
    return cv.astype(np.float32)


class CVScaler:
    """Scales CV signals from one range to another.

    This is essential for modular synthesis where different modules
    expect different CV ranges:
    - Oscillators output [-1, 1] (bipolar)
    - Envelopes output [0, 1] (unipolar)
    - Panners expect [-1, 1] (bipolar)
    - Volume/Clipper expect [0, 1] (unipolar)

    Example:
        >>> from src.engine import SineOscillator, ModulatedVolume
        >>> # Scale oscillator [-1, 1] to volume [0, 1]
        >>> lfo = SineOscillator(2, amplitude=1.0)
        >>> scaler = CVScaler(lfo, input_range=(-1, 1), output_range=(0, 1))
        >>> volume = ModulatedVolume(scaler)
    """

    def __init__(
        self,
        source: AudioComponent,
        input_range: tuple[float, float] = (-1.0, 1.0),
        output_range: tuple[float, float] = (0.0, 1.0),
        clamp: bool = True,
    ):
        """Initialize CV scaler.

        Args:
            source: Source component that provides CV values
            input_range: Expected input range as (min, max)
            output_range: Desired output range as (min, max)
            clamp: If True, clamp output to output_range (default True)

        Raises:
            TypeError: If source doesn't implement iterator protocol
            ValueError: If ranges are invalid
        """
        if not (hasattr(source, "__iter__") and hasattr(source, "__next__")):
            raise TypeError(
                f"source must implement iterator protocol, got {type(source).__name__}"
            )

        # Validate ranges
        if input_range[0] >= input_range[1]:
            raise ValueError(
                f"input_range min ({input_range[0]}) must be < max ({input_range[1]})"
            )
        if output_range[0] >= output_range[1]:
            raise ValueError(
                f"output_range min ({output_range[0]}) must be < max "
                f"({output_range[1]})"
            )

        self._source = source
        self._iterator = iter(source)

        # Calculate scaling parameters
        self._in_min, self._in_max = input_range
        self._out_min, self._out_max = output_range

        # Pre-calculate scaling factors for efficiency
        self._in_range = self._in_max - self._in_min
        self._out_range = self._out_max - self._out_min
        self._scale = self._out_range / self._in_range
        self._offset = self._out_min - (self._in_min * self._scale)

        self._clamp = clamp

        logger.debug(
            f"CVScaler: {input_range} → {output_range}, "
            f"scale={self._scale:.3f}, offset={self._offset:.3f}"
        )

    def __iter__(self):
        """Reset iterator."""
        self._iterator = iter(self._source)
        return self

    def __next__(self):
        """Get next scaled value."""
        value = next(self._iterator)
        scaled = value * self._scale + self._offset

        if self._clamp:
            return np.clip(scaled, self._out_min, self._out_max)
        return scaled

    def get_samples(self, n: int, **kwargs) -> np.ndarray:
        """Get n scaled samples (vectorized).

        Args:
            n: Number of samples
            **kwargs: Passed to source get_samples if available

        Returns:
            Array of scaled samples
        """
        n = validate_sample_count(n)

        # Try to get samples from source efficiently
        vectorized_method = getattr(type(self._source), "get_samples_vectorized", None)
        has_vectorized_override = (
            vectorized_method is not None
            and vectorized_method is not AudioComponent.get_samples_vectorized
        )
        if not kwargs and has_vectorized_override:
            samples = self._source.get_samples_vectorized(n)
        elif hasattr(self._source, "get_samples"):
            samples = self._source.get_samples(n, **kwargs)
        else:
            # Fallback to iterator
            samples = np.array([next(self._iterator) for _ in range(n)])

        # Scale
        scaled = samples * self._scale + self._offset

        if self._clamp:
            return np.clip(scaled, self._out_min, self._out_max).astype(np.float32)
        return scaled.astype(np.float32)

    def trigger_release(self):
        """Forward trigger_release to source if supported."""
        if hasattr(self._source, "trigger_release"):
            self._source.trigger_release()

    @property
    def ended(self):
        """Check if source has ended."""
        if hasattr(self._source, "ended"):
            return self._source.ended
        return False


# Convenience functions for common CV scaling scenarios


def bipolar_to_unipolar(source: AudioComponent, clamp: bool = True) -> CVScaler:
    """Scale bipolar [-1, 1] to unipolar [0, 1].

    Common use case: LFO modulating volume or clipper threshold.

    Args:
        source: Bipolar CV source (e.g., oscillator)
        clamp: Clamp output to [0, 1]

    Returns:
        CVScaler instance

    Example:
        >>> from src.engine import SineOscillator, ModulatedVolume
        >>> lfo = SineOscillator(2)  # Output: [-1, 1]
        >>> scaled = bipolar_to_unipolar(lfo)  # Output: [0, 1]
        >>> volume = ModulatedVolume(scaled)
    """
    return CVScaler(
        source, input_range=(-1.0, 1.0), output_range=(0.0, 1.0), clamp=clamp
    )


def unipolar_to_bipolar(source: AudioComponent, clamp: bool = True) -> CVScaler:
    """Scale unipolar [0, 1] to bipolar [-1, 1].

    Common use case: Envelope modulating pan position.

    Args:
        source: Unipolar CV source (e.g., envelope)
        clamp: Clamp output to [-1, 1]

    Returns:
        CVScaler instance

    Example:
        >>> from src.engine import ADSREnvelope, ModulatedPanner
        >>> env = ADSREnvelope(attack=0.1, decay=0.2, sustain=0.7, release=0.3)
        >>> scaled = unipolar_to_bipolar(env)  # Output: [-1, 1]
        >>> panner = ModulatedPanner(scaled)
    """
    return CVScaler(
        source, input_range=(0.0, 1.0), output_range=(-1.0, 1.0), clamp=clamp
    )


def scale_cv(
    source: AudioComponent,
    from_range: tuple[float, float],
    to_range: tuple[float, float],
    clamp: bool = True,
) -> CVScaler:
    """General purpose CV scaling.

    Args:
        source: CV source
        from_range: Input range as (min, max)
        to_range: Output range as (min, max)
        clamp: Clamp output to to_range

    Returns:
        CVScaler instance

    Example:
        >>> from src.engine import SineOscillator, ModulatedVolume
        >>> lfo = SineOscillator(2, amplitude=0.5)  # Output: [-0.5, 0.5]
        >>> scaled = scale_cv(lfo, from_range=(-0.5, 0.5), to_range=(0, 1))
        >>> volume = ModulatedVolume(scaled)
    """
    return CVScaler(source, input_range=from_range, output_range=to_range, clamp=clamp)
