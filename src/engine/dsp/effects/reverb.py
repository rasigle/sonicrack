"""Reverb effect for audio processing."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from src.constants import DEFAULT_SAMPLE_RATE
from src.engine.core.component import (
    ComponentCategory,
    ComponentDescriptor,
    ParameterDescriptor,
)
from src.engine.core.parameter import AutomationMode, RuntimeParameter, SmoothingPolicy
from src.engine.core.registry import register_component
from src.engine.core.sample_mode import VALID_SAMPLE_MODES, SampleMode
from src.engine.dsp.modifiers.base import Modifier
from src.engine.utils.validation import (
    validate_sample_count,
    validate_sample_rate,
)


@register_component()
class Reverb(Modifier):
    """Simple reverb effect using multiple comb and allpass filters.

    Creates a sense of space by simulating sound reflections in a room.
    Uses a simplified Freeverb-style algorithm with comb filters and allpass filters.

    Can be used in two ways:
    1. In Chain as Modifier: Chain(osc, Reverb(room_size=0.6, damping=0.5))
    2. Wrapping a source: Reverb(source=osc, room_size=0.6, damping=0.5)

    Args:
        source: Optional input audio source (for wrapping usage). Default: None
        room_size: Room size (0.0 to 1.0). Default: 0.5
        damping: High frequency damping (0.0 to 1.0). Default: 0.5
        mix: Dry/wet mix (0.0 to 1.0). Default: 0.3
        sample_rate: Sample rate in Hz. Default: DEFAULT_SAMPLE_RATE
    """

    descriptor = ComponentDescriptor(
        name="Reverb",
        category=ComponentCategory.EFFECT,
        description="Reverb effect using comb and allpass filters",
        tags=["effect", "reverb", "space", "room"],
        parameters={
            "room_size": ParameterDescriptor(
                name="room_size",
                default=0.5,
                minimum=0.0,
                maximum=1.0,
                description="Room size control.",
                smoothing_policy=SmoothingPolicy.LINEAR,
                smoothing_duration_ms=50.0,
                automation_mode=AutomationMode.CONTROL_RATE,
            ),
            "damping": ParameterDescriptor(
                name="damping",
                default=0.5,
                minimum=0.0,
                maximum=1.0,
                description="High-frequency damping amount.",
                smoothing_policy=SmoothingPolicy.LINEAR,
                smoothing_duration_ms=50.0,
                automation_mode=AutomationMode.CONTROL_RATE,
            ),
            "mix": ParameterDescriptor(
                name="mix",
                default=0.3,
                minimum=0.0,
                maximum=1.0,
                description="Dry/wet mix.",
                smoothing_policy=SmoothingPolicy.LINEAR,
                smoothing_duration_ms=20.0,
                automation_mode=AutomationMode.CONTROL_RATE,
            ),
            "sample_rate": ParameterDescriptor(
                name="sample_rate",
                default=DEFAULT_SAMPLE_RATE,
                minimum=1.0,
                unit="Hz",
                description="Processing sample rate.",
                smoothing_policy=SmoothingPolicy.NONE,
                automation_mode=AutomationMode.NONE,
            ),
        },
        fluent_api_name="reverb",
    )

    def __init__(
        self,
        source: Any | None = None,
        room_size: float = 0.5,
        damping: float = 0.5,
        mix: float = 0.3,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize reverb effect.

        Args:
            source: Optional input audio source
            room_size: Room size (0.0-1.0)
            damping: High frequency damping (0.0-1.0)
            mix: Dry/wet mix (0.0-1.0)
            sample_rate: Sample rate
        """
        super().__init__(*args, **kwargs)
        self.source = source  # Optional!
        self._sample_rate = validate_sample_rate(sample_rate)

        # Create RuntimeParameters for smoothed parameters with custom initial values
        room_size_descriptor = ParameterDescriptor(
            name="room_size",
            default=room_size,  # Use passed value as default
            minimum=0.0,
            maximum=1.0,
            smoothing_policy=SmoothingPolicy.LINEAR,
            smoothing_duration_ms=50.0,
            automation_mode=AutomationMode.CONTROL_RATE,
        )
        self._room_size_param = RuntimeParameter(
            room_size_descriptor, self._sample_rate
        )

        damping_descriptor = ParameterDescriptor(
            name="damping",
            default=damping,  # Use passed value as default
            minimum=0.0,
            maximum=1.0,
            smoothing_policy=SmoothingPolicy.LINEAR,
            smoothing_duration_ms=50.0,
            automation_mode=AutomationMode.CONTROL_RATE,
        )
        self._damping_param = RuntimeParameter(damping_descriptor, self._sample_rate)

        mix_descriptor = ParameterDescriptor(
            name="mix",
            default=mix,  # Use passed value as default
            minimum=0.0,
            maximum=1.0,
            smoothing_policy=SmoothingPolicy.LINEAR,
            smoothing_duration_ms=20.0,
            automation_mode=AutomationMode.CONTROL_RATE,
        )
        self._mix_param = RuntimeParameter(mix_descriptor, self._sample_rate)

        # Freeverb-inspired delay line lengths (in samples at 44.1kHz)
        # Scaled to current sample rate
        scale = self._sample_rate / 44100.0

        # Comb filter delay lengths (prime numbers for density)
        self._comb_delays = [
            int(1557 * scale),
            int(1617 * scale),
            int(1491 * scale),
            int(1422 * scale),
            int(1277 * scale),
            int(1356 * scale),
            int(1188 * scale),
            int(1116 * scale),
        ]

        # Allpass filter delay lengths
        self._allpass_delays = [
            int(225 * scale),
            int(556 * scale),
            int(441 * scale),
            int(341 * scale),
        ]

        # Create buffers for comb filters
        self._comb_buffers = [
            np.zeros(delay, dtype=np.float32) for delay in self._comb_delays
        ]
        self._comb_positions = [0] * len(self._comb_delays)
        self._comb_filter_states = [0.0] * len(self._comb_delays)
        self._comb_count = len(self._comb_buffers)
        self._comb_buffer_lengths = [len(buffer) for buffer in self._comb_buffers]

        # Create buffers for allpass filters
        self._allpass_buffers = [
            np.zeros(delay, dtype=np.float32) for delay in self._allpass_delays
        ]
        self._allpass_positions = [0] * len(self._allpass_delays)
        self._allpass_count = len(self._allpass_buffers)
        self._allpass_buffer_lengths = [len(buffer) for buffer in self._allpass_buffers]
        self._sample_shape: tuple[int, ...] = ()

    def _ensure_buffer_shape(self, sample_shape: tuple[int, ...]) -> None:
        """Resize reverb memory for mono or multi-channel frames."""
        if sample_shape == self._sample_shape:
            return

        self._sample_shape = sample_shape
        self._comb_buffers = [
            np.zeros((delay, *sample_shape), dtype=np.float32)
            for delay in self._comb_delays
        ]
        self._comb_positions = [0] * len(self._comb_delays)
        self._comb_filter_states = [
            np.zeros(sample_shape, dtype=np.float32) for _ in self._comb_delays
        ]
        self._comb_buffer_lengths = [len(buffer) for buffer in self._comb_buffers]

        self._allpass_buffers = [
            np.zeros((delay, *sample_shape), dtype=np.float32)
            for delay in self._allpass_delays
        ]
        self._allpass_positions = [0] * len(self._allpass_delays)
        self._allpass_buffer_lengths = [len(buffer) for buffer in self._allpass_buffers]

    @property
    def room_size(self) -> float:
        """float: Room size (0.0-1.0)."""
        return self._room_size_param._target_value

    @room_size.setter
    def room_size(self, value: float):
        """Set room size."""
        self._room_size_param.value = value

    @property
    def damping(self) -> float:
        """float: Damping amount (0.0-1.0)."""
        return self._damping_param._target_value

    @damping.setter
    def damping(self, value: float):
        """Set damping amount."""
        self._damping_param.value = value

    @property
    def mix(self) -> float:
        """float: Dry/wet mix (0.0-1.0)."""
        return self._mix_param._target_value

    @mix.setter
    def mix(self, value: float):
        """Set mix amount."""
        self._mix_param.value = value

    def _process_comb(self, input_val: float, index: int) -> float:
        """Process one comb filter.

        Args:
            input_val: Input sample
            index: Comb filter index

        Returns:
            Filtered sample
        """
        buffer = self._comb_buffers[index]
        pos = self._comb_positions[index]

        # Read from delay line
        output = buffer[pos]

        # Apply damping (one-pole lowpass)
        filtered = output * self._damp2 + self._comb_filter_states[index] * self._damp1
        self._comb_filter_states[index] = filtered

        # Write input + feedback to buffer
        buffer[pos] = input_val + filtered * self._feedback

        # Update position
        self._comb_positions[index] = (pos + 1) % self._comb_buffer_lengths[index]

        return output

    def _process_allpass(self, input_val: float, index: int) -> float:
        """Process one allpass filter.

        Args:
            input_val: Input sample
            index: Allpass filter index

        Returns:
            Filtered sample
        """
        buffer = self._allpass_buffers[index]
        pos = self._allpass_positions[index]

        # Read from delay line
        delayed = buffer[pos]

        # Allpass calculation
        output = -input_val + delayed
        buffer[pos] = input_val + delayed * 0.5

        # Update position
        self._allpass_positions[index] = (pos + 1) % self._allpass_buffer_lengths[index]

        return output

    def _process_sample(self, input_sample: float) -> float:
        """Process one mono sample through all reverb delay lines."""
        input_value = np.asarray(input_sample, dtype=np.float32)
        self._ensure_buffer_shape(input_value.shape)

        # Use current parameter values for single sample processing
        room_size = self._room_size_param.value
        damping = self._damping_param.value
        mix = self._mix_param.value

        feedback = 0.84 + room_size * 0.14
        damp1 = damping * 0.4
        damp2 = 1.0 - damp1

        comb_sum = np.zeros(input_value.shape, dtype=np.float32)
        for index in range(self._comb_count):
            buffer = self._comb_buffers[index]
            pos = self._comb_positions[index]

            output = buffer[pos]
            filtered = output * damp2 + self._comb_filter_states[index] * damp1
            self._comb_filter_states[index] = filtered
            buffer[pos] = input_value + filtered * feedback
            self._comb_positions[index] = (pos + 1) % self._comb_buffer_lengths[index]

            comb_sum += output

        wet = comb_sum / self._comb_count

        for index in range(self._allpass_count):
            buffer = self._allpass_buffers[index]
            pos = self._allpass_positions[index]

            delayed = buffer[pos]
            output = -wet + delayed
            buffer[pos] = wet + delayed * 0.5
            self._allpass_positions[index] = (pos + 1) % self._allpass_buffer_lengths[
                index
            ]
            wet = output

        output_value = input_value * (1.0 - mix) + wet * mix
        if output_value.shape == ():
            return float(output_value)
        return output_value.astype(np.float32, copy=False)

    def _process_buffer(self, input_samples: np.ndarray) -> np.ndarray:
        """Process a mono or multi-channel sample buffer through the reverb."""
        input_samples = np.asarray(input_samples, dtype=np.float32)
        if input_samples.size == 0:
            return input_samples.copy()

        self._ensure_buffer_shape(input_samples.shape[1:])
        output_samples = np.empty_like(input_samples, dtype=np.float32)

        # Get smoothed parameter envelopes
        num_samples = len(input_samples)
        room_size_envelope = self._room_size_param.get_interpolated_buffer(num_samples)
        damping_envelope = self._damping_param.get_interpolated_buffer(num_samples)
        mix_envelope = self._mix_param.get_interpolated_buffer(num_samples)

        comb_buffers = self._comb_buffers
        comb_positions = self._comb_positions
        comb_states = self._comb_filter_states
        comb_lengths = self._comb_buffer_lengths
        comb_count = self._comb_count

        allpass_buffers = self._allpass_buffers
        allpass_positions = self._allpass_positions
        allpass_lengths = self._allpass_buffer_lengths
        allpass_count = self._allpass_count

        for sample_index, input_sample in enumerate(input_samples):
            # Update coefficients per sample for smooth transitions
            room_size = room_size_envelope[sample_index]
            damping = damping_envelope[sample_index]
            mix = mix_envelope[sample_index]

            feedback = 0.84 + room_size * 0.14
            damp1 = damping * 0.4
            damp2 = 1.0 - damp1
            dry_mix = 1.0 - mix

            input_value = input_sample
            comb_sum = np.zeros(input_samples.shape[1:], dtype=np.float32)

            for index in range(comb_count):
                buffer = comb_buffers[index]
                pos = comb_positions[index]

                output = buffer[pos]
                filtered = output * damp2 + comb_states[index] * damp1
                comb_states[index] = filtered
                buffer[pos] = input_value + filtered * feedback

                pos += 1
                if pos == comb_lengths[index]:
                    pos = 0
                comb_positions[index] = pos

                comb_sum += output

            wet = comb_sum / comb_count

            for index in range(allpass_count):
                buffer = allpass_buffers[index]
                pos = allpass_positions[index]

                delayed = buffer[pos]
                output = -wet + delayed
                buffer[pos] = wet + delayed * 0.5

                pos += 1
                if pos == allpass_lengths[index]:
                    pos = 0
                allpass_positions[index] = pos
                wet = output

            output_samples[sample_index] = input_value * dry_mix + wet * mix

        return output_samples

    def __call__(
        self, val: float | tuple[float, ...] | np.ndarray
    ) -> float | tuple[float, ...] | np.ndarray:
        """Apply reverb to value(s) - Modifier interface.

        Args:
            val: Input value (scalar or array)

        Returns:
            Reverbed value (same type as input)
        """
        # Handle scalar
        if isinstance(val, (float, int, np.number)):
            return self._process_sample(float(val))

        if isinstance(val, tuple):
            result = self._process_sample(np.asarray(val, dtype=np.float32))
            return tuple(float(sample) for sample in np.asarray(result))

        input_samples = np.asarray(val)
        return self._process_buffer(input_samples)

    def __iter__(self):
        """Initialize iterator."""
        if self.source is not None:
            source = cast(Any, self.source)
            iter(source)
        return self

    def __next__(self) -> float:
        """Get next sample with reverb applied."""
        if self.source is None:
            raise ValueError("source is required for iterator usage")

        source = cast(Any, self.source)
        return self(next(source))

    def get_samples_vectorized(self, n: int) -> np.ndarray:
        """Generate n reverb samples.

        Note: Due to the sequential nature of the filters, this processes
        samples one at a time but returns them as a vectorized array.

        Args:
            n: Number of samples to generate

        Returns:
            Array of reverb samples
        """
        n = validate_sample_count(n)
        if self.source is None:
            raise ValueError("source is required for get_samples_vectorized()")

        # Get input samples
        source = cast(Any, self.source)
        input_samples = source.get_samples_vectorized(n)
        return self._process_buffer(input_samples)

    def get_samples(
        self, n: int, mode: SampleMode = "vectorized", **kwargs
    ) -> np.ndarray:
        """Get n samples using specified mode."""
        if mode not in VALID_SAMPLE_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'auto', 'iterator', or 'vectorized'."
            )
        if mode == "auto":
            mode = "vectorized" if n >= 512 else "iterator"
        if mode == "vectorized":
            return self.get_samples_vectorized(n)

        n = validate_sample_count(n)
        return np.array([next(self) for _ in range(n)], dtype=np.float32)

    # Backward-compatible properties for internal state access
    @property
    def _room_size(self) -> float:
        """Backward compatibility: current room size."""
        return self._room_size_param.value

    @property
    def _damping(self) -> float:
        """Backward compatibility: current damping."""
        return self._damping_param.value

    @property
    def _mix(self) -> float:
        """Backward compatibility: current mix."""
        return self._mix_param.value

    @property
    def _feedback(self) -> float:
        """Backward compatibility: current feedback coefficient."""
        room_size = self._room_size_param.value
        return 0.84 + room_size * 0.14

    @property
    def _damp1(self) -> float:
        """Backward compatibility: damping coefficient 1."""
        damping = self._damping_param.value
        return damping * 0.4

    @property
    def _damp2(self) -> float:
        """Backward compatibility: damping coefficient 2."""
        return 1.0 - self._damp1
