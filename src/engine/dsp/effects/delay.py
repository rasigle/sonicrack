"""Delay effect for audio processing."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from src.constants import DEFAULT_SAMPLE_RATE
from src.engine.core.component import (
    ComponentCategory,
    ComponentDescriptor,
    ParameterDescriptor,
)
from src.engine.core.registry import register_component
from src.engine.core.sample_mode import VALID_SAMPLE_MODES, SampleMode
from src.engine.dsp.modifiers.base import Modifier
from src.engine.utils.validation import (
    validate_numeric_range,
    validate_sample_count,
    validate_sample_rate,
)


@register_component()
class Delay(Modifier):
    """Digital delay effect with feedback.

    Creates echo/delay effects by storing samples in a delay buffer and mixing
    them back with configurable delay time and feedback amount.

    Can be used in two ways:
    1. In Chain as Modifier: Chain(osc, Delay(delay_time=0.3, feedback=0.4))
    2. Wrapping a source: Delay(source=osc, delay_time=0.3, feedback=0.4)

    Args:
        source: Optional input audio source (for wrapping usage). Default: None
        delay_time: Delay time in seconds (0.001 to 2.0). Default: 0.5
        feedback: Feedback amount (0.0 to 0.95). Default: 0.3
        mix: Dry/wet mix (0.0 to 1.0). Default: 0.5
        sample_rate: Sample rate in Hz. Default: DEFAULT_SAMPLE_RATE
    """

    descriptor = ComponentDescriptor(
        name="Delay",
        category=ComponentCategory.EFFECT,
        description="Digital delay effect with feedback",
        tags=["effect", "delay", "echo"],
        parameters={
            "delay_time": ParameterDescriptor(
                name="delay_time",
                default=0.5,
                minimum=0.001,
                maximum=2.0,
                unit="s",
                description="Delay time.",
            ),
            "feedback": ParameterDescriptor(
                name="feedback",
                default=0.3,
                minimum=0.0,
                maximum=0.95,
                description="Delay feedback amount.",
            ),
            "mix": ParameterDescriptor(
                name="mix",
                default=0.5,
                minimum=0.0,
                maximum=1.0,
                description="Dry/wet mix.",
            ),
            "sample_rate": ParameterDescriptor(
                name="sample_rate",
                default=DEFAULT_SAMPLE_RATE,
                minimum=1.0,
                unit="Hz",
                description="Processing sample rate.",
            ),
        },
        fluent_api_name="delay",
    )

    _prev_feedback: float

    def __init__(
        self,
        source: Any | None = None,
        delay_time: float = 0.5,
        feedback: float = 0.3,
        mix: float = 0.5,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize delay effect.

        Args:
            source: Optional input audio source
            delay_time: Delay time in seconds
            feedback: Feedback amount (0.0-0.95)
            mix: Dry/wet mix (0.0-1.0)
            sample_rate: Sample rate

        Raises:
            ValueError: If parameters are out of valid range
        """
        super().__init__(*args, **kwargs)
        self.source = source  # Optional!
        self._sample_rate = validate_sample_rate(sample_rate)
        self._delay_time = validate_numeric_range(
            delay_time, 0.001, 2.0, name="delay_time"
        )
        self._feedback = validate_numeric_range(feedback, 0.0, 0.95, name="feedback")
        self._mix = validate_numeric_range(mix, 0.0, 1.0, name="mix")

        # Create delay buffer (circular buffer)
        max_delay_samples = int(2.0 * self._sample_rate)  # Max 2 seconds
        self._buffer = np.zeros(max_delay_samples, dtype=np.float32)
        self._buffer_size = max_delay_samples
        self._write_pos = 0
        self._sample_shape: tuple[int, ...] = ()

        # Calculate delay in samples
        self._delay_samples = int(self._delay_time * self._sample_rate)
        self._prev_feedback: float = self._feedback

    def reset_buffer(self):
        """Clear the delay buffer to prevent clicks when reusing the delay."""
        self._buffer.fill(0)
        self._write_pos = 0

    def _ensure_buffer_shape(self, sample_shape: tuple[int, ...]) -> None:
        """Resize delay memory for mono or multi-channel frames."""
        if sample_shape == self._sample_shape:
            return
        self._sample_shape = sample_shape
        self._buffer = np.zeros(
            (self._buffer_size, *sample_shape), dtype=np.float32
        )
        self._write_pos = 0

    def _process_frame(self, input_sample: float | np.ndarray) -> float | np.ndarray:
        sample = np.asarray(input_sample, dtype=np.float32)
        self._ensure_buffer_shape(sample.shape)

        read_pos = (self._write_pos - self._delay_samples) % self._buffer_size
        delayed_sample = self._buffer[read_pos].copy()
        self._buffer[self._write_pos] = sample + delayed_sample * self._feedback
        self._write_pos = (self._write_pos + 1) % self._buffer_size

        output = sample * (1.0 - self._mix) + delayed_sample * self._mix
        if output.shape == ():
            return float(output)
        return output.astype(np.float32, copy=False)

    def _process_buffer(self, input_samples: np.ndarray) -> np.ndarray:
        input_samples = np.asarray(input_samples, dtype=np.float32)
        if input_samples.size == 0:
            return input_samples.copy()

        self._ensure_buffer_shape(input_samples.shape[1:])
        output_samples = np.empty_like(input_samples, dtype=np.float32)

        for index, input_sample in enumerate(input_samples):
            read_pos = (self._write_pos - self._delay_samples) % self._buffer_size
            delayed_sample = self._buffer[read_pos].copy()
            output_samples[index] = (
                input_sample * (1.0 - self._mix) + delayed_sample * self._mix
            )
            self._buffer[self._write_pos] = (
                input_sample + delayed_sample * self._feedback
            )
            self._write_pos = (self._write_pos + 1) % self._buffer_size

        return output_samples

    @property
    def delay_time(self) -> float:
        """float: Delay time in seconds."""
        return self._delay_time

    @delay_time.setter
    def delay_time(self, value: float):
        """Set delay time."""
        self._delay_time = validate_numeric_range(value, 0.001, 2.0, name="delay_time")
        new_delay_samples = int(self._delay_time * self._sample_rate)

        # If delay time changed significantly, clear buffer to avoid clicks
        if abs(new_delay_samples - self._delay_samples) > 10:
            self._buffer.fill(0)
            self._write_pos = 0

        self._delay_samples = new_delay_samples

    @property
    def feedback(self) -> float:
        """float: Feedback amount (0.0-0.95)."""
        return self._feedback

    @feedback.setter
    def feedback(self, value: float):
        """Set feedback amount."""
        self._feedback = validate_numeric_range(value, 0.0, 0.95, name="feedback")
        previous_feedback = getattr(self, "_prev_feedback", self._feedback)

        # If feedback is being reduced significantly, optionally reduce buffer content
        # to prevent lingering echoes that might sound like clicks
        if previous_feedback > 0.7 and self._feedback < 0.3:
            # Fade out buffer content to prevent sudden silence
            self._buffer *= 0.5

        self._prev_feedback = self._feedback

    @property
    def mix(self) -> float:
        """float: Dry/wet mix (0.0-1.0)."""
        return self._mix

    @mix.setter
    def mix(self, value: float):
        """Set mix amount."""
        self._mix = validate_numeric_range(value, 0.0, 1.0, name="mix")

    def __call__(
        self, val: float | tuple[float, ...] | np.ndarray
    ) -> float | tuple[float, ...] | np.ndarray:
        """Apply delay to value(s) - Modifier interface.

        Args:
            val: Input value (scalar or array)

        Returns:
            Delayed value (same type as input)
        """
        if isinstance(val, (float, int, np.number)):
            return self._process_frame(float(val))

        if isinstance(val, tuple):
            result = self._process_frame(np.asarray(val, dtype=np.float32))
            return tuple(float(sample) for sample in np.asarray(result))

        return self._process_buffer(np.asarray(val, dtype=np.float32))

    def __iter__(self):
        """Initialize iterator."""
        if self.source is not None:
            source = cast(Any, self.source)
            iter(source)
        return self

    def __next__(self) -> float:
        """Get next sample with delay applied."""
        if self.source is None:
            raise ValueError("source is required for iterator usage")

        source = cast(Any, self.source)
        return self(next(source))

    def get_samples_vectorized(self, n: int) -> np.ndarray:
        """Generate n delayed samples using vectorized processing.

        Args:
            n: Number of samples to generate

        Returns:
            Array of delayed samples
        """
        n = validate_sample_count(n)
        if self.source is None:
            raise ValueError("source is required for get_samples_vectorized()")

        source = cast(Any, self.source)
        return self._process_buffer(source.get_samples_vectorized(n))

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
