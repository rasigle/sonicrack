"""Audio effects: reverb, delay, distortion, and more.

This module provides various audio effects that can be chained into the audio pipeline.
All effects derive from Modifier and can be used both in Chain and as standalone
effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from src.constants import DEFAULT_SAMPLE_RATE
from src.engine.audio_component import (
    AudioComponent,
    ComponentCategory,
    ComponentDescriptor,
    ParameterDescriptor,
)
from src.engine.audio_component_registry import register_component
from src.engine.modifier import Modifier
from src.engine.validation import (
    validate_numeric_range,
    validate_sample_count,
    validate_sample_rate,
)

if TYPE_CHECKING:
    pass


@register_component()
class Distortion(Modifier):
    """Distortion effect with multiple distortion types.

    Applies non-linear waveshaping to the input signal, creating harmonic distortion.
    Supports various distortion algorithms including soft clipping, hard clipping,
    and waveshaping.

    Can be used in two ways:
    1. In Chain as Modifier: Chain(osc, Distortion(drive=2.0))
    2. Wrapping a source: Distortion(source=osc, drive=2.0)

    Args:
        source: Optional input audio source (for wrapping usage). Default: None
        drive: Amount of distortion (0.0 to 10.0). Default: 1.0
        mix: Dry/wet mix (0.0 = dry, 1.0 = wet). Default: 1.0
            output_gain: Output level compensation (0.0 to 2.0). Default: 0.5
        distortion_type: Type of distortion ("soft", "hard", "fuzz", "tube").
            Default: "soft"
    """

    descriptor = ComponentDescriptor(
        name="Distortion",
        category=ComponentCategory.EFFECT,
        description="Distortion effect with multiple distortion types",
        tags=["effect", "distortion", "overdrive", "waveshaper"],
        parameters={
            "drive": ParameterDescriptor(
                name="drive",
                default=1.0,
                minimum=0.0,
                maximum=10.0,
                description="Distortion pre-gain amount.",
            ),
            "mix": ParameterDescriptor(
                name="mix",
                default=1.0,
                minimum=0.0,
                maximum=1.0,
                description="Dry/wet mix.",
            ),
            "output_gain": ParameterDescriptor(
                name="output_gain",
                default=0.5,
                minimum=0.0,
                maximum=2.0,
                description="Post-distortion output gain.",
            ),
            "distortion_type": ParameterDescriptor(
                name="distortion_type",
                default="soft",
                choices=("soft", "hard", "fuzz", "tube"),
                description="Waveshaping algorithm.",
            ),
        },
        fluent_api_name="distortion",
    )

    def __init__(
        self,
        source: AudioComponent | None = None,
        drive: float = 1.0,
        mix: float = 1.0,
        output_gain: float = 0.5,
        distortion_type: str = "soft",
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize distortion effect.

        Args:
            source: Optional input audio source (for wrapping usage)
            drive: Distortion amount (0.0-10.0)
            mix: Dry/wet mix (0.0-1.0)
            output_gain: Output level (0.0-2.0)
            distortion_type: Distortion algorithm type

        Raises:
            ValueError: If parameters are out of valid range
        """
        super().__init__(*args, **kwargs)
        self.source = source  # Optional!
        self._drive = validate_numeric_range(drive, 0.0, 10.0, name="drive")
        self._mix = validate_numeric_range(mix, 0.0, 1.0, name="mix")
        self._output_gain = validate_numeric_range(
            output_gain, 0.0, 2.0, name="output_gain"
        )

        valid_types = ["soft", "hard", "fuzz", "tube"]
        if distortion_type not in valid_types:
            raise ValueError(
                f"distortion_type must be one of {valid_types}, got '{distortion_type}'"
            )
        self._distortion_type = distortion_type

    @property
    def drive(self) -> float:
        """float: Distortion drive amount (0.0-10.0)."""
        return self._drive

    @drive.setter
    def drive(self, value: float):
        """Set drive amount."""
        self._drive = validate_numeric_range(value, 0.0, 10.0, name="drive")

    @property
    def mix(self) -> float:
        """float: Dry/wet mix (0.0-1.0)."""
        return self._mix

    @mix.setter
    def mix(self, value: float):
        """Set mix amount."""
        self._mix = validate_numeric_range(value, 0.0, 1.0, name="mix")

    @property
    def output_gain(self) -> float:
        """float: Output gain (0.0-2.0)."""
        return self._output_gain

    @output_gain.setter
    def output_gain(self, value: float):
        """Set output gain."""
        self._output_gain = validate_numeric_range(
            value, 0.0, 2.0, name="output_gain"
        )

    @property
    def distortion_type(self) -> str:
        """str: Distortion type."""
        return self._distortion_type

    @distortion_type.setter
    def distortion_type(self, value: str):
        """Set distortion type."""
        valid_types = ["soft", "hard", "fuzz", "tube"]
        if value not in valid_types:
            raise ValueError(
                f"distortion_type must be one of {valid_types}, got '{value}'"
            )
        self._distortion_type = value

    def _apply_distortion(self, samples: np.ndarray) -> np.ndarray:
        """Apply selected distortion algorithm to samples.

        Args:
            samples: Input samples

        Returns:
            Distorted samples
        """
        # Pre-amplify based on drive
        amplified = samples * (1.0 + self._drive)

        if self._distortion_type == "soft":
            # Soft clipping using tanh
            distorted = np.tanh(amplified)

        elif self._distortion_type == "hard":
            # Hard clipping
            distorted = np.clip(amplified, -1.0, 1.0)

        elif self._distortion_type == "fuzz":
            # Fuzz using asymmetric clipping
            distorted = np.where(
                amplified > 0,
                np.tanh(amplified * 2.0) * 0.7,
                np.clip(amplified * 1.5, -1.0, 0) * 0.8,
            )

        elif self._distortion_type == "tube":
            # Tube-style distortion (soft knee)
            threshold = 0.3
            distorted = np.where(
                np.abs(amplified) < threshold,
                amplified,
                np.sign(amplified)
                * (
                    threshold
                    + (1.0 - threshold)
                    * np.tanh((np.abs(amplified) - threshold) / (1.0 - threshold))
                ),
            )
        else:
            # Fallback to soft clipping if unknown type
            distorted = np.tanh(amplified)

        return np.asarray(distorted, dtype=np.float32)

    def __call__(self, val: float | np.ndarray) -> float | np.ndarray:
        """Apply distortion to value(s) - Modifier interface.

        This allows Distortion to be used directly in Chain.

        Args:
            val: Input value (scalar or array)

        Returns:
            Distorted value (same type as input)
        """
        # Handle scalar
        if isinstance(val, (float, int, np.number)):
            dry_sample = float(val)
            wet_sample = self._apply_distortion(
                np.array([dry_sample], dtype=np.float32)
            )[0]
            mixed = dry_sample * (1.0 - self._mix) + wet_sample * self._mix
            return mixed * self._output_gain

        # Handle array
        dry_samples = np.asarray(val, dtype=np.float32)
        wet_samples = self._apply_distortion(dry_samples)
        mixed = dry_samples * (1.0 - self._mix) + wet_samples * self._mix
        return (mixed * self._output_gain).astype(np.float32)

    def __iter__(self):
        """Initialize iterator."""
        if self.source is not None:
            source = cast(Any, self.source)
            iter(source)
        return self

    def __next__(self) -> float:
        """Get next sample with distortion applied."""
        if self.source is None:
            raise ValueError("source is required for iterator usage")

        source = cast(Any, self.source)
        dry = next(source)

        # Apply distortion
        wet = self._apply_distortion(np.array([dry]))[0]

        # Mix dry and wet
        mixed = dry * (1.0 - self._mix) + wet * self._mix

        # Apply output gain
        return mixed * self._output_gain

    def get_samples_vectorized(self, n: int) -> np.ndarray:
        """Generate n distorted samples using vectorized processing.

        Args:
            n: Number of samples to generate

        Returns:
            Array of distorted samples

        Raises:
            ValueError: If source is None (required for this method)
        """
        n = validate_sample_count(n)
        if self.source is None:
            raise ValueError("source is required for get_samples_vectorized()")

        # Get input samples
        source = cast(Any, self.source)
        dry = source.get_samples_vectorized(n)

        # Apply distortion
        wet = self._apply_distortion(dry)

        # Mix dry and wet
        mixed = dry * (1.0 - self._mix) + wet * self._mix

        # Apply output gain
        return (mixed * self._output_gain).astype(np.float32)

    def get_samples(self, n: int, mode: str = "vectorized", **kwargs) -> np.ndarray:
        """Get n samples using specified mode."""
        if mode == "vectorized":
            return self.get_samples_vectorized(n)

        n = validate_sample_count(n)
        return np.array([next(self) for _ in range(n)], dtype=np.float32)


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
        source: AudioComponent | None = None,
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
        self._feedback = validate_numeric_range(
            feedback, 0.0, 0.95, name="feedback"
        )
        self._mix = validate_numeric_range(mix, 0.0, 1.0, name="mix")

        # Create delay buffer (circular buffer)
        max_delay_samples = int(2.0 * self._sample_rate)  # Max 2 seconds
        self._buffer = np.zeros(max_delay_samples, dtype=np.float32)
        self._buffer_size = max_delay_samples
        self._write_pos = 0

        # Calculate delay in samples
        self._delay_samples = int(self._delay_time * self._sample_rate)
        self._prev_feedback: float = self._feedback

    def reset_buffer(self):
        """Clear the delay buffer to prevent clicks when reusing the delay."""
        self._buffer.fill(0)
        self._write_pos = 0

    @property
    def delay_time(self) -> float:
        """float: Delay time in seconds."""
        return self._delay_time

    @delay_time.setter
    def delay_time(self, value: float):
        """Set delay time."""
        self._delay_time = validate_numeric_range(
            value, 0.001, 2.0, name="delay_time"
        )
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

    def __call__(self, val: float | np.ndarray) -> float | np.ndarray:
        """Apply delay to value(s) - Modifier interface.

        Args:
            val: Input value (scalar or array)

        Returns:
            Delayed value (same type as input)
        """
        # Handle scalar
        if isinstance(val, (float, int, np.number)):
            input_sample = float(val)
            read_pos = (self._write_pos - self._delay_samples) % self._buffer_size
            delayed_sample = self._buffer[read_pos]
            output_sample = delayed_sample
            feedback_sample = input_sample + delayed_sample * self._feedback
            self._buffer[self._write_pos] = feedback_sample
            self._write_pos = (self._write_pos + 1) % self._buffer_size
            return input_sample * (1.0 - self._mix) + output_sample * self._mix

        # Handle array
        input_samples = np.asarray(val)
        output_samples = np.zeros(len(input_samples), dtype=np.float32)

        for i in range(len(input_samples)):
            read_pos = (self._write_pos - self._delay_samples) % self._buffer_size
            delayed_sample = self._buffer[read_pos]
            output_samples[i] = delayed_sample
            feedback_sample = input_samples[i] + delayed_sample * self._feedback
            self._buffer[self._write_pos] = feedback_sample
            self._write_pos = (self._write_pos + 1) % self._buffer_size

        return (input_samples * (1.0 - self._mix) + output_samples * self._mix).astype(
            np.float32
        )

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

        # Get input sample
        source = cast(Any, self.source)
        input_sample = next(source)

        # Calculate read position (circular buffer)
        read_pos = (self._write_pos - self._delay_samples) % self._buffer_size

        # Read delayed sample
        delayed_sample = self._buffer[read_pos]

        # Create output with feedback
        output_sample = delayed_sample
        feedback_sample = input_sample + delayed_sample * self._feedback

        # Write to buffer
        self._buffer[self._write_pos] = feedback_sample
        self._write_pos = (self._write_pos + 1) % self._buffer_size

        # Mix dry and wet
        return input_sample * (1.0 - self._mix) + output_sample * self._mix

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

        # Get input samples
        source = cast(Any, self.source)
        input_samples = source.get_samples_vectorized(n)
        output_samples = np.zeros(n, dtype=np.float32)

        # Process each sample (delay requires sequential processing for feedback)
        for i in range(n):
            # Calculate read position
            read_pos = (self._write_pos - self._delay_samples) % self._buffer_size

            # Read delayed sample
            delayed_sample = self._buffer[read_pos]

            # Create output with feedback
            output_samples[i] = delayed_sample
            feedback_sample = input_samples[i] + delayed_sample * self._feedback

            # Write to buffer
            self._buffer[self._write_pos] = feedback_sample
            self._write_pos = (self._write_pos + 1) % self._buffer_size

        # Mix dry and wet
        return (input_samples * (1.0 - self._mix) + output_samples * self._mix).astype(
            np.float32
        )

    def get_samples(self, n: int, mode: str = "vectorized", **kwargs) -> np.ndarray:
        """Get n samples using specified mode."""
        if mode == "vectorized":
            return self.get_samples_vectorized(n)

        n = validate_sample_count(n)
        return np.array([next(self) for _ in range(n)], dtype=np.float32)


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
            ),
            "damping": ParameterDescriptor(
                name="damping",
                default=0.5,
                minimum=0.0,
                maximum=1.0,
                description="High-frequency damping amount.",
            ),
            "mix": ParameterDescriptor(
                name="mix",
                default=0.3,
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
        fluent_api_name="reverb",
    )

    def __init__(
        self,
        source: AudioComponent | None = None,
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
        self._room_size = validate_numeric_range(
            room_size, 0.0, 1.0, name="room_size"
        )
        self._damping = validate_numeric_range(damping, 0.0, 1.0, name="damping")
        self._mix = validate_numeric_range(mix, 0.0, 1.0, name="mix")

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

        # Update feedback coefficients
        self._update_coefficients()

    def _update_coefficients(self):
        """Update filter coefficients based on room size and damping."""
        # Room size affects feedback amount
        self._feedback = 0.84 + self._room_size * 0.14

        # Damping coefficients for lowpass in comb filters
        self._damp1 = self._damping * 0.4
        self._damp2 = 1.0 - self._damp1

    @property
    def room_size(self) -> float:
        """float: Room size (0.0-1.0)."""
        return self._room_size

    @room_size.setter
    def room_size(self, value: float):
        """Set room size."""
        self._room_size = validate_numeric_range(value, 0.0, 1.0, name="room_size")
        self._update_coefficients()

    @property
    def damping(self) -> float:
        """float: Damping amount (0.0-1.0)."""
        return self._damping

    @damping.setter
    def damping(self, value: float):
        """Set damping amount."""
        self._damping = validate_numeric_range(value, 0.0, 1.0, name="damping")
        self._update_coefficients()

    @property
    def mix(self) -> float:
        """float: Dry/wet mix (0.0-1.0)."""
        return self._mix

    @mix.setter
    def mix(self, value: float):
        """Set mix amount."""
        self._mix = validate_numeric_range(value, 0.0, 1.0, name="mix")

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
        comb_sum = 0.0
        for index in range(self._comb_count):
            buffer = self._comb_buffers[index]
            pos = self._comb_positions[index]

            output = float(buffer[pos])
            filtered = (
                output * self._damp2 + self._comb_filter_states[index] * self._damp1
            )
            self._comb_filter_states[index] = filtered
            buffer[pos] = input_sample + filtered * self._feedback
            self._comb_positions[index] = (pos + 1) % self._comb_buffer_lengths[index]

            comb_sum += output

        wet = comb_sum / self._comb_count

        for index in range(self._allpass_count):
            buffer = self._allpass_buffers[index]
            pos = self._allpass_positions[index]

            delayed = float(buffer[pos])
            output = -wet + delayed
            buffer[pos] = wet + delayed * 0.5
            self._allpass_positions[index] = (pos + 1) % self._allpass_buffer_lengths[
                index
            ]
            wet = output

        return input_sample * (1.0 - self._mix) + wet * self._mix

    def __call__(self, val: float | np.ndarray) -> float | np.ndarray:
        """Apply reverb to value(s) - Modifier interface.

        Args:
            val: Input value (scalar or array)

        Returns:
            Reverbed value (same type as input)
        """
        # Handle scalar
        if isinstance(val, (float, int, np.number)):
            return self._process_sample(float(val))

        # Handle array
        input_samples = np.asarray(val)
        output_samples = np.zeros(len(input_samples), dtype=np.float32)

        for i in range(len(input_samples)):
            output_samples[i] = self._process_sample(float(input_samples[i]))

        return output_samples

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

        # Get input sample
        source = cast(Any, self.source)
        input_sample = next(source)

        return self._process_sample(float(input_sample))

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
        output_samples = np.zeros(n, dtype=np.float32)

        # Process each sample through the reverb
        for i in range(n):
            output_samples[i] = self._process_sample(float(input_samples[i]))

        return output_samples

    def get_samples(self, n: int, mode: str = "vectorized", **kwargs) -> np.ndarray:
        """Get n samples using specified mode."""
        if mode == "vectorized":
            return self.get_samples_vectorized(n)

        n = validate_sample_count(n)
        return np.array([next(self) for _ in range(n)], dtype=np.float32)
