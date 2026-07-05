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

# Pre-computed constants for the feedback/damping curves
_FEEDBACK_MIN = 0.28
_FEEDBACK_RANGE = 0.70
_DAMPING_SCALE = 0.4
_COMB_COUNT = 8
_ALLPASS_COUNT = 4


@register_component()
class Reverb(Modifier):
    """Reverb effect using multiple comb and allpass filters.

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
        self.source = source
        self._sample_rate = validate_sample_rate(sample_rate)

        # RuntimeParameters with custom initial values
        self._room_size_param = RuntimeParameter(
            ParameterDescriptor(
                name="room_size",
                default=room_size,
                minimum=0.0,
                maximum=1.0,
                smoothing_policy=SmoothingPolicy.LINEAR,
                smoothing_duration_ms=50.0,
                automation_mode=AutomationMode.CONTROL_RATE,
            ),
            self._sample_rate,
        )

        self._damping_param = RuntimeParameter(
            ParameterDescriptor(
                name="damping",
                default=damping,
                minimum=0.0,
                maximum=1.0,
                smoothing_policy=SmoothingPolicy.LINEAR,
                smoothing_duration_ms=50.0,
                automation_mode=AutomationMode.CONTROL_RATE,
            ),
            self._sample_rate,
        )

        self._mix_param = RuntimeParameter(
            ParameterDescriptor(
                name="mix",
                default=mix,
                minimum=0.0,
                maximum=1.0,
                smoothing_policy=SmoothingPolicy.LINEAR,
                smoothing_duration_ms=20.0,
                automation_mode=AutomationMode.CONTROL_RATE,
            ),
            self._sample_rate,
        )

        # Freeverb-inspired delay line lengths (in samples at 44.1kHz)
        scale = self._sample_rate / 44100.0

        self._comb_delays = [
            int(d * scale)
            for d in (1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116)
        ]
        self._allpass_delays = [int(d * scale) for d in (225, 556, 441, 341)]

        self._sample_shape: tuple[int, ...] = ()
        self._comb_buffers: list[np.ndarray] = []
        self._comb_positions: list[int] = []
        self._comb_filter_states: list[np.ndarray] = []
        self._comb_buffer_lengths: list[int] = []
        self._allpass_buffers: list[np.ndarray] = []
        self._allpass_positions: list[int] = []
        self._allpass_buffer_lengths: list[int] = []

        self._init_buffers(())

    def _init_buffers(self, sample_shape: tuple[int, ...]) -> None:
        """Allocate delay line buffers and reset all filter state."""
        self._sample_shape = sample_shape

        self._comb_buffers = [
            np.zeros((delay, *sample_shape), dtype=np.float32)
            for delay in self._comb_delays
        ]
        self._comb_positions = [0] * _COMB_COUNT
        self._comb_filter_states = [
            np.zeros(sample_shape, dtype=np.float32) for _ in self._comb_delays
        ]
        self._comb_buffer_lengths = [len(b) for b in self._comb_buffers]

        self._allpass_buffers = [
            np.zeros((delay, *sample_shape), dtype=np.float32)
            for delay in self._allpass_delays
        ]
        self._allpass_positions = [0] * _ALLPASS_COUNT
        self._allpass_buffer_lengths = [len(b) for b in self._allpass_buffers]

    def _ensure_buffer_shape(self, sample_shape: tuple[int, ...]) -> None:
        """Resize reverb memory only when the audio shape actually changes."""
        if sample_shape != self._sample_shape:
            self._init_buffers(sample_shape)

    def _reset_state(self) -> None:
        """Zero all delay lines and filter states (preserves buffer shape)."""
        for i in range(_COMB_COUNT):
            self._comb_buffers[i].fill(0.0)
            self._comb_positions[i] = 0
            self._comb_filter_states[i].fill(0.0)

        for i in range(_ALLPASS_COUNT):
            self._allpass_buffers[i].fill(0.0)
            self._allpass_positions[i] = 0

    # -- public properties ---------------------------------------------------

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

    # -- scalar processing (zero-allocation iterator path) -------------------

    def _process_sample(self, input_sample: float) -> float:
        """Process one mono sample. Pure-Python, no numpy allocation."""
        # Read current smoothed parameter values
        room_size = self._room_size_param.value
        damping = self._damping_param.value
        mix = self._mix_param.value

        feedback = _FEEDBACK_MIN + room_size * _FEEDBACK_RANGE
        damp1 = damping * _DAMPING_SCALE
        damp2 = 1.0 - damp1

        # Comb filters
        comb_sum = 0.0
        cb = self._comb_buffers
        cp = self._comb_positions
        cs = self._comb_filter_states
        cl = self._comb_buffer_lengths

        for i in range(_COMB_COUNT):
            pos = cp[i]
            output = cb[i][pos]
            filtered = output * damp2 + cs[i] * damp1
            cs[i] = filtered
            cb[i][pos] = input_sample + filtered * feedback

            pos += 1
            if pos == cl[i]:
                pos = 0
            cp[i] = pos

            comb_sum += output

        wet = comb_sum * (1.0 / _COMB_COUNT)

        # Allpass filters
        ap = self._allpass_buffers
        app = self._allpass_positions
        apl = self._allpass_buffer_lengths

        for i in range(_ALLPASS_COUNT):
            pos = app[i]
            delayed = ap[i][pos]
            output = -wet + delayed
            ap[i][pos] = wet + delayed * 0.5

            pos += 1
            if pos == apl[i]:
                pos = 0
            app[i] = pos
            wet = output

        return input_sample * (1.0 - mix) + wet * mix

    # -- buffer processing ---------------------------------------------------

    def _process_buffer(self, input_samples: np.ndarray) -> np.ndarray:
        """Process a mono or multi-channel sample buffer through the reverb."""
        input_samples = np.asarray(input_samples, dtype=np.float32)
        if input_samples.size == 0:
            return input_samples.copy()

        self._ensure_buffer_shape(input_samples.shape[1:])
        output_samples = np.empty_like(input_samples, dtype=np.float32)

        num_samples = len(input_samples)
        room_size_env = self._room_size_param.get_interpolated_buffer(num_samples)
        damping_env = self._damping_param.get_interpolated_buffer(num_samples)
        mix_env = self._mix_param.get_interpolated_buffer(num_samples)

        # Cache locals for inner loop speed
        cb = self._comb_buffers
        cp = self._comb_positions
        cs = self._comb_filter_states
        cl = self._comb_buffer_lengths
        ap = self._allpass_buffers
        app = self._allpass_positions
        apl = self._allpass_buffer_lengths
        sample_shape = input_samples.shape[1:]

        for si in range(num_samples):
            input_value = input_samples[si]
            room_size = room_size_env[si]
            damping = damping_env[si]
            mix = mix_env[si]

            feedback = _FEEDBACK_MIN + room_size * _FEEDBACK_RANGE
            damp1 = damping * _DAMPING_SCALE
            damp2 = 1.0 - damp1
            dry_mix = 1.0 - mix

            comb_sum = np.zeros(sample_shape, dtype=np.float32)

            for i in range(_COMB_COUNT):
                pos = cp[i]
                output = cb[i][pos]
                filtered = output * damp2 + cs[i] * damp1
                cs[i] = filtered
                cb[i][pos] = input_value + filtered * feedback

                pos += 1
                if pos == cl[i]:
                    pos = 0
                cp[i] = pos

                comb_sum += output

            wet = comb_sum * (1.0 / _COMB_COUNT)

            for i in range(_ALLPASS_COUNT):
                pos = app[i]
                delayed = ap[i][pos]
                output = -wet + delayed
                ap[i][pos] = wet + delayed * 0.5

                pos += 1
                if pos == apl[i]:
                    pos = 0
                app[i] = pos
                wet = output

            output_samples[si] = input_value * dry_mix + wet * mix

        return output_samples

    # -- Modifier interface --------------------------------------------------

    def __call__(
        self, val: float | tuple[float, ...] | np.ndarray
    ) -> float | tuple[float, ...] | np.ndarray:
        """Apply reverb to value(s) - Modifier interface.

        Args:
            val: Input value (scalar, stereo tuple, or array)

        Returns:
            Reverbed value (same type as input)
        """
        if isinstance(val, (float, int, np.number)):
            return self._process_sample(float(val))

        if isinstance(val, tuple):
            arr = np.asarray(val, dtype=np.float32)
            result = self._process_buffer(arr.reshape(len(arr), 1))
            return tuple(float(x) for x in result.ravel())

        return self._process_buffer(np.asarray(val))

    # -- Iterator interface --------------------------------------------------

    def __iter__(self):
        """Initialize iterator and reset reverb state."""
        self._reset_state()
        if self.source is not None:
            iter(cast(Any, self.source))
        return self

    def __next__(self) -> float:
        """Get next sample with reverb applied."""
        if self.source is None:
            raise ValueError("source is required for iterator usage")
        return self._process_sample(next(cast(Any, self.source)))

    # -- Vectorized interface ------------------------------------------------

    def get_samples_vectorized(self, n: int) -> np.ndarray:
        """Generate n reverb samples.

        Args:
            n: Number of samples to generate

        Returns:
            Array of reverb samples
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

    # -- Backward-compatible internal state accessors ------------------------

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
