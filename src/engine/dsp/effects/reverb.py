"""Reverb effect for audio processing."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

try:
    import numba

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

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


# Numba-JIT compiled reverb processing for mono buffers
if HAS_NUMBA:

    @numba.jit(nopython=True, cache=True, fastmath=True, nogil=True)
    def _process_reverb_numba(
        input_samples: np.ndarray,
        comb_buffers: list,
        comb_positions: np.ndarray,
        comb_states: np.ndarray,
        comb_lengths: np.ndarray,
        allpass_buffers: list,
        allpass_positions: np.ndarray,
        allpass_lengths: np.ndarray,
        room_size_env: np.ndarray,
        damping_env: np.ndarray,
        mix_env: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Numba-JIT compiled mono reverb processing.

        Args:
            input_samples: Mono input audio buffer, shape (n,)
            comb_buffers: List of mono comb filter delay buffers
            comb_positions: Current positions in comb buffers, shape (8,)
            comb_states: Scalar damping states for comb filters, shape (8,)
            comb_lengths: Lengths of comb filter buffers, shape (8,)
            allpass_buffers: List of mono allpass filter delay buffers
            allpass_positions: Current positions in allpass buffers, shape (4,)
            allpass_lengths: Lengths of allpass filter buffers, shape (4,)
            room_size_env: Room size envelope, shape (n,)
            damping_env: Damping envelope, shape (n,)
            mix_env: Mix envelope, shape (n,)

        Returns:
            Tuple of:
                output_samples,
                updated comb positions,
                updated comb states,
                updated allpass positions.
        """
        output_samples = np.empty(len(input_samples), dtype=np.float32)

        feedback_min = np.float32(_FEEDBACK_MIN)
        feedback_range = np.float32(_FEEDBACK_RANGE)
        damping_scale = np.float32(_DAMPING_SCALE)
        one = np.float32(1.0)
        half = np.float32(0.5)
        inv_comb_count = np.float32(1.0 / _COMB_COUNT)

        for sample_idx in range(len(input_samples)):
            input_value = input_samples[sample_idx]

            # Calculate coefficients for this sample.
            feedback = feedback_min + room_size_env[sample_idx] * feedback_range
            damp1 = damping_env[sample_idx] * damping_scale
            damp2 = one - damp1
            mix = mix_env[sample_idx]
            dry_mix = one - mix

            comb_sum = np.float32(0.0)

            # Process comb filters.
            for i in range(_COMB_COUNT):
                pos = comb_positions[i]
                delayed = comb_buffers[i][pos]

                # Damping filter.
                filtered = delayed * damp2 + comb_states[i] * damp1
                comb_states[i] = filtered

                # Write to buffer.
                comb_buffers[i][pos] = input_value + filtered * feedback

                # Advance position.
                pos += 1
                if pos >= comb_lengths[i]:
                    pos = 0
                comb_positions[i] = pos

                comb_sum += delayed

            wet = comb_sum * inv_comb_count

            # Process allpass filters.
            for i in range(_ALLPASS_COUNT):
                pos = allpass_positions[i]
                delayed = allpass_buffers[i][pos]

                output = -wet + delayed
                allpass_buffers[i][pos] = wet + delayed * half

                pos += 1
                if pos >= allpass_lengths[i]:
                    pos = 0
                allpass_positions[i] = pos

                wet = output

            # Mix dry and wet.
            output_samples[sample_idx] = input_value * dry_mix + wet * mix

        return output_samples, comb_positions, comb_states, allpass_positions


@register_component()
class Reverb(Modifier):
    """Reverb effect using multiple comb and allpass filters.

    Creates a sense of space by simulating sound reflections in a room.
    Uses a simplified Freeverb-style algorithm with comb filters and allpass filters.

    Can be used in two ways:
    1. In Chain as Modifier: Chain(osc, Reverb(room_size=0.6, damping=0.5))
    2. Wrapping a source: Reverb(source=osc, room_size=0.6, damping=0.5)

    Args:
        source: Optional input audio source. Default: None
        room_size: Room size, 0.0 to 1.0. Default: 0.5
        damping: High frequency damping, 0.0 to 1.0. Default: 0.5
        mix: Dry/wet mix, 0.0 to 1.0. Default: 0.3
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
        """Initialize reverb effect."""
        super().__init__(*args, **kwargs)
        self.source = source
        self._sample_rate = validate_sample_rate(sample_rate)

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

        # Freeverb-inspired delay line lengths at 44.1 kHz.
        scale = self._sample_rate / 44100.0

        self._comb_delays = [
            max(1, int(d * scale))
            for d in (1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116)
        ]
        self._allpass_delays = [
            max(1, int(d * scale))
            for d in (225, 556, 441, 341)
        ]

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
        """Zero all delay lines and filter states while preserving buffer shape."""
        for i in range(_COMB_COUNT):
            self._comb_buffers[i].fill(0.0)
            self._comb_positions[i] = 0
            self._comb_filter_states[i].fill(0.0)

        for i in range(_ALLPASS_COUNT):
            self._allpass_buffers[i].fill(0.0)
            self._allpass_positions[i] = 0

    @staticmethod
    def _states_to_mono_array(states: list[np.ndarray]) -> np.ndarray:
        """Convert list of scalar state arrays into a 1-D float32 array.

        The regular Python path stores states as arrays so mono and multichannel
        can share the same code. Numba's mono path needs scalar states in a
        simple 1-D array.
        """
        return np.asarray(
            [float(np.asarray(state)) for state in states],
            dtype=np.float32,
        )

    @staticmethod
    def _mono_array_to_states(states: np.ndarray) -> list[np.ndarray]:
        """Convert a 1-D mono state array back to list-of-arrays representation."""
        return [
            np.asarray(value, dtype=np.float32)
            for value in states
        ]

    # -- public properties ---------------------------------------------------

    @property
    def room_size(self) -> float:
        """float: Room size, 0.0 to 1.0."""
        return self._room_size_param._target_value

    @room_size.setter
    def room_size(self, value: float):
        """Set room size."""
        self._room_size_param.value = value

    @property
    def damping(self) -> float:
        """float: Damping amount, 0.0 to 1.0."""
        return self._damping_param._target_value

    @damping.setter
    def damping(self, value: float):
        """Set damping amount."""
        self._damping_param.value = value

    @property
    def mix(self) -> float:
        """float: Dry/wet mix, 0.0 to 1.0."""
        return self._mix_param._target_value

    @mix.setter
    def mix(self, value: float):
        """Set mix amount."""
        self._mix_param.value = value

    # -- scalar processing ---------------------------------------------------

    def _process_sample(self, input_sample: float) -> float:
        """Process one mono sample."""
        room_size = self._room_size_param.value
        damping = self._damping_param.value
        mix = self._mix_param.value

        feedback = _FEEDBACK_MIN + room_size * _FEEDBACK_RANGE
        damp1 = damping * _DAMPING_SCALE
        damp2 = 1.0 - damp1

        comb_sum = 0.0

        cb = self._comb_buffers
        cp = self._comb_positions
        cs = self._comb_filter_states
        cl = self._comb_buffer_lengths

        for i in range(_COMB_COUNT):
            pos = cp[i]
            delayed = cb[i][pos]

            filtered = delayed * damp2 + cs[i] * damp1
            cs[i] = np.asarray(filtered, dtype=np.float32)

            cb[i][pos] = input_sample + filtered * feedback

            pos += 1
            if pos == cl[i]:
                pos = 0
            cp[i] = pos

            comb_sum += delayed

        wet = comb_sum * (1.0 / _COMB_COUNT)

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

        return float(input_sample * (1.0 - mix) + wet * mix)

    # -- buffer processing ---------------------------------------------------

    def _process_buffer(self, input_samples: np.ndarray) -> np.ndarray:
        """Process a mono or multichannel sample buffer through the reverb."""
        input_samples = np.asarray(input_samples, dtype=np.float32)

        if input_samples.size == 0:
            return input_samples.copy()

        self._ensure_buffer_shape(input_samples.shape[1:])

        num_samples = len(input_samples)
        room_size_env = self._room_size_param.get_interpolated_buffer(num_samples)
        damping_env = self._damping_param.get_interpolated_buffer(num_samples)
        mix_env = self._mix_param.get_interpolated_buffer(num_samples)

        # Fast mono path.
        #
        # The ordinary state representation is list[np.ndarray]. For mono,
        # each state is a 0-D array. Numba cannot assign a float into a
        # reflected list of 0-D arrays, so we convert the state to a simple
        # 1-D float32 array before calling the JIT function.
        if HAS_NUMBA and input_samples.ndim == 1:
            comb_positions = np.asarray(self._comb_positions, dtype=np.int64)
            comb_lengths = np.asarray(self._comb_buffer_lengths, dtype=np.int64)
            allpass_positions = np.asarray(self._allpass_positions, dtype=np.int64)
            allpass_lengths = np.asarray(self._allpass_buffer_lengths, dtype=np.int64)
            comb_states = self._states_to_mono_array(self._comb_filter_states)

            output_samples, new_comb_pos, new_comb_states, new_allpass_pos = (
                _process_reverb_numba(
                    input_samples,
                    self._comb_buffers,
                    comb_positions,
                    comb_states,
                    comb_lengths,
                    self._allpass_buffers,
                    allpass_positions,
                    allpass_lengths,
                    np.asarray(room_size_env, dtype=np.float32),
                    np.asarray(damping_env, dtype=np.float32),
                    np.asarray(mix_env, dtype=np.float32),
                )
            )

            self._comb_positions = new_comb_pos.astype(np.int64).tolist()
            self._comb_filter_states = self._mono_array_to_states(new_comb_states)
            self._allpass_positions = new_allpass_pos.astype(np.int64).tolist()

            return output_samples

        # Fallback Python implementation.
        output_samples = np.empty_like(input_samples, dtype=np.float32)

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
                delayed = cb[i][pos]

                filtered = delayed * damp2 + cs[i] * damp1
                cs[i] = np.asarray(filtered, dtype=np.float32)

                cb[i][pos] = input_value + filtered * feedback

                pos += 1
                if pos == cl[i]:
                    pos = 0
                cp[i] = pos

                comb_sum += delayed

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
        """Apply reverb to value(s)."""
        if isinstance(val, (float, int, np.number)):
            return self._process_sample(float(val))

        if isinstance(val, tuple):
            arr = np.asarray(val, dtype=np.float32)
            result = self._process_buffer(arr.reshape(len(arr), 1))
            return tuple(float(x) for x in result.ravel())

        return self._process_buffer(np.asarray(val, dtype=np.float32))

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
        """Generate n reverb samples."""
        n = validate_sample_count(n)

        if self.source is None:
            raise ValueError("source is required for get_samples_vectorized()")

        source = cast(Any, self.source)
        return self._process_buffer(source.get_samples_vectorized(n))

    def get_samples(
        self,
        n: int,
        mode: SampleMode = "vectorized",
        **kwargs: Any,
    ) -> np.ndarray:
        """Get n samples using the specified mode."""
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
