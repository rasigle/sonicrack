"""Digital filters for audio signal processing.

This module provides Butterworth IIR filter design and application utilities
for frequency-domain audio processing. Filters can be used to shape the
spectral content of audio signals (low-pass, high-pass, band-pass).

Functions:
    create_butter_filter: Design Butterworth filter coefficients.
    apply_filter: Apply filter to audio signal using zero-phase filtering.

Example:
    >>> import numpy as np
    >>> from src.constants import DEFAULT_SAMPLE_RATE
    >>>
    >>> # Create a low-pass filter at 1kHz
    >>> b, a = butter(
    ...     order=4,
    ...     cutoff=1000,
    ...     fs=DEFAULT_SAMPLE_RATE,
    ...     btype="low"
    ... )
    >>>
    >>> # Apply filter to signal
    >>> signal = np.random.randn(1000)
    >>> filtered = apply_filter(b, a, signal)

Filter Types:
    - "low": Low-pass filter (attenuates high frequencies)
    - "high": High-pass filter (attenuates low frequencies)
    - "band": Band-pass filter (passes frequencies in a range)

Note:
    This module uses scipy.signal.filtfilt for zero-phase filtering,
    which prevents phase distortion in the filtered signal. The
    create_butter_filter function is named to avoid shadowing
    scipy.signal.butter.
"""

from typing import Literal, cast

import numpy as np
from scipy.signal import butter as scipy_butter
from scipy.signal import filtfilt, lfilter, lfilter_zi

from src.constants import DEFAULT_SAMPLE_RATE
from src.engine.core.component import (
    ComponentCategory,
    ComponentDescriptor,
    ParameterDescriptor,
    make_parameter_descriptors,
)
from src.engine.core.parameter import RuntimeParameter, SmoothingPolicy
from src.engine.core.registry import register_component
from src.engine.dsp.modifiers.base import Modifier
from src.engine.utils.math import db_to_linear
from src.engine.utils.validation import validate_sample_rate


@register_component()
class ButterworthFilter(Modifier):
    """Butterworth IIR filter for frequency-domain audio processing.

    This filter provides low-pass, high-pass, and band-pass filtering with
    optimized performance for real-time audio processing. Uses scipy's IIR
    filter implementation with stateful processing for iterator mode.

    Args:
        cutoff: Cutoff frequency in Hz. For band-pass, use tuple (low, high).
        order: Filter order (higher = steeper rolloff, default: 4).
        filter_type: Filter type - "low", "high", or "band".
        sample_rate: Sample rate in Hz (default: 44100).

    Example:
        >>> # Low-pass filter at 1kHz
        >>> lpf = ButterworthFilter(cutoff=1000, filter_type="low")
        >>> samples = np.random.randn(1000)
        >>> filtered = lpf(samples)
        >>>
        >>> # Band-pass filter 200-2000 Hz
        >>> bpf = ButterworthFilter(cutoff=(200, 2000), filter_type="band")
    """

    descriptor = ComponentDescriptor(
        name="ButterworthFilter",
        category=ComponentCategory.MODIFIER,
        parameters=make_parameter_descriptors(
            "cutoff",
            "order",
            "filter_type",
            "sample_rate",
            cutoff=ParameterDescriptor(
                name="cutoff",
                default=1000.0,
                minimum=0.0,
                unit="Hz",
                description="Cutoff frequency or band-pass frequency range.",
            ),
            order=ParameterDescriptor(
                name="order",
                default=4,
                minimum=1,
                maximum=10,
                description="Butterworth filter order.",
            ),
            filter_type=ParameterDescriptor(
                name="filter_type",
                default="low",
                choices=("low", "high", "band"),
                description="Butterworth filter type.",
            ),
        ),
        description="Butterworth IIR filter (low-pass, high-pass, band-pass)",
        tags=["filter", "frequency", "butterworth", "iir"],
    )

    FilterType = Literal["low", "high", "band"]

    def __init__(
        self,
        cutoff: float | tuple[float, float] = 1000.0,
        order: int = 4,
        filter_type: Literal["low", "high", "band"] = "low",
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ):
        """Initialize Butterworth filter.

        Args:
            cutoff: Cutoff frequency/frequencies in Hz
            order: Filter order (1-10 recommended)
            filter_type: "low", "high", or "band"
            sample_rate: Sample rate in Hz
        """
        super().__init__()

        self.sample_rate = validate_sample_rate(sample_rate)
        self._cutoff = cutoff
        self._order = order
        self._filter_type: ButterworthFilter.FilterType = filter_type

        # Validate filter type
        if filter_type not in ("low", "high", "band"):
            raise ValueError(
                f"filter_type must be 'low', 'high', or 'band', got '{filter_type}'"
            )

        # Validate cutoff for band-pass
        if filter_type == "band":
            if not isinstance(cutoff, (tuple, list)) or len(cutoff) != 2:
                raise ValueError(
                    "For band-pass filter, cutoff must be tuple (low_freq, high_freq)"
                )
            if cutoff[0] >= cutoff[1]:
                raise ValueError(
                    f"Low cutoff ({cutoff[0]}) must be less than high cutoff "
                    f"({cutoff[1]})"
                )

        # Design filter coefficients
        self._b, self._a = self._design_filter()

        # Initialize zero filter state for realtime-style streaming.
        self._zi = np.zeros_like(lfilter_zi(self._b, self._a))
        self._filter_state = self._zi.copy()

    def _design_filter(self) -> tuple[np.ndarray, np.ndarray]:
        """Design Butterworth filter coefficients.

        Returns:
            Tuple of (b, a) filter coefficients
        """
        nyq = 0.5 * self.sample_rate

        wn: float | list[float]
        if isinstance(self._cutoff, (list, tuple)):
            # Band-pass filter
            wn = [c / nyq for c in self._cutoff]
        else:
            # Low-pass or high-pass
            wn = self._cutoff / nyq

        b, a = scipy_butter(
            self._order,
            wn,
            btype=cast(ButterworthFilter.FilterType, self._filter_type),
            analog=False,
        )
        return b, a

    def __iter__(self):
        """Initialize iterator - reset filter state."""
        self.reset_state()
        return self

    def reset_state(self) -> None:
        """Reset the filter delay state."""
        self._filter_state = self._zi.copy()

    def __next__(self):
        """Not used - filter requires buffered processing."""
        raise NotImplementedError(
            "ButterworthFilter requires vectorized processing. "
            "Use scale_vectorized() instead of iterator mode."
        )

    def _filter_scalar(self, val: float) -> float:
        """Apply the filter to a single scalar sample using the current state."""
        filtered, self._filter_state = lfilter(
            self._b, self._a, [val], zi=self._filter_state
        )
        return float(filtered[0])

    def __call__(
        self, val: float | tuple[float, ...] | np.ndarray
    ) -> float | tuple[float, ...] | np.ndarray:
        """Apply filter to single sample or array.

        For arrays, this delegates to scale_vectorized for optimal performance.
        For single samples, it uses stateful filtering.

        Args:
            val: Input sample (mono float, stereo tuple, or numpy array)

        Returns:
            Filtered sample or array (same type as input)
        """
        # Handle numpy arrays (vectorized path)
        if isinstance(val, np.ndarray):
            return self.scale_vectorized(val)

        # Handle stereo tuples
        if isinstance(val, tuple):
            return tuple(self._filter_scalar(v) for v in val)

        # Apply filter with state for single samples
        return self._filter_scalar(val)

    def scale_vectorized(self, samples: np.ndarray) -> np.ndarray:
        """Apply filter to array of samples (optimized vectorized version).

        This is the recommended method for processing audio buffers as it's
        much faster than sample-by-sample processing.

        Args:
            samples: Input samples as numpy array

        Returns:
            Filtered samples (same shape as input)
        """
        if samples.size == 0:
            return samples

        filter_state = self._filter_state_for(samples)
        filtered, self._filter_state = lfilter(
            self._b,
            self._a,
            samples,
            axis=0,
            zi=filter_state,
        )

        return filtered.astype(np.float32)

    def _filter_state_for(self, samples: np.ndarray) -> np.ndarray:
        """Return filter state shaped for filtering along the sample axis."""
        expected_shape = (self._zi.shape[0],) + samples.shape[1:]
        if self._filter_state.shape == expected_shape:
            return self._filter_state

        if samples.ndim == 1:
            self._filter_state = self._zi.copy()
            return self._filter_state

        self._filter_state = np.broadcast_to(
            self._zi.reshape((self._zi.shape[0],) + (1,) * (samples.ndim - 1)),
            expected_shape,
        ).copy()
        return self._filter_state

    @property
    def cutoff(self) -> float | tuple[float, float]:
        """Get cutoff frequency/frequencies."""
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value: float | tuple[float, float]):
        """Set cutoff frequency and redesign filter."""
        self._cutoff = value
        self._b, self._a = self._design_filter()
        self._zi = np.zeros_like(lfilter_zi(self._b, self._a))
        self.reset_state()

    @property
    def order(self) -> int:
        """Get filter order."""
        return self._order

    @order.setter
    def order(self, value: int):
        """Set filter order and redesign filter."""
        self._order = value
        self._b, self._a = self._design_filter()
        self._zi = np.zeros_like(lfilter_zi(self._b, self._a))
        self.reset_state()

    @property
    def filter_type(self) -> str:
        """Get filter type."""
        return self._filter_type


@register_component()
class BiquadResonantFilter(Modifier):
    """Stateful RBJ biquad filter with resonance, drive, and modulation support.

    This is intended as a synth-character filter. ``ButterworthFilter`` remains
    the neutral utility filter; this component provides a resonant cutoff peak
    and optional saturation for more musical sweeps.
    """

    FilterType = Literal["low", "high", "band", "notch"]

    descriptor = ComponentDescriptor(
        name="BiquadResonantFilter",
        category=ComponentCategory.MODIFIER,
        parameters=make_parameter_descriptors(
            "cutoff",
            "resonance",
            "filter_type",
            "drive_db",
            "output_gain_db",
            "sample_rate",
            cutoff=ParameterDescriptor(
                name="cutoff",
                default=1000.0,
                minimum=20.0,
                unit="Hz",
                description="Cutoff or center frequency.",
                smoothing_policy=SmoothingPolicy.EXPONENTIAL,
                smoothing_duration_ms=50.0,
            ),
            resonance=ParameterDescriptor(
                name="resonance",
                default=0.707,
                minimum=0.1,
                maximum=30.0,
                description="Filter Q/resonance.",
                smoothing_policy=SmoothingPolicy.LINEAR,
                smoothing_duration_ms=50.0,
            ),
            filter_type=ParameterDescriptor(
                name="filter_type",
                default="low",
                choices=("low", "high", "band", "notch"),
                description="Resonant biquad filter type.",
            ),
            drive_db=ParameterDescriptor(
                name="drive_db",
                default=0.0,
                unit="dB",
                description="Pre-filter saturation drive.",
                smoothing_policy=SmoothingPolicy.LINEAR,
                smoothing_duration_ms=50.0,
            ),
            output_gain_db=ParameterDescriptor(
                name="output_gain_db",
                default=0.0,
                unit="dB",
                description="Post-filter output gain.",
                smoothing_policy=SmoothingPolicy.LINEAR,
                smoothing_duration_ms=50.0,
            ),
        ),
        description="Resonant RBJ biquad synth filter with drive",
        tags=["filter", "biquad", "resonant", "synth", "drive"],
    )

    def __init__(
        self,
        cutoff: float = 1000.0,
        resonance: float = 0.707,
        filter_type: FilterType = "low",
        drive_db: float = 0.0,
        output_gain_db: float = 0.0,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
    ) -> None:
        """Initialize resonant biquad filter."""
        super().__init__()
        self.sample_rate = validate_sample_rate(sample_rate)
        self._filter_type: BiquadResonantFilter.FilterType = filter_type
        self._filter_state: np.ndarray | None = None

        if filter_type not in ("low", "high", "band", "notch"):
            raise ValueError(
                "filter_type must be 'low', 'high', 'band', or 'notch', "
                f"got '{filter_type}'"
            )

        # Create RuntimeParameters for smoothed parameters
        cutoff_descriptor = ParameterDescriptor(
            name="cutoff",
            default=cutoff,
            minimum=20.0,
            unit="Hz",
            smoothing_policy=SmoothingPolicy.EXPONENTIAL,
            smoothing_duration_ms=50.0,
        )
        self._cutoff_param = RuntimeParameter(cutoff_descriptor, self.sample_rate)

        resonance_descriptor = ParameterDescriptor(
            name="resonance",
            default=resonance,
            minimum=0.1,
            maximum=30.0,
            smoothing_policy=SmoothingPolicy.LINEAR,
            smoothing_duration_ms=50.0,
        )
        self._resonance_param = RuntimeParameter(resonance_descriptor, self.sample_rate)

        drive_db_descriptor = ParameterDescriptor(
            name="drive_db",
            default=drive_db,
            unit="dB",
            smoothing_policy=SmoothingPolicy.LINEAR,
            smoothing_duration_ms=50.0,
        )
        self._drive_db_param = RuntimeParameter(drive_db_descriptor, self.sample_rate)

        output_gain_db_descriptor = ParameterDescriptor(
            name="output_gain_db",
            default=output_gain_db,
            unit="dB",
            smoothing_policy=SmoothingPolicy.LINEAR,
            smoothing_duration_ms=50.0,
        )
        self._output_gain_db_param = RuntimeParameter(
            output_gain_db_descriptor, self.sample_rate
        )

        # Design initial filter coefficients
        self._b, self._a = self._design_filter(
            self._cutoff_param.value, self._resonance_param.value
        )

    @property
    def cutoff(self) -> float:
        return self._cutoff_param._target_value

    @cutoff.setter
    def cutoff(self, value: float) -> None:
        self._cutoff_param.value = float(value)
        # Update static coefficients for non-modulated processing
        self._b, self._a = self._design_filter(
            self._cutoff_param._target_value, self._resonance_param._target_value
        )

    @property
    def resonance(self) -> float:
        return self._resonance_param._target_value

    @resonance.setter
    def resonance(self, value: float) -> None:
        self._resonance_param.value = float(value)
        # Update static coefficients for non-modulated processing
        self._b, self._a = self._design_filter(
            self._cutoff_param._target_value, self._resonance_param._target_value
        )

    @property
    def filter_type(self) -> str:
        return self._filter_type

    @filter_type.setter
    def filter_type(self, value: FilterType) -> None:
        if value not in ("low", "high", "band", "notch"):
            raise ValueError(
                f"filter_type must be 'low', 'high', 'band', or 'notch', got '{value}'"
            )
        self._filter_type = value
        self._b, self._a = self._design_filter(
            self._cutoff_param._target_value, self._resonance_param._target_value
        )
        self.reset_state()

    @property
    def drive_db(self) -> float:
        return self._drive_db_param._target_value

    @drive_db.setter
    def drive_db(self, value: float) -> None:
        self._drive_db_param.value = float(value)

    @property
    def output_gain_db(self) -> float:
        return self._output_gain_db_param._target_value

    @output_gain_db.setter
    def output_gain_db(self, value: float) -> None:
        self._output_gain_db_param.value = float(value)

    def reset_state(self) -> None:
        """Reset filter delay memory."""
        self._filter_state = None

    def __iter__(self):
        self.reset_state()
        return self

    def __next__(self):
        raise NotImplementedError(
            "BiquadResonantFilter requires an input signal. Call it with samples."
        )

    def __call__(
        self, val: float | tuple[float, ...] | np.ndarray
    ) -> float | tuple[float, ...] | np.ndarray:
        """Apply the filter to scalar, stereo tuple, or array input."""
        if isinstance(val, np.ndarray):
            return self.scale_vectorized(val)
        if isinstance(val, tuple):
            result = self.scale_vectorized(np.asarray([val], dtype=np.float32))[0]
            return tuple(float(sample) for sample in result)
        return float(self.scale_vectorized(np.asarray([val], dtype=np.float32))[0])

    def scale_vectorized(self, samples: np.ndarray) -> np.ndarray:
        """Apply the current static filter coefficients to a sample buffer."""
        if samples.size == 0:
            return samples

        # Check if any parameters are smoothing
        num_samples = len(samples)
        cutoff_smoothing = self._cutoff_param._smoothing_samples_remaining > 0
        resonance_smoothing = self._resonance_param._smoothing_samples_remaining > 0

        # If parameters are smoothing, use per-sample modulation for smooth transitions
        if cutoff_smoothing or resonance_smoothing:
            cutoff_envelope = self._cutoff_param.get_interpolated_buffer(num_samples)
            resonance_envelope = self._resonance_param.get_interpolated_buffer(
                num_samples
            )
            drive_envelope = self._drive_db_param.get_interpolated_buffer(num_samples)
            output_gain_envelope = self._output_gain_db_param.get_interpolated_buffer(
                num_samples
            )

            return self.process_modulated(
                samples,
                cutoff_values=cutoff_envelope,
                resonance_values=resonance_envelope,
                drive_db_values=drive_envelope,
                output_gain_db_values=output_gain_envelope,
            )

        # Use static filtering with current RuntimeParameter values
        shaped = self._apply_drive(
            np.asarray(samples, dtype=np.float32),
            drive_db_value=self._drive_db_param.value,
        )
        zi = self._filter_state_for(shaped)
        filtered, self._filter_state = lfilter(
            self._b,
            self._a,
            shaped,
            axis=0,
            zi=zi,
        )
        return self._apply_output_gain(
            filtered, output_gain_db_value=self._output_gain_db_param.value
        ).astype(np.float32)

    def process_modulated(
        self,
        samples: np.ndarray,
        cutoff_values: np.ndarray | None = None,
        resonance_values: np.ndarray | None = None,
        drive_db_values: np.ndarray | None = None,
        output_gain_db_values: np.ndarray | None = None,
    ) -> np.ndarray:
        """Process a buffer with optional per-sample cutoff modulation."""
        samples = np.asarray(samples, dtype=np.float32)
        if samples.size == 0:
            return samples
        if (
            cutoff_values is None
            and resonance_values is None
            and drive_db_values is None
            and output_gain_db_values is None
        ):
            return self.scale_vectorized(samples)

        shaped = self._apply_drive(samples, drive_db_values)
        cutoff_array = self._fit_cutoff_values(cutoff_values, len(shaped))
        resonance_array = self._fit_resonance_values(resonance_values, len(shaped))
        result = self._process_modulated_samples(shaped, cutoff_array, resonance_array)
        return self._apply_output_gain(result, output_gain_db_values).astype(np.float32)

    def configure(
        self,
        *,
        cutoff: float,
        resonance: float,
        filter_type: FilterType,
        drive_db: float,
        output_gain_db: float,
    ) -> None:
        """Update runtime parameters while preserving delay state when possible."""
        reset_state = filter_type != self._filter_type
        self._cutoff_param.value = float(cutoff)
        self._resonance_param.value = float(resonance)
        self._filter_type = filter_type
        self._drive_db_param.value = float(drive_db)
        self._output_gain_db_param.value = float(output_gain_db)
        self._b, self._a = self._design_filter(
            self._cutoff_param._target_value, self._resonance_param._target_value
        )
        if reset_state:
            self.reset_state()

    def _process_modulated_samples(
        self,
        samples: np.ndarray,
        cutoff_values: np.ndarray,
        resonance_values: np.ndarray,
    ) -> np.ndarray:
        trailing_shape = samples.shape[1:]
        state_shape = (2,) + trailing_shape
        if self._filter_state is None or self._filter_state.shape != state_shape:
            self._filter_state = np.zeros(state_shape, dtype=np.float64)

        z1 = self._filter_state[0]
        z2 = self._filter_state[1]
        output = np.empty_like(samples, dtype=np.float32)
        b0, b1, b2, a1, a2 = self._design_filter_values(
            cutoff_values,
            resonance_values,
        )
        for index, sample in enumerate(samples):
            y = b0[index] * sample + z1
            z1 = b1[index] * sample - a1[index] * y + z2
            z2 = b2[index] * sample - a2[index] * y
            output[index] = y

        self._filter_state = np.stack((z1, z2)).astype(np.float64, copy=False)
        return output

    def _filter_state_for(self, samples: np.ndarray) -> np.ndarray:
        expected_shape = (2,) + samples.shape[1:]
        if self._filter_state is None or self._filter_state.shape != expected_shape:
            self._filter_state = np.zeros(expected_shape, dtype=np.float64)
        return self._filter_state

    def _fit_parameter_values(
        self,
        values: np.ndarray | None,
        num_samples: int,
        fallback: float,
        *,
        minimum: float,
        maximum: float,
    ) -> np.ndarray:
        if values is None:
            return np.full(num_samples, fallback, dtype=np.float32)

        parameter_values = np.asarray(values, dtype=np.float32).reshape(-1)
        if len(parameter_values) == num_samples:
            return np.clip(parameter_values, minimum, maximum)
        if len(parameter_values) < num_samples:
            padded = np.empty(num_samples, dtype=np.float32)
            padded[: len(parameter_values)] = parameter_values
            padded[len(parameter_values) :] = (
                parameter_values[-1] if len(parameter_values) else fallback
            )
            return np.clip(padded, minimum, maximum)
        return np.clip(parameter_values[:num_samples], minimum, maximum)

    def _fit_cutoff_values(
        self, cutoff_values: np.ndarray | None, num_samples: int
    ) -> np.ndarray:
        return self._fit_parameter_values(
            cutoff_values,
            num_samples,
            self._cutoff_param.value,
            minimum=1.0,
            maximum=self.sample_rate * 0.475,
        )

    def _fit_resonance_values(
        self, resonance_values: np.ndarray | None, num_samples: int
    ) -> np.ndarray:
        return self._fit_parameter_values(
            resonance_values,
            num_samples,
            self._resonance_param.value,
            minimum=0.1,
            maximum=30.0,
        )

    def _design_filter(
        self, cutoff: float, resonance: float
    ) -> tuple[np.ndarray, np.ndarray]:
        nyq = self.sample_rate * 0.5
        safe_cutoff = float(np.clip(cutoff, 1.0, nyq * 0.95))
        q = float(np.clip(resonance, 0.1, 30.0))
        omega = 2.0 * np.pi * safe_cutoff / self.sample_rate
        sin_omega = np.sin(omega)
        cos_omega = np.cos(omega)
        alpha = sin_omega / (2.0 * q)

        if self._filter_type == "low":
            b0 = (1.0 - cos_omega) * 0.5
            b1 = 1.0 - cos_omega
            b2 = (1.0 - cos_omega) * 0.5
        elif self._filter_type == "high":
            b0 = (1.0 + cos_omega) * 0.5
            b1 = -(1.0 + cos_omega)
            b2 = (1.0 + cos_omega) * 0.5
        elif self._filter_type == "band":
            b0 = alpha
            b1 = 0.0
            b2 = -alpha
        else:
            b0 = 1.0
            b1 = -2.0 * cos_omega
            b2 = 1.0

        a0 = 1.0 + alpha
        a1 = -2.0 * cos_omega
        a2 = 1.0 - alpha
        b = np.asarray([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
        a = np.asarray([1.0, a1 / a0, a2 / a0], dtype=np.float64)
        return b, a

    def _design_filter_values(
        self,
        cutoff_values: np.ndarray,
        resonance_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        nyq = self.sample_rate * 0.5
        safe_cutoff = np.clip(cutoff_values, 1.0, nyq * 0.95).astype(np.float64)
        q = np.clip(resonance_values, 0.1, 30.0).astype(np.float64)
        omega = 2.0 * np.pi * safe_cutoff / self.sample_rate
        sin_omega = np.sin(omega)
        cos_omega = np.cos(omega)
        alpha = sin_omega / (2.0 * q)

        if self._filter_type == "low":
            b0 = (1.0 - cos_omega) * 0.5
            b1 = 1.0 - cos_omega
            b2 = (1.0 - cos_omega) * 0.5
        elif self._filter_type == "high":
            b0 = (1.0 + cos_omega) * 0.5
            b1 = -(1.0 + cos_omega)
            b2 = (1.0 + cos_omega) * 0.5
        elif self._filter_type == "band":
            b0 = alpha
            b1 = np.zeros_like(alpha)
            b2 = -alpha
        else:
            b0 = np.ones_like(alpha)
            b1 = -2.0 * cos_omega
            b2 = np.ones_like(alpha)

        a0 = 1.0 + alpha
        a1 = -2.0 * cos_omega
        a2 = 1.0 - alpha
        return (
            b0 / a0,
            b1 / a0,
            b2 / a0,
            a1 / a0,
            a2 / a0,
        )

    def _apply_drive(
        self,
        samples: np.ndarray,
        drive_db_values: np.ndarray | None = None,
        drive_db_value: float | None = None,
    ) -> np.ndarray:
        drive: float | np.ndarray
        if drive_db_values is None:
            # Use scalar value (from RuntimeParameter or direct parameter)
            drive_value = (
                drive_db_value
                if drive_db_value is not None
                else self._drive_db_param.value
            )
            drive = float(db_to_linear(drive_value))
        else:
            drive = np.asarray(db_to_linear(drive_db_values), dtype=np.float32)

        if np.all(drive <= 1.0001):
            return samples
        driven = np.tanh(samples * drive) / np.tanh(drive)
        return np.where(drive <= 1.0001, samples, driven).astype(np.float32)

    def _apply_output_gain(
        self,
        samples: np.ndarray,
        output_gain_db_values: np.ndarray | None = None,
        output_gain_db_value: float | None = None,
    ) -> np.ndarray:
        if output_gain_db_values is None:
            # Use scalar value (from RuntimeParameter or direct parameter)
            gain_value = (
                output_gain_db_value
                if output_gain_db_value is not None
                else self._output_gain_db_param.value
            )
            return samples * db_to_linear(gain_value)
        return samples * db_to_linear(output_gain_db_values)


# Utility functions for standalone use


def butter(
    order: int,
    cutoff: float | tuple[float, float] | list[float] | np.ndarray,
    fs: float,
    btype: Literal["low", "high", "band"] = "low",
):
    """
    Design Butterworth IIR filter coefficients.

    Args:
        order: Filter order.
        cutoff: Cutoff frequency/frequencies in Hz. For band filters pass (low, high).
        fs: Sampling frequency in Hz.
        btype: {'low', 'high', 'band'} Filter type.

    Returns:
        b, a : ndarray
            Numerator (b) and denominator (a) polynomials of the IIR filter.
    """
    nyq = 0.5 * fs
    wn: float | list[float]
    if isinstance(cutoff, (list, tuple, np.ndarray)):
        wn = [c / nyq for c in cutoff]
    else:
        wn = cutoff / nyq

    b, a = scipy_butter(
        order, wn, btype=cast(Literal["low", "high", "band"], btype), analog=False
    )
    return b, a


def apply_filter(b, a, x):
    """Apply zero-phase filter (filtfilt) to signal x.

    Args:
        b: Numerator coefficients of the filter.
        a: Denominator coefficients of the filter.
        x: Input signal array.

    Returns:
        Filtered signal array.
    """
    return filtfilt(b, a, x)
