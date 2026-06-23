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
from src.engine.audio_component import ComponentCategory, ComponentDescriptor
from src.engine.audio_component_registry import register_component
from src.engine.modifier import Modifier
from src.engine.validation import validate_sample_rate


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
        config_params=["cutoff", "order", "filter_type", "sample_rate"],
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
