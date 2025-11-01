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

import numpy as np
from scipy.signal import filtfilt


def butter(order, cutoff, fs, btype="low"):
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
    if isinstance(cutoff, (list, tuple, np.ndarray)):
        wn = [c / nyq for c in cutoff]
    else:
        wn = cutoff / nyq

    from scipy.signal import butter

    b, a = butter(order, wn, btype=btype, analog=False)
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
