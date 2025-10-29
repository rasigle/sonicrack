import numpy as np
from scipy.signal import filtfilt


def butter(order, cutoff, fs, btype='low'):
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
