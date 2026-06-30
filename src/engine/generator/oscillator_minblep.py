"""Shared minBLEP utilities for bandlimited oscillator edges."""

import numpy as np

TWO_PI = 2 * np.pi
VCV_MINBLEP_ZERO_CROSSINGS = 16
VCV_MINBLEP_OVERSAMPLE = 16


def _blackman_harris(position: np.ndarray) -> np.ndarray:
    return (
        0.35875
        - 0.48829 * np.cos(TWO_PI * position)
        + 0.14128 * np.cos(2 * TWO_PI * position)
        - 0.01168 * np.cos(3 * TWO_PI * position)
    )


def minimum_phase_minblep_table(
    zero_crossings: int = VCV_MINBLEP_ZERO_CROSSINGS,
    oversample: int = VCV_MINBLEP_OVERSAMPLE,
) -> np.ndarray:
    n = 2 * zero_crossings * oversample
    positions = np.arange(n, dtype=np.float64) / oversample - zero_crossings
    impulse = np.sinc(positions)
    impulse *= _blackman_harris(np.arange(n, dtype=np.float64) / (n - 1))

    spectrum = np.fft.fft(impulse)
    log_magnitude = np.log(np.maximum(np.abs(spectrum), np.exp(-10.0)))
    cepstrum = np.fft.ifft(log_magnitude)
    cepstrum[1 : n // 2] *= 2.0
    cepstrum[n // 2 :] = 0.0
    minimum_phase = np.fft.ifft(np.exp(np.fft.fft(cepstrum))).real

    step = np.cumsum(minimum_phase) / oversample
    step = np.concatenate(([0.0], step[:-1]))
    step /= step[-1] + minimum_phase[-1] / oversample
    return np.asarray(step - 1.0, dtype=np.float32)
