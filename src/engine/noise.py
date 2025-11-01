"""Noise generators for audio synthesis and sound design.

This module provides various types of noise generators commonly used in
synthesizers and sound design. Each noise type has distinct spectral
characteristics suitable for different applications.

Functions:
    white_noise: Generate white noise (equal energy across all frequencies).
    pink_noise: Generate pink noise (1/f spectrum, perceptually balanced).
    brownian_noise: Generate Brownian/brown noise (1/f² spectrum).
    blue_noise: Generate blue noise (increasing energy with frequency).
    perlin_noise: Generate smooth, organic Perlin noise.

Example:
    >>> from src.constants import DEFAULT_SAMPLE_RATE
    >>>
    >>> # Generate 1 second of white noise
    >>> noise = white_noise(dur=1.0, amplitude=0.5, sr=DEFAULT_SAMPLE_RATE)
    >>>
    >>> # Generate pink noise for ambient sound
    >>> ambient = pink_noise(dur=2.0, amplitude=0.3)
    >>>
    >>> # Generate Perlin noise for smooth modulation
    >>> modulation = perlin_noise(
    ...     dur=5.0,
    ...     frequency=2.0,
    ...     amplitude=1.0,
    ...     octaves=3
    ... )

Noise Types:
    - White: Flat spectrum, harsh sound, useful for percussion
    - Pink: 1/f spectrum, perceptually balanced, natural sound
    - Brown: 1/f² spectrum, deep rumble, sub-bass content
    - Blue: Increasing with frequency, bright, airy sound
    - Perlin: Smooth, organic variation, excellent for modulation

Applications:
    - Percussion synthesis (white/pink noise)
    - Ambient soundscapes (pink/brown noise)
    - Wind/air sounds (blue noise)
    - LFO/modulation sources (Perlin noise)
    - Dithering and testing (white noise)

Note:
    All noise generators return NumPy arrays. Use the seed parameter
    for reproducible results in testing and composition.
"""

import numpy as np

from src.constants import DEFAULT_SAMPLE_RATE


def white_noise(dur: float, amplitude=1.0, sr: float = DEFAULT_SAMPLE_RATE, seed=None):
    """Generate white noise.

    Args:
        dur: Duration of the output noise in seconds.
        amplitude: Amplitude of the noise.
        sr: Sample rate.
        seed: Random seed for reproducibility.

    Returns:
        A numpy array of white noise.
    """
    if seed is not None:
        np.random.seed(seed)

    length = int(dur * sr)
    return amplitude * np.random.uniform(-1, 1, length)


def pink_noise(length, amplitude=1.0, seed=None):
    """Generate pink noise.

    Args:
        length: Length of the output noise array.
        amplitude: Amplitude of the noise.
        seed: Random seed for reproducibility.

    Returns:
        A numpy array of pink noise.
    """
    if seed is not None:
        np.random.seed(seed)

    # Voss-McCartney algorithm for pink noise
    num_rows = 16
    array = np.zeros((num_rows, length))
    array[0, :] = np.random.uniform(-1, 1, length)

    for i in range(1, num_rows):
        step = 2**i
        for j in range(0, length, step):
            array[i, j : j + step] = np.random.uniform(-1, 1)

    pink = np.sum(array, axis=0) / num_rows
    return amplitude * pink


def brownian_noise(
    dur: float, amplitude=1.0, sr: float = DEFAULT_SAMPLE_RATE, seed=None
):
    """Generate Brownian noise (red noise).

    Args:
        dur: Duration of the output noise in seconds.
        amplitude: Amplitude of the noise.
        sr: Sample rate.
        seed: Random seed for reproducibility.

    Returns:
        A numpy array of Brownian noise.
    """
    if seed is not None:
        np.random.seed(seed)

    length = int(dur * sr)
    white = np.random.uniform(-1, 1, length)
    brown = np.cumsum(white)
    brown = brown / np.max(np.abs(brown))  # Normalize
    return amplitude * brown


def blue_noise(dur: float, amplitude=1.0, sr: float = DEFAULT_SAMPLE_RATE, seed=None):
    """Generate blue noise.

    Args:
        dur: Duration of the output noise in seconds.
        amplitude: Amplitude of the noise.
        sr: Sample rate.
        seed: Random seed for reproducibility.

    Returns:
        A numpy array of blue noise.
    """
    if seed is not None:
        np.random.seed(seed)

    length = int(dur * sr)
    white = np.random.uniform(-1, 1, length)
    blue = np.diff(white, prepend=0)
    blue = blue / np.max(np.abs(blue))  # Normalize
    return amplitude * blue


def perlin_noise(dur: float, scale=10, sr: float = DEFAULT_SAMPLE_RATE, seed=None):
    """Generate 1D Perlin noise.

    Args:
        dur: Duration of the output noise in seconds.
        scale: Scale of the noise (higher values = more variation).
        sr: Sample rate.
        seed: Random seed for reproducibility.

    Returns:
        A numpy array of Perlin noise.
    """
    if seed is not None:
        np.random.seed(seed)

    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def grad(hash, x):
        h = hash & 15
        grad = 1 + (h & 7)  # Gradient value from 1 to 8
        if (h & 8) != 0:
            grad = -grad
        return grad * x

    # Generate permutation table
    p = np.arange(256, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()

    length = int(dur * sr)
    noise = np.zeros(length)
    for i in range(length):
        x = i / scale
        xi = int(np.floor(x)) & 255
        xf = x - np.floor(x)
        u = fade(xf)

        aa = p[xi]
        ab = p[xi + 1]

        x1 = grad(aa, xf)
        x2 = grad(ab, xf - 1)

        noise[i] = (1 - u) * x1 + u * x2

    return noise
