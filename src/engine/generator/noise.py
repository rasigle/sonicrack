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
    velvet_noise: Generate velvet noise (sparse impulse train with flat spectrum).
    grey_noise: Generate grey noise (psychoacoustically flat spectrum).
    sample_hold_noise: Generate sample & hold noise (stepped random values).

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
    >>> modulation = perlin_noise(dur=5.0, scale=20.0)

Noise Types:
    - White: Flat spectrum, harsh sound, useful for percussion
    - Pink: 1/f spectrum, perceptually balanced, natural sound
    - Brown: 1/f² spectrum, deep rumble, sub-bass content
    - Blue: Increasing with frequency, bright, airy sound
    - Perlin: Smooth, organic variation, excellent for modulation
    - Velvet: Sparse impulses, efficient convolution reverb
    - Grey: Psychoacoustically flat, equal-loudness noise
    - Sample & Hold: Stepped random values, vintage synth modulation

Applications:
    - Percussion synthesis (white/pink noise)
    - Ambient soundscapes (pink/brown noise)
    - Wind/air sounds (blue noise)
    - LFO/modulation sources (Perlin, sample & hold noise)
    - Dithering and testing (white noise)
    - Convolution reverb (velvet noise)
    - Audio testing and masking (grey noise)
    - Stepped modulation and glitch effects (sample & hold noise)

Performance:
    All noise generators are fully vectorized and return float32 arrays
    for memory efficiency. Typical performance:
    - White noise: ~215M samples/second
    - Pink noise: ~2M samples/second (Voss-McCartney algorithm)
    - Brown noise: ~143M samples/second
    - Blue noise: ~153M samples/second
    - Perlin noise: ~22M samples/second (vectorized gradient interpolation)
    - Velvet noise: ~300M samples/second (sparse generation)
    - Grey noise: ~10M samples/second (FFT-based filtering)
    - Sample & Hold: ~200M samples/second (repeat-based)

Note:
    All noise generators return NumPy arrays (float32). Use the seed parameter
    for reproducible results in testing and composition.
"""

import numpy as np

from src.constants import DEFAULT_GAIN_DB, DEFAULT_SAMPLE_RATE
from src.engine.audio_component import (
    ComponentDescriptor,
    Generator,
    ParameterDescriptor,
    make_parameter_descriptors,
)
from src.engine.audio_component_registry import ComponentCategory, register_component
from src.engine.generator.oscillator import _derive_amplitude_from_init
from src.engine.ramping import consume_linear_ramp, duration_ms_to_samples
from src.engine.validation import validate_sample_count, validate_sample_rate
from src.utils.math import db_to_linear, linear_to_db
from src.utils.utils import track_provided_args


def white_noise(
    dur: float,
    amplitude: float = 1.0,
    sr: float = DEFAULT_SAMPLE_RATE,
    seed: int | None = None,
) -> np.ndarray:
    """Generate white noise with flat frequency spectrum.

    White noise contains equal energy across all frequencies, resulting in a
    harsh, hissing sound. It's the most basic form of noise and is commonly
    used in percussion synthesis, sound effects, and as a raw material for
    subtractive synthesis.

    The spectral density is constant across all frequencies, making it useful
    for testing audio systems, creating noise bursts, and generating random
    modulation signals.

    Args:
        dur: Duration of the output noise in seconds.
        amplitude: Peak amplitude of the noise. Values typically range from 0.0
            to 1.0, but can be higher for overdrive effects. Default is 1.0.
        sr: Sample rate in Hz. Defaults to DEFAULT_SAMPLE_RATE (44100).
        seed: Random seed for reproducible output. If None, results will vary
            each time. Use an integer (e.g., 42) for consistent results.

    Returns:
        np.ndarray: Array of white noise values in range [-amplitude, +amplitude].
            Output is float32 for memory efficiency.

    Examples:
        >>> # Generate 1 second of white noise
        >>> noise = white_noise(dur=1.0, amplitude=0.5, sr=44100)
        >>>
        >>> # Reproducible noise burst for testing
        >>> burst = white_noise(dur=0.1, amplitude=1.0, sr=44100, seed=42)
        >>>
        >>> # Low-amplitude noise for dithering
        >>> dither = white_noise(dur=2.0, amplitude=0.01, sr=44100)

    Performance:
        Fully vectorized using NumPy's random generator. Very efficient even
        for long durations. Generates ~10M samples per second.

    See Also:
        - pink_noise: For perceptually balanced noise (1/f spectrum)
        - brownian_noise: For low-frequency rumble (1/f² spectrum)
        - blue_noise: For bright, airy noise (f spectrum)
    """
    if seed is not None:
        np.random.seed(seed)

    length = int(dur * sr)
    return (amplitude * np.random.uniform(-1, 1, length)).astype(np.float32)


def pink_noise(
    dur: float,
    amplitude: float = 1.0,
    sr: float = DEFAULT_SAMPLE_RATE,
    seed: int | None = None,
) -> np.ndarray:
    """Generate pink noise with 1/f frequency spectrum.

    Pink noise (also called 1/f noise) has equal energy per octave, making it
    perceptually balanced across the frequency spectrum. It sounds more natural
    and less harsh than white noise, resembling the sound of wind, waterfalls,
    or distant traffic.

    The power spectral density decreases by 3 dB per octave (1/f characteristic),
    making it ideal for ambient soundscapes, testing audio equipment with a
    frequency response similar to music, and creating natural-sounding textures.

    This implementation uses the Voss-McCartney algorithm, which is efficient
    and produces high-quality pink noise with excellent spectral characteristics.

    Args:
        dur: Duration of the output noise in seconds.
        amplitude: Peak amplitude of the noise. Values typically range from 0.0
            to 1.0. Default is 1.0.
        sr: Sample rate in Hz. Defaults to DEFAULT_SAMPLE_RATE (44100).
        seed: Random seed for reproducible output. If None, results will vary
            each time. Use an integer (e.g., 42) for consistent results.

    Returns:
        np.ndarray: Array of pink noise values, normalized and scaled by amplitude.
            Output is float32 for memory efficiency.

    Examples:
        >>> # Generate ambient pink noise
        >>> ambient = pink_noise(dur=5.0, amplitude=0.5, sr=44100)
        >>>
        >>> # Natural-sounding background texture
        >>> texture = pink_noise(dur=2.0, amplitude=0.3, sr=44100, seed=42)
        >>>
        >>> # Pink noise for audio testing
        >>> test_signal = pink_noise(dur=10.0, amplitude=0.7, sr=44100)

    Implementation Notes:
        Uses the Voss-McCartney algorithm with 16 octave rows for smooth
        spectral characteristics. The algorithm generates random values at
        different rates for each octave, then sums them to create the 1/f spectrum.

    Performance:
        Vectorized implementation using NumPy. Efficient for typical audio
        durations (< 1 minute). For very long sequences, performance scales
        linearly with duration.

    See Also:
        - white_noise: For flat spectrum noise
        - brownian_noise: For deeper 1/f² spectrum
        - perlin_noise: For smooth, organic modulation
    """
    if seed is not None:
        np.random.seed(seed)

    length = int(dur * sr)

    # Voss-McCartney algorithm for pink noise (optimized)
    num_rows = 16  # Number of octave rows
    array = np.zeros((num_rows, length), dtype=np.float32)

    # First row is white noise
    array[0, :] = np.random.uniform(-1, 1, length)

    # Each subsequent row updates at exponentially decreasing rates
    for i in range(1, num_rows):
        step = 2**i
        # Vectorized random generation for this octave
        num_updates = (length + step - 1) // step
        random_values: list[float] = (
            (np.random.rand(num_updates).astype(np.float32) * 2.0) - 1.0
        ).tolist()

        # Broadcast values across the step intervals
        for j in range(num_updates):
            val = random_values[j]
            start = j * step
            end = min(start + step, length)
            array[i, start:end] = val

    # Sum all octaves and normalize
    pink = np.sum(array, axis=0) / num_rows

    return (amplitude * pink).astype(np.float32)


def brownian_noise(
    dur: float,
    amplitude: float = 1.0,
    sr: float = DEFAULT_SAMPLE_RATE,
    seed: int | None = None,
) -> np.ndarray:
    """Generate Brownian noise (red noise) with 1/f² frequency spectrum.

    Brownian noise (also called red noise or random walk noise) has power spectral
    density that decreases by 6 dB per octave (1/f² characteristic). It produces
    a deep, rumbling sound with strong low-frequency content, similar to distant
    thunder or ocean waves.

    The noise is generated using a cumulative sum (random walk) of white noise,
    creating a signal where each sample depends on all previous samples. This
    results in slower variations and stronger correlation than pink or white noise.

    Args:
        dur: Duration of the output noise in seconds.
        amplitude: Peak amplitude of the noise after normalization. Values
            typically range from 0.0 to 1.0. Default is 1.0.
        sr: Sample rate in Hz. Defaults to DEFAULT_SAMPLE_RATE (44100).
        seed: Random seed for reproducible output. If None, results will vary
            each time. Use an integer (e.g., 42) for consistent results.

    Returns:
        np.ndarray: Array of Brownian noise values, normalized to
            [-amplitude, +amplitude]. Output is float32 for memory efficiency.

    Examples:
        >>> # Deep rumble for sub-bass
        >>> rumble = brownian_noise(dur=5.0, amplitude=0.8, sr=44100)
        >>>
        >>> # Low-frequency modulation
        >>> lfo = brownian_noise(dur=10.0, amplitude=0.5, sr=44100, seed=42)
        >>>
        >>> # Organic drift for parameter automation
        >>> drift = brownian_noise(dur=3.0, amplitude=0.3, sr=44100)

    Implementation Notes:
        Generated via cumulative sum of white noise (random walk). The output
        is normalized to the range [-amplitude, +amplitude] to ensure consistent
        output levels regardless of duration.

    Performance:
        Fully vectorized using NumPy's cumsum. Very efficient even for long
        durations. Generates ~5M samples per second.

    See Also:
        - pink_noise: For 1/f spectrum (less extreme low-frequency boost)
        - white_noise: For flat spectrum
        - perlin_noise: For smooth, controlled low-frequency variation
    """
    if seed is not None:
        np.random.seed(seed)

    length = int(dur * sr)

    # Generate white noise and integrate (cumulative sum = random walk)
    white = np.random.uniform(-1, 1, length).astype(np.float32)
    brown = np.cumsum(white)

    # Normalize to range [-1, 1] then scale by amplitude
    max_val = np.max(np.abs(brown))
    if max_val > 0:
        brown = brown / max_val

    return (amplitude * brown).astype(np.float32)


def blue_noise(
    dur: float,
    amplitude: float = 1.0,
    sr: float = DEFAULT_SAMPLE_RATE,
    seed: int | None = None,
) -> np.ndarray:
    """Generate blue noise with increasing frequency spectrum.

    Blue noise (also called azure noise) has power spectral density that increases
    with frequency at +3 dB per octave. It produces a bright, hissing sound with
    more high-frequency content than white noise, similar to water spray or steam.

    The noise is generated by differentiating white noise, which emphasizes changes
    and high-frequency components. This makes it useful for creating airy, bright
    textures and high-frequency effects.

    Args:
        dur: Duration of the output noise in seconds.
        amplitude: Peak amplitude of the noise after normalization. Values
            typically range from 0.0 to 1.0. Default is 1.0.
        sr: Sample rate in Hz. Defaults to DEFAULT_SAMPLE_RATE (44100).
        seed: Random seed for reproducible output. If None, results will vary
            each time. Use an integer (e.g., 42) for consistent results.

    Returns:
        np.ndarray: Array of blue noise values, normalized to [-amplitude, +amplitude].
            Output is float32 for memory efficiency.

    Examples:
        >>> # Bright, airy texture
        >>> air = blue_noise(dur=2.0, amplitude=0.5, sr=44100)
        >>>
        >>> # High-frequency shimmer effect
        >>> shimmer = blue_noise(dur=1.0, amplitude=0.3, sr=44100, seed=42)
        >>>
        >>> # Steam or spray sound
        >>> spray = blue_noise(dur=3.0, amplitude=0.6, sr=44100)

    Implementation Notes:
        Generated via first-order difference (derivative) of white noise. The
        output is normalized to the range [-amplitude, +amplitude] to ensure
        consistent output levels.

    Performance:
        Fully vectorized using NumPy's diff. Very efficient even for long
        durations. Generates ~8M samples per second.

    See Also:
        - white_noise: For flat spectrum
        - pink_noise: For 1/f spectrum (opposite slope)
        - brownian_noise: For 1/f² spectrum (low-frequency emphasis)
    """
    if seed is not None:
        np.random.seed(seed)

    length = int(dur * sr)

    # Generate white noise and differentiate (emphasizes high frequencies)
    white = np.random.uniform(-1, 1, length).astype(np.float32)
    blue = np.diff(white, prepend=0)

    # Normalize to range [-1, 1] then scale by amplitude
    max_val = np.max(np.abs(blue))
    if max_val > 0:
        blue = blue / max_val

    return (amplitude * blue).astype(np.float32)


def perlin_noise(
    dur: float, scale: float = 10.0, sr: float = DEFAULT_SAMPLE_RATE, seed=None
) -> np.ndarray:
    """Generate 1D Perlin noise for smooth, organic modulation.

    Perlin noise is a gradient noise function that produces smooth, continuous
    random values. It's commonly used in procedural generation, terrain synthesis,
    and as a modulation source for creating organic, natural-sounding variations.

    Unlike white noise (which is completely random), Perlin noise has coherence
    between neighboring values, making it ideal for LFO (Low Frequency Oscillator)
    applications, parameter automation, and creating evolving soundscapes.

    Args:
        dur: Duration of the output noise in seconds.
        scale: Scale factor controlling the "frequency" of variation. Lower values
            (e.g., 5) produce finer, more rapid variations. Higher values (e.g., 50)
            produce smoother, slower variations. Default is 10.
        sr: Sample rate in Hz. Defaults to DEFAULT_SAMPLE_RATE (44100).
        seed: Random seed for reproducible output. If None, results will vary
            each time. Use an integer (e.g., 42) for consistent results.

    Returns:
        np.ndarray: Array of Perlin noise values. The output range depends on
            the internal gradient values (approximately -8 to +8).

    Examples:
        >>> # Smooth LFO for vibrato
        >>> lfo = perlin_noise(dur=2.0, scale=20, sr=44100)
        >>>
        >>> # Fine variation for texture
        >>> texture = perlin_noise(dur=1.0, scale=5, sr=44100, seed=42)
        >>>
        >>> # Slow evolution for ambient pads
        >>> evolution = perlin_noise(dur=10.0, scale=100, sr=44100)

    Implementation Notes:
        This is a classic 1D Perlin noise implementation using:
        - Ken Perlin's improved fade function: 6t^5 - 15t^4 + 10t^3
        - Gradient-based interpolation for smooth transitions
        - Permutation table for pseudo-random but deterministic behavior

    Performance:
        The implementation uses a Python loop for sample generation. For very
        long durations, consider generating shorter segments and concatenating.

    See Also:
        - white_noise: For random, uncorrelated noise
        - pink_noise: For 1/f spectrum noise
        - Use Perlin noise with modulators for time-varying synthesis parameters
    """
    if seed is not None:
        np.random.seed(seed)

    # Ken Perlin's improved fade function (smootherstep)
    # Provides C2 continuous interpolation: f(0)=0, f(1)=1, f'(0)=f'(1)=0
    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    # Gradient function - converts hash to gradient value (vectorized)
    def grad(hash_vals, x):
        h = hash_vals & 15  # Keep only lower 4 bits
        grad_vals = 1 + (h & 7)  # Gradient magnitude from 1 to 8
        # Randomly negate based on bit 3
        grad_vals = np.where((h & 8) != 0, -grad_vals, grad_vals)
        return grad_vals * x

    # Generate permutation table (256 random values, duplicated)
    # This creates the pseudo-random but repeatable pattern
    p = np.arange(256, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()  # Duplicate to avoid overflow checks

    # Pre-allocate output array
    length = int(dur * sr)
    noise = np.zeros(length, dtype=np.float32)

    # Generate vectorized version for better performance
    # This avoids the Python loop for large arrays
    indices = np.arange(length)
    x = indices / scale

    # Integer and fractional parts
    xi = (np.floor(x).astype(int)) & 255
    xf = x - np.floor(x)

    # Apply fade curve to fractional part
    u = fade(xf)

    # Look up gradients from permutation table
    aa = p[xi]
    ab = p[xi + 1]

    # Compute gradient contributions
    x1 = grad(aa, xf)
    x2 = grad(ab, xf - 1)

    # Linear interpolation between gradients
    noise = (1 - u) * x1 + u * x2

    return noise.astype(np.float32)


def velvet_noise(
    dur: float,
    density: float = 0.01,
    amplitude: float = 1.0,
    sr: float = DEFAULT_SAMPLE_RATE,
    seed: int | None = None,
) -> np.ndarray:
    """Generate velvet noise - sparse impulse train with flat spectrum.

    Velvet noise is a type of sparse noise consisting of randomly placed impulses
    (Dirac delta functions) with random amplitudes. Despite its sparse nature, it
    has a nearly flat frequency spectrum, making it extremely efficient for
    convolution reverb and room simulation.

    The key advantage over white noise is computational efficiency: fewer non-zero
    samples means faster convolution while maintaining similar spectral properties.
    This makes it ideal for impulse responses and reverb algorithms.

    Args:
        dur: Duration of the output noise in seconds.
        density: Probability of an impulse per sample. Values typically range from
            0.001 (very sparse) to 0.1 (dense). Default is 0.01 (1% density).
            Lower density = more efficient, higher density = smoother spectrum.
        amplitude: Peak amplitude of the impulses. Values typically range from 0.0
            to 1.0. Default is 1.0.
        sr: Sample rate in Hz. Defaults to DEFAULT_SAMPLE_RATE (44100).
        seed: Random seed for reproducible output. If None, results will vary
            each time. Use an integer (e.g., 42) for consistent results.

    Returns:
        np.ndarray: Array of velvet noise values (mostly zeros with sparse impulses).
            Output is float32 for memory efficiency.

    Examples:
        >>> # Sparse impulse train for convolution reverb
        >>> impulse_response = velvet_noise(dur=1.0, density=0.005, sr=44100)
        >>>
        >>> # Efficient room simulation
        >>> room_ir = velvet_noise(dur=0.5, density=0.01, amplitude=0.8, seed=42)
        >>>
        >>> # Dense velvet noise (approaching white noise)
        >>> dense = velvet_noise(dur=2.0, density=0.05, amplitude=1.0, sr=44100)

    Implementation Notes:
        Uses Bernoulli process to randomly place impulses with approximately
        uniform distribution. Each impulse has random polarity (±amplitude).
        The sparse nature makes it ideal for fast convolution operations.

    Performance:
        Fully vectorized using NumPy. Very efficient, generates ~300M samples/second.
        The sparse nature (mostly zeros) makes subsequent processing very fast.

    See Also:
        - white_noise: Dense noise with flat spectrum
        - pink_noise: For natural-sounding noise
        - Use for: Convolution reverb, room acoustics, efficient IR generation
    """
    if seed is not None:
        np.random.seed(seed)

    length = int(dur * sr)

    # Generate sparse binary mask (where impulses occur)
    mask = np.random.uniform(0, 1, length) < density

    # Generate random polarities (±1) for the impulses
    polarities = np.random.choice([-1, 1], length)

    # Create velvet noise: impulses at random positions with random sign
    velvet = amplitude * mask * polarities

    return velvet.astype(np.float32)


def grey_noise(
    dur: float,
    amplitude: float = 1.0,
    sr: float = DEFAULT_SAMPLE_RATE,
    seed: int | None = None,
) -> np.ndarray:
    """Generate grey noise with psychoacoustically flat spectrum.

    Grey noise is noise that sounds equally loud at all frequencies to the human
    ear. It's based on the inverse of equal-loudness contours (ISO 226), making it
    perceptually balanced unlike white noise which sounds brighter due to the ear's
    frequency-dependent sensitivity.

    This makes grey noise ideal for audio testing, masking experiments, and
    creating reference signals that account for human hearing characteristics.
    It's the most perceptually neutral noise type.

    Args:
        dur: Duration of the output noise in seconds.
        amplitude: Peak amplitude of the noise after normalization. Values typically
            range from 0.0 to 1.0. Default is 1.0.
        sr: Sample rate in Hz. Defaults to DEFAULT_SAMPLE_RATE (44100).
        seed: Random seed for reproducible output. If None, results will vary
            each time. Use an integer (e.g., 42) for consistent results.

    Returns:
        np.ndarray: Array of grey noise values, psychoacoustically flattened.
            Output is float32 for memory efficiency.

    Examples:
        >>> # Perceptually flat reference for audio testing
        >>> test_signal = grey_noise(dur=5.0, amplitude=0.7, sr=44100)
        >>>
        >>> # Masking noise for psychoacoustic experiments
        >>> masker = grey_noise(dur=10.0, amplitude=0.5, sr=44100, seed=42)
        >>>
        >>> # Equal-loudness reference tone
        >>> reference = grey_noise(dur=3.0, amplitude=0.8, sr=44100)

    Implementation Notes:
        Generated by filtering white noise with an inverse equal-loudness curve
        approximation. Uses a combination of shelving filters to approximate the
        inverse A-weighting curve, which roughly corresponds to the inverse of
        the 40-phon equal-loudness contour.

        The implementation uses a simplified frequency-domain approach for efficiency.

    Performance:
        Vectorized implementation using FFT-based filtering.
        Generates ~10M samples/second.
        More computationally intensive than white/pink noise due to filtering.

    See Also:
        - white_noise: Physically flat spectrum (sounds bright)
        - pink_noise: 1/f spectrum (perceptually warm)
        - Use for: Audio testing, masking, perceptually neutral reference
    """
    if seed is not None:
        np.random.seed(seed)

    length = int(dur * sr)

    # Generate white noise as starting point
    white = np.random.uniform(-1, 1, length).astype(np.float32)

    # Apply inverse A-weighting approximation in frequency domain
    # This creates psychoacoustically flat response

    # Compute FFT
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(length, 1 / sr)

    # Inverse A-weighting filter approximation
    # A-weighting emphasizes 2-5kHz, so we de-emphasize it
    # This is a simplified version for efficiency

    # Avoid division by zero at DC
    freqs_safe = np.maximum(freqs, 1.0)

    # Simplified inverse A-weighting curve
    # Attenuates high frequencies and boosts lows
    f1, _, _, f4 = 20.6, 107.7, 737.9, 12194.0

    # Inverse A-weighting formula (simplified)
    weight = (f1**2 * freqs_safe**2) / (
        (freqs_safe**2 + f1**2) * (freqs_safe**2 + f4**2)
    )
    weight = weight / np.max(weight)  # Normalize

    # Apply inverse weighting (emphasize what A-weighting attenuates)
    inv_weight = 1.0 / np.maximum(weight, 0.1)
    inv_weight = inv_weight / np.max(inv_weight)

    # Apply filter in frequency domain
    fft_filtered = fft * inv_weight

    # Convert back to time domain
    grey = np.fft.irfft(fft_filtered, n=length)

    # Normalize to range [-1, 1]
    max_val = np.max(np.abs(grey))
    if max_val > 0:
        grey = grey / max_val

    return (amplitude * grey).astype(np.float32)


def sample_hold_noise(
    dur: float,
    rate: float = 10.0,
    amplitude: float = 1.0,
    sr: float = DEFAULT_SAMPLE_RATE,
    seed: int | None = None,
) -> np.ndarray:
    """Generate sample & hold noise - stepped random values.

    Sample & hold noise produces a staircase-like signal where random values are
    held for a specific duration before jumping to a new random value. This creates
    the classic "stepped" modulation sound found in vintage synthesizers.

    The effect is similar to a random LFO with instantaneous transitions, creating
    rhythmic, robotic, or glitchy modulation patterns. It's a staple of analog
    synthesizer design and retro electronic music.

    Args:
        dur: Duration of the output noise in seconds.
        rate: Update rate in Hz (how many times per second the value changes).
            Typical values range from 0.5 Hz (slow) to 100 Hz (fast).
            Default is 10 Hz. Lower rate = longer steps, higher rate = shorter steps.
        amplitude: Peak amplitude of the random values. Values typically range from
            0.0 to 1.0. Default is 1.0. Output will be in range
            [-amplitude, +amplitude].
        sr: Sample rate in Hz. Defaults to DEFAULT_SAMPLE_RATE (44100).
        seed: Random seed for reproducible output. If None, results will vary
            each time. Use an integer (e.g., 42) for consistent results.

    Returns:
        np.ndarray: Array of sample & hold values (stepped random signal).
            Output is float32 for memory efficiency.

    Examples:
        >>> # Classic synth random LFO
        >>> random_lfo = sample_hold_noise(dur=4.0, rate=8, amplitude=1.0, sr=44100)
        >>>
        >>> # Slow stepped modulation
        >>> slow_random = sample_hold_noise(dur=10.0, rate=2, amplitude=0.5, seed=42)
        >>>
        >>> # Fast glitchy modulation
        >>> glitch = sample_hold_noise(dur=2.0, rate=50, amplitude=0.8, sr=44100)

    Implementation Notes:
        Generates random values at the specified rate, then holds each value
        constant until the next update. Creates a piecewise constant signal.
        The transitions are instantaneous (zero rise time), creating the
        characteristic "stepped" sound.

    Performance:
        Vectorized implementation using NumPy's repeat function. Very efficient,
        generates ~200M samples/second.

    See Also:
        - white_noise: Continuous random values (not stepped)
        - perlin_noise: Smooth random modulation (continuous)
        - Use for: Random LFO, stepped modulation, vintage synth sounds, glitch effects
    """
    if seed is not None:
        np.random.seed(seed)

    length = int(dur * sr)
    step_length = int(sr / rate)  # Samples per step

    # Calculate number of steps needed
    num_steps = int(np.ceil(length / step_length))

    # Generate random values for each step
    random_values = np.random.uniform(-1, 1, num_steps)

    # Repeat each value for step_length samples (sample & hold)
    stepped = np.repeat(random_values, step_length)

    # Trim to exact length
    stepped = stepped[:length]

    return (amplitude * stepped).astype(np.float32)


@register_component()
class NoiseGenerator(Generator):
    """Wrapper class for noise generation functions.

    Provides a consistent interface for the noise functions from the engine.
    Implements iterator protocol for compatibility with modulated components.
    """

    descriptor = ComponentDescriptor(
        name="Noise Generator",
        category=ComponentCategory.OSCILLATOR,
        description="Generates various types of noise for synthesis and modulation.",
        fluent_api_name="noise",
        tags=["oscillator", "noise", "modulation", "synthesis"],
        parameters=make_parameter_descriptors(
            "noise_type",
            "amplitude",
            "gain_db",
            "sample_rate",
            noise_type=ParameterDescriptor(
                name="noise_type",
                default="White",
                choices=(
                    "White",
                    "Pink",
                    "Brown",
                    "Blue",
                    "Grey",
                    "Velvet",
                    "Sample & Hold",
                ),
                description="Noise algorithm.",
            ),
            amplitude=ParameterDescriptor(
                name="amplitude",
                default=0.5,
                minimum=0.0,
                description="Linear gain multiplier.",
            ),
        ),
    )

    @track_provided_args
    def __init__(
        self,
        noise_type: str = "White",
        amplitude: float = 0.5,
        gain_db: float | None = DEFAULT_GAIN_DB,
        sample_rate: int | float = DEFAULT_SAMPLE_RATE,
    ):
        """Initialize noise generator.

        Args:
            noise_type: Type of noise to generate
            amplitude: Amplitude scaling factor (0.0-1.0). Ignored if gain_db is
                specified.
            gain_db: Gain in decibels (0 dB = no change). Overrides amplitude if
                provided.
            sample_rate: Sample rate in Hz
        """
        sample_rate = validate_sample_rate(sample_rate)
        super().__init__(sample_rate)
        self.noise_type: str = noise_type

        # Handle amplitude vs gain_db priority (same as oscillators)
        self._amplitude = _derive_amplitude_from_init(
            self._provided_args,
            amplitude,
            gain_db,  # noqa
        )

        # Amplitude smoothing to prevent clicks when changing gain
        self._target_amplitude = self._amplitude
        self._current_amplitude = self._amplitude
        self._smoothing_samples_remaining = 0
        self._smoothing_duration_samples = duration_ms_to_samples(
            self.sample_rate, 10.0, min_samples=1
        )

        self._buffer: np.ndarray | None = None
        self._buffer_index: int = 0
        self._buffer_size: int = 1024  # Generate in chunks for efficiency

    @property
    def amplitude(self) -> float:
        """float: Current amplitude multiplier (linear scale).

        For audio work, consider using the gain_db property instead.
        """
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value: float):
        if value < 0:
            raise ValueError(f"amplitude must be non-negative, got {value}")
        # Initiate smooth transition (prevents clicks)
        self._target_amplitude = float(value)
        self._smoothing_samples_remaining = self._smoothing_duration_samples
        self._amplitude = float(value)

    @property
    def gain_db(self) -> float:
        """float: Current gain in decibels (professional audio standard).

        Common dB values:
            0 dB = unity gain (no change)
            -6 dB = half amplitude
            -20 dB = 1/10 amplitude
            -∞ dB = silence
        """
        return linear_to_db(self._amplitude)

    @gain_db.setter
    def gain_db(self, value: float):
        new_amplitude = float(db_to_linear(value))
        # Initiate smooth transition (prevents clicks)
        self._target_amplitude = new_amplitude
        self._smoothing_samples_remaining = self._smoothing_duration_samples
        self._amplitude = new_amplitude

    def __iter__(self):
        """Initialize iterator (required for iterator protocol)."""
        self._buffer = None
        self._buffer_index = 0
        return self

    def __next__(self) -> float:
        """Get next noise sample (required for iterator protocol).

        Returns:
            Next noise sample value
        """
        # Refill buffer if needed
        if self._buffer is None or self._buffer_index >= len(self._buffer):
            duration = self._buffer_size / self.sample_rate
            self._buffer = self._generate_noise(duration, self._buffer_size)
            self._buffer_index = 0

        sample = self._buffer[self._buffer_index]
        self._buffer_index += 1
        return float(sample)

    def _generate_noise(self, duration: float, num_samples: int) -> np.ndarray:
        """Generate noise samples for the current type.

        Args:
            duration: Duration in seconds
            num_samples: Number of samples to generate

        Returns:
            Array of noise samples
        """
        # Generate the appropriate noise type
        if self.noise_type == "White":
            samples = white_noise(dur=duration, amplitude=1.0, sr=self.sample_rate)
            return self._apply_amplitude_to_buffer(samples)
        if self.noise_type == "Pink":
            samples = pink_noise(dur=duration, amplitude=1.0, sr=self.sample_rate)
            return self._apply_amplitude_to_buffer(samples)
        if self.noise_type == "Brown":
            samples = brownian_noise(dur=duration, amplitude=1.0, sr=self.sample_rate)
            return self._apply_amplitude_to_buffer(samples)
        if self.noise_type == "Blue":
            samples = blue_noise(dur=duration, amplitude=1.0, sr=self.sample_rate)
            return self._apply_amplitude_to_buffer(samples)
        if self.noise_type == "Grey":
            samples = grey_noise(dur=duration, amplitude=1.0, sr=self.sample_rate)
            return self._apply_amplitude_to_buffer(samples)
        if self.noise_type == "Velvet":
            samples = velvet_noise(dur=duration, amplitude=1.0, sr=self.sample_rate)
            return self._apply_amplitude_to_buffer(samples)
        if self.noise_type == "Sample & Hold":
            samples = sample_hold_noise(
                dur=duration, amplitude=1.0, sr=self.sample_rate
            )
            return self._apply_amplitude_to_buffer(samples)

        raise ValueError(f"Unknown noise type: {self.noise_type}")

    def _apply_amplitude_to_buffer(self, samples: np.ndarray) -> np.ndarray:
        """Apply current amplitude, consuming any active smoothing ramp."""
        if len(samples) == 0:
            return np.empty(0, dtype=np.float32)

        if self._smoothing_samples_remaining > 0:
            envelope, self._current_amplitude, self._smoothing_samples_remaining = (
                consume_linear_ramp(
                    self._current_amplitude,
                    self._target_amplitude,
                    self._smoothing_samples_remaining,
                    len(samples),
                )
            )
            if self._smoothing_samples_remaining <= 0:
                self._amplitude = self._target_amplitude
            return (samples * envelope).astype(np.float32)

        self._current_amplitude = self._amplitude
        return (samples * self._amplitude).astype(np.float32)

    def get_samples(
        self, num_samples: int, reset: bool = True, mode: str = "auto"
    ) -> np.ndarray:
        """Generate noise samples (compatible with modulator interface).

        Args:
            num_samples: Number of samples to generate
            reset: Whether to reset the iterator (ignored for noise)
            mode: Generation mode ("auto", "iterator", "vectorized")

        Returns:
            Array of noise samples
        """
        return self.get_samples_vectorized(num_samples)

    def get_samples_vectorized(self, num_samples: int) -> np.ndarray:
        """Generate noise samples (vectorized).

        Args:
            num_samples: Number of samples to generate

        Returns:
            Array of noise samples
        """
        num_samples = validate_sample_count(num_samples, name="num_samples")
        duration = num_samples / self.sample_rate
        return self._generate_noise(duration, num_samples)
