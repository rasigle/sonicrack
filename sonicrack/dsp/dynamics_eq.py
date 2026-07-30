"""Limiter and parametric EQ processors (NumPy, mono)."""

from __future__ import annotations

import numpy as np


class Limiter:
    """Simple look-ahead-free peak limiter with makeup gain."""

    def __init__(
        self,
        threshold: float = 0.9,
        release_ms: float = 50.0,
        makeup: float = 1.0,
        sample_rate: int = 44100,
    ) -> None:
        self.threshold = threshold
        self.release_ms = release_ms
        self.makeup = makeup
        self.sample_rate = sample_rate
        self._envelope = 0.0

    def reset(self) -> None:
        self._envelope = 0.0

    def process(self, samples: np.ndarray) -> np.ndarray:
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x

        thr = max(1e-4, float(self.threshold))
        release = max(1.0, float(self.release_ms))
        # Attack nearly instant; release from envelope of peak excess.
        release_coef = np.exp(-1.0 / (0.001 * release * self.sample_rate))
        env = self._envelope
        makeup = float(self.makeup)

        out = np.empty_like(x)
        for i, sample in enumerate(x):
            level = abs(float(sample))
            if level > env:
                env = level
            else:
                env = release_coef * env + (1.0 - release_coef) * level
            gain = 1.0 if env <= thr else thr / env
            out[i] = sample * gain * makeup

        self._envelope = env
        return out

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        return self.process(samples)


class ParametricEQ:
    """Single peaking (bell) EQ band with optional low/high shelf modes."""

    def __init__(
        self,
        frequency: float = 1000.0,
        gain_db: float = 0.0,
        q: float = 1.0,
        mode: str = "peak",
        sample_rate: int = 44100,
    ) -> None:
        self.frequency = frequency
        self.gain_db = gain_db
        self.q = q
        self.mode = mode
        self.sample_rate = sample_rate
        self._x1 = 0.0
        self._x2 = 0.0
        self._y1 = 0.0
        self._y2 = 0.0
        self._b0 = 1.0
        self._b1 = 0.0
        self._b2 = 0.0
        self._a1 = 0.0
        self._a2 = 0.0
        self._update_coeffs()

    def reset(self) -> None:
        self._x1 = self._x2 = 0.0
        self._y1 = self._y2 = 0.0

    def _update_coeffs(self) -> None:
        sr = float(self.sample_rate)
        f0 = float(np.clip(self.frequency, 20.0, sr * 0.45))
        q = max(0.1, float(self.q))
        a = 10.0 ** (float(self.gain_db) / 40.0)
        w0 = 2.0 * np.pi * f0 / sr
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / (2.0 * q)
        mode = (self.mode or "peak").lower()

        if mode == "low_shelf":
            two_sqrt_a_alpha = 2.0 * np.sqrt(a) * alpha
            b0 = a * ((a + 1) - (a - 1) * cos_w0 + two_sqrt_a_alpha)
            b1 = 2.0 * a * ((a - 1) - (a + 1) * cos_w0)
            b2 = a * ((a + 1) - (a - 1) * cos_w0 - two_sqrt_a_alpha)
            a0 = (a + 1) + (a - 1) * cos_w0 + two_sqrt_a_alpha
            a1 = -2.0 * ((a - 1) + (a + 1) * cos_w0)
            a2 = (a + 1) + (a - 1) * cos_w0 - two_sqrt_a_alpha
        elif mode == "high_shelf":
            two_sqrt_a_alpha = 2.0 * np.sqrt(a) * alpha
            b0 = a * ((a + 1) + (a - 1) * cos_w0 + two_sqrt_a_alpha)
            b1 = -2.0 * a * ((a - 1) + (a + 1) * cos_w0)
            b2 = a * ((a + 1) + (a - 1) * cos_w0 - two_sqrt_a_alpha)
            a0 = (a + 1) - (a - 1) * cos_w0 + two_sqrt_a_alpha
            a1 = 2.0 * ((a - 1) - (a + 1) * cos_w0)
            a2 = (a + 1) - (a - 1) * cos_w0 - two_sqrt_a_alpha
        else:
            # Peaking
            b0 = 1.0 + alpha * a
            b1 = -2.0 * cos_w0
            b2 = 1.0 - alpha * a
            a0 = 1.0 + alpha / a
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha / a

        self._b0 = b0 / a0
        self._b1 = b1 / a0
        self._b2 = b2 / a0
        self._a1 = a1 / a0
        self._a2 = a2 / a0

    def process(self, samples: np.ndarray) -> np.ndarray:
        self._update_coeffs()
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x

        b0, b1, b2 = self._b0, self._b1, self._b2
        a1, a2 = self._a1, self._a2
        x1, x2 = self._x1, self._x2
        y1, y2 = self._y1, self._y2

        out = np.empty_like(x)
        for i, xn in enumerate(x):
            yn = b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2, x1 = x1, xn
            y2, y1 = y1, yn
            out[i] = yn

        self._x1, self._x2 = x1, x2
        self._y1, self._y2 = y1, y2
        return out.astype(np.float32, copy=False)

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        return self.process(samples)
