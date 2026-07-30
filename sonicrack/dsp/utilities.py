"""Simple CV/audio utility processors."""

from __future__ import annotations

import numpy as np


class Attenuverter:
    """Scale and offset a signal: ``out = in * amount + offset``."""

    def __init__(self, amount: float = 1.0, offset: float = 0.0) -> None:
        self.amount = amount
        self.offset = offset

    def process(self, samples: np.ndarray) -> np.ndarray:
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        return x * float(self.amount) + float(self.offset)

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        return self.process(samples)


class SignalMult:
    """Multiply two signals (ring mod / VCA-style product)."""

    def __init__(self, amount: float = 1.0) -> None:
        self.amount = amount

    def process(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        x = np.asarray(a, dtype=np.float32).reshape(-1)
        y = np.asarray(b, dtype=np.float32).reshape(-1)
        n = max(x.size, y.size)
        if x.size < n:
            x = np.pad(x, (0, n - x.size))
        if y.size < n:
            y = np.pad(y, (0, n - y.size))
        return (x * y * float(self.amount)).astype(np.float32, copy=False)

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self.process(a, b)
