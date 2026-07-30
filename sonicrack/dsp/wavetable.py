"""Simple morphable wavetable oscillator."""

from __future__ import annotations

import numpy as np


def _build_tables(table_size: int = 2048) -> np.ndarray:
    """Return stacked tables: sine, triangle, saw, square."""
    phase = np.linspace(0.0, 1.0, table_size, endpoint=False, dtype=np.float64)
    sine = np.sin(2.0 * np.pi * phase)
    triangle = 2.0 * np.abs(2.0 * phase - 1.0) - 1.0
    saw = 2.0 * phase - 1.0
    square = np.where(phase < 0.5, 1.0, -1.0)
    return np.stack([sine, triangle, saw, square], axis=0).astype(np.float32)


class WavetableOscillator:
    """Morphable 4-table oscillator with linear table interpolation."""

    def __init__(
        self,
        frequency: float = 440.0,
        morph: float = 0.0,
        gain: float = 0.5,
        sample_rate: int = 44100,
    ) -> None:
        self.frequency = frequency
        self.morph = morph
        self.gain = gain
        self.sample_rate = sample_rate
        self._tables = _build_tables()
        self._table_size = self._tables.shape[1]
        self._phase = 0.0

    def reset(self) -> None:
        self._phase = 0.0

    def get_samples(self, num_samples: int) -> np.ndarray:
        n = max(0, int(num_samples))
        if n == 0:
            return np.zeros(0, dtype=np.float32)

        sr = max(1, int(self.sample_rate))
        freq = float(np.clip(self.frequency, 0.0, sr * 0.45))
        morph = float(np.clip(self.morph, 0.0, 1.0))
        gain = float(self.gain)
        size = self._table_size
        tables = self._tables
        n_tables = tables.shape[0]

        # Morph position across tables 0..n_tables-1
        pos = morph * (n_tables - 1)
        t0 = int(np.floor(pos))
        t1 = min(t0 + 1, n_tables - 1)
        t_frac = pos - t0

        phase = self._phase
        phase_inc = freq / sr
        out = np.empty(n, dtype=np.float32)
        for i in range(n):
            idx = phase * size
            i0 = int(idx) % size
            i1 = (i0 + 1) % size
            frac = idx - np.floor(idx)
            s0 = (1.0 - frac) * tables[t0, i0] + frac * tables[t0, i1]
            s1 = (1.0 - frac) * tables[t1, i0] + frac * tables[t1, i1]
            out[i] = ((1.0 - t_frac) * s0 + t_frac * s1) * gain
            phase += phase_inc
            if phase >= 1.0:
                phase -= np.floor(phase)

        self._phase = phase
        return out

    def __call__(self, num_samples: int) -> np.ndarray:
        return self.get_samples(num_samples)
