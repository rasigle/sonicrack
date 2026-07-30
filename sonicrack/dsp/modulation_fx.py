"""Chorus and phaser processors (NumPy, mono)."""

from __future__ import annotations

import numpy as np


class Chorus:
    """Simple modulated delay chorus with dry/wet mix."""

    def __init__(
        self,
        rate_hz: float = 0.8,
        depth_ms: float = 3.5,
        delay_ms: float = 12.0,
        mix: float = 0.5,
        feedback: float = 0.15,
        sample_rate: int = 44100,
    ) -> None:
        self.rate_hz = rate_hz
        self.depth_ms = depth_ms
        self.delay_ms = delay_ms
        self.mix = mix
        self.feedback = feedback
        self.sample_rate = sample_rate
        max_delay = int(sample_rate * 0.05) + 8
        self._buffer = np.zeros(max_delay, dtype=np.float32)
        self._write_pos = 0
        self._lfo_phase = 0.0

    def reset(self) -> None:
        self._buffer.fill(0.0)
        self._write_pos = 0
        self._lfo_phase = 0.0

    def process(self, samples: np.ndarray) -> np.ndarray:
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x

        n = x.size
        sr = float(self.sample_rate)
        buf = self._buffer
        buf_len = len(buf)
        write = self._write_pos
        phase = self._lfo_phase
        phase_inc = 2.0 * np.pi * max(0.01, float(self.rate_hz)) / sr
        base_delay = max(1.0, float(self.delay_ms) * 0.001 * sr)
        depth = max(0.0, float(self.depth_ms) * 0.001 * sr)
        mix = float(np.clip(self.mix, 0.0, 1.0))
        fb = float(np.clip(self.feedback, 0.0, 0.95))

        out = np.empty(n, dtype=np.float32)
        for i in range(n):
            mod = base_delay + depth * np.sin(phase)
            phase += phase_inc
            if phase > 2.0 * np.pi:
                phase -= 2.0 * np.pi

            read_pos = write - mod
            i0 = int(np.floor(read_pos)) % buf_len
            i1 = (i0 + 1) % buf_len
            frac = read_pos - np.floor(read_pos)
            delayed = (1.0 - frac) * buf[i0] + frac * buf[i1]

            dry = float(x[i])
            buf[write] = dry + delayed * fb
            write = (write + 1) % buf_len
            out[i] = dry * (1.0 - mix) + delayed * mix

        self._write_pos = write
        self._lfo_phase = phase
        return out

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        return self.process(samples)


class Phaser:
    """All-pass cascade phaser with LFO and feedback."""

    def __init__(
        self,
        rate_hz: float = 0.4,
        depth: float = 0.7,
        feedback: float = 0.4,
        stages: int = 4,
        mix: float = 0.5,
        sample_rate: int = 44100,
    ) -> None:
        self.rate_hz = rate_hz
        self.depth = depth
        self.feedback = feedback
        self.stages = max(2, min(12, int(stages)))
        self.mix = mix
        self.sample_rate = sample_rate
        self._z = np.zeros(self.stages, dtype=np.float64)
        self._lfo_phase = 0.0
        self._fb_sample = 0.0

    def reset(self) -> None:
        self._z.fill(0.0)
        self._lfo_phase = 0.0
        self._fb_sample = 0.0

    def process(self, samples: np.ndarray) -> np.ndarray:
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x

        n = x.size
        sr = float(self.sample_rate)
        phase = self._lfo_phase
        phase_inc = 2.0 * np.pi * max(0.01, float(self.rate_hz)) / sr
        depth = float(np.clip(self.depth, 0.0, 1.0))
        fb = float(np.clip(self.feedback, 0.0, 0.95))
        mix = float(np.clip(self.mix, 0.0, 1.0))
        z = self._z
        stages = len(z)
        y_fb = self._fb_sample

        out = np.empty(n, dtype=np.float32)
        for i in range(n):
            # Map LFO to all-pass coefficient in a musical range.
            lfo = 0.5 + 0.5 * np.sin(phase)
            phase += phase_inc
            if phase > 2.0 * np.pi:
                phase -= 2.0 * np.pi
            # a in ~[-0.9, -0.1] when depth is high
            a = -0.1 - 0.8 * depth * lfo

            dry = float(x[i])
            y = dry + y_fb * fb
            for s in range(stages):
                x_in = y
                y = -a * x_in + z[s]
                z[s] = x_in + a * y
            y_fb = y
            out[i] = dry * (1.0 - mix) + y * mix

        self._lfo_phase = phase
        self._fb_sample = y_fb
        return out

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        return self.process(samples)
