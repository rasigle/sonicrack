import numpy as np
from scipy.io import wavfile


class Sequencer:
    """
    Phase-continuous sine-wave sequencer for N steps (default 16) with per-step tempo.
    - Maintains phase continuity between notes to avoid discontinuities.
    - Applies a short attack and release envelope per note (ms) to avoid clicks.
    - Can render to a numpy array, save as WAV, or play (via IPython Audio if available).
    """

    def __init__(
        self, steps=16, sample_rate=44100, gain=0.25, attack_ms=6.0, release_ms=6.0
    ):
        self.steps = steps
        self.sr = sample_rate
        self.gain = float(gain)
        self.attack_ms = float(attack_ms)
        self.release_ms = float(release_ms)
        self.frequencies = np.zeros(self.steps, dtype=float)
        self.bpm = np.full(self.steps, 120.0, dtype=float)
        self.beats_per_step = np.ones(self.steps, dtype=float)

    def set_notes(self, freqs):
        arr = np.asarray(freqs, dtype=float)
        if arr.size != self.steps:
            raise ValueError(f"Expected {self.steps} notes, got {arr.size}")
        self.frequencies = arr.copy()

    def set_tempo(self, bpm_or_list, beats_per_step=None):
        """Set BPM per step (can be scalar or list). beats_per_step can be scalar or list."""
        bpm = np.asarray(bpm_or_list, dtype=float)
        if bpm.size == 1:
            self.bpm = np.full(self.steps, float(bpm), dtype=float)
        elif bpm.size == self.steps:
            self.bpm = bpm.copy()
        else:
            raise ValueError("bpm_or_list must be scalar or length == steps")
        if beats_per_step is None:
            self.beats_per_step = np.ones(self.steps, dtype=float)
        else:
            bps = np.asarray(beats_per_step, dtype=float)
            if bps.size == 1:
                self.beats_per_step = np.full(self.steps, float(bps), dtype=float)
            elif bps.size == self.steps:
                self.beats_per_step = bps.copy()
            else:
                raise ValueError("beats_per_step must be scalar or length == steps")

    def _note_duration_seconds(self, idx):
        return float(self.beats_per_step[idx]) * 60.0 / float(self.bpm[idx])

    def render(self, sample_type=np.float32):
        """Render the whole sequence as a single continuous waveform (phase-continuous)."""
        durations = [self._note_duration_seconds(i) for i in range(self.steps)]
        samples_per_step = [int(np.round(d * self.sr)) for d in durations]
        total_samples = sum(samples_per_step)
        if total_samples <= 0:
            return np.zeros(0, dtype=sample_type)

        out = np.zeros(total_samples, dtype=np.float32)
        phase = 0.0
        two_pi = 2.0 * np.pi
        idx = 0
        attack_samps = max(1, int(self.attack_ms * 0.001 * self.sr))
        release_samps = max(1, int(self.release_ms * 0.001 * self.sr))

        for step in range(self.steps):
            f = float(self.frequencies[step])
            n = samples_per_step[step]
            if n <= 0:
                continue
            t = np.arange(n)
            if f <= 0.0:
                wave = np.zeros(n, dtype=np.float32)
            else:
                incr = two_pi * f / self.sr
                phases = phase + incr * t
                phases = np.mod(phases, two_pi)
                wave = np.sin(phases).astype(np.float32)
                phase = float((phases[-1] + incr) % two_pi)
            env = np.ones(n, dtype=np.float32)
            a = min(attack_samps, n)
            if a > 0:
                env[:a] = np.linspace(0.0, 1.0, a, endpoint=False)
            r = min(release_samps, n)
            if r > 0:
                env[-r:] = np.linspace(1.0, 0.0, r, endpoint=False)
            out[idx : idx + n] = wave * env * self.gain
            idx += n

        out = np.nan_to_num(out)
        maxv = np.max(np.abs(out)) if out.size else 0.0
        if maxv > 1.0:
            out = out / maxv
        return out.astype(sample_type)

    def save_wav(self, filename, data):
        i16 = np.int16(np.clip(data, -1.0, 1.0) * 32767)
        wavfile.write(filename, int(self.sr), i16)
        return filename


# prepare 16 frequencies (Hz)
freqs = [
    261.63,
    329.63,
    392.00,
    523.25,
    440.00,
    392.00,
    329.63,
    261.63,
    196.00,
    246.94,
    293.66,
    349.23,
    392.00,
    440.00,
    493.88,
    523.25,
]

seq = Sequencer(steps=16, sample_rate=44100, gain=0.20, attack_ms=6.0, release_ms=8.0)
seq.set_notes(freqs)

# per-step tempo (BPM) — can vary each step
bpms = [90, 95, 100, 110, 120, 140, 160, 180, 180, 160, 140, 120, 110, 100, 95, 90]
seq.set_tempo(bpms)

wave = seq.render()
seq.save_wav("sequencer_output.wav", wave)
# In a notebook you can play with IPython.display.Audio(wave, rate=44100)
