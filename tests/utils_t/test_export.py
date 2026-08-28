"""Tests for offline WAV bounce helpers."""

from __future__ import annotations

import numpy as np

from sonicrack.runtime.export import bounce_output_module, write_wav


def test_write_wav_peak_normalizes_and_writes(tmp_path) -> None:
    hot = np.column_stack(
        (np.full(32, 2.0, dtype=np.float32), np.full(32, -2.0, dtype=np.float32))
    )
    path = write_wav(tmp_path / "bounce", hot, sample_rate=8000)
    assert path.suffix == ".wav"
    assert path.is_file()
    from scipy.io import wavfile

    rate, data = wavfile.read(str(path))
    assert rate == 8000
    assert data.shape[0] == 32
    assert float(np.max(np.abs(data))) <= 1.0 + 1e-5


class _FakeOutput:
    """Duck-typed OutputModule for bounce tests (no Qt / audio device)."""

    def __init__(self) -> None:
        self.audio_output = type("AudioOutput", (), {"is_playing": True})()
        self.stopped = False
        self.started = False

    def render_offline(self, num_frames: int) -> np.ndarray:
        return np.full(num_frames, 0.25, dtype=np.float32)

    def stop_playback(self, graceful: bool = True) -> None:
        del graceful
        self.stopped = True
        self.audio_output.is_playing = False

    def start_playback(self) -> bool:
        self.started = True
        self.audio_output.is_playing = True
        return True


def test_bounce_output_module_upmixes_stereo_and_restores_playback() -> None:
    fake = _FakeOutput()
    audio = bounce_output_module(fake, 0.05, buffer_size=32)
    assert audio.ndim == 2
    assert audio.shape[1] == 2
    assert audio.shape[0] > 32
    np.testing.assert_allclose(audio[:, 0], audio[:, 1])
    assert fake.stopped
    assert fake.started
