"""Tests for offline WAV bounce helpers."""

from __future__ import annotations

import numpy as np

from sonicrack.runtime.export import write_wav


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
