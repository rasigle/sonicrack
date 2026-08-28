"""Offline bounce of the current patch through the Output module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from sonicrack.config.audio_config import audio_config
from sonicrack.gui.modules.output.output import OutputModule


def bounce_output_module(
    output_module: OutputModule,
    duration_seconds: float,
    *,
    buffer_size: int | None = None,
) -> np.ndarray:
    """Render the Output module's graph to a stereo float32 buffer.

    Stops live playback for the duration of the bounce so the audio callback
    cannot race the offline renderer.
    """
    seconds = max(0.05, float(duration_seconds))
    sample_rate = int(audio_config.sample_rate)
    frames = max(1, int(round(seconds * sample_rate)))
    block = int(buffer_size or audio_config.buffer_size)
    block = max(32, block)

    was_playing = bool(
        getattr(output_module, "audio_output", None)
        and output_module.audio_output.is_playing
    )
    if was_playing:
        output_module.stop_playback(graceful=False)

    chunks: list[np.ndarray] = []
    remaining = frames
    try:
        while remaining > 0:
            n = min(block, remaining)
            chunk = np.asarray(output_module._generate_samples(n), dtype=np.float32)
            if chunk.ndim == 1:
                chunk = np.column_stack((chunk, chunk))
            elif chunk.ndim == 2 and chunk.shape[1] == 1:
                chunk = np.column_stack((chunk[:, 0], chunk[:, 0]))
            chunks.append(chunk[:n, :2])
            remaining -= n
    finally:
        if was_playing:
            output_module.start_playback()

    if not chunks:
        return np.zeros((frames, 2), dtype=np.float32)
    audio = np.concatenate(chunks, axis=0)
    if audio.shape[0] > frames:
        audio = audio[:frames]
    return audio.astype(np.float32, copy=False)


def write_wav(
    path: str | Path,
    audio: np.ndarray,
    sample_rate: int | None = None,
) -> Path:
    """Write stereo or mono float32 audio to a WAV file (peak-safe)."""
    out_path = Path(path)
    if out_path.suffix.lower() != ".wav":
        out_path = out_path.with_suffix(".wav")
    samples = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 1.0:
        samples = samples * np.float32(0.99 / peak)
    rate = int(sample_rate or audio_config.sample_rate)
    wavfile.write(str(out_path), rate, samples)
    return out_path
