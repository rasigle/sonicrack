"""Realtime-oriented tests for the audio output callback path."""

import tracemalloc

import numpy as np
import pytest

from src.audio_io import AudioOutput


@pytest.mark.parametrize("buffer_size", [64, 128, 256, 512, 1024, 2048])
def test_callback_reuses_buffers_for_common_buffer_sizes(buffer_size: int):
    """Callback processing should not grow reusable buffers in steady state."""
    source = np.ones((buffer_size, 2), dtype=np.float32)
    outdata = np.zeros((buffer_size, 2), dtype=np.float32)
    audio = AudioOutput(buffer_size=buffer_size, audio_callback=lambda n: source[:n])
    audio._master_volume_smoothing_samples = 0

    ramp_id = id(audio._ramp_buffer)
    index_id = id(audio._ramp_index_buffer)

    for _ in range(16):
        audio._sounddevice_callback(outdata, buffer_size, None, None)

    assert id(audio._ramp_buffer) == ramp_id
    assert id(audio._ramp_index_buffer) == index_id
    assert audio.callback_error_count == 0
    np.testing.assert_array_almost_equal(outdata, 0.7)


def test_callback_grows_reusable_buffers_for_unexpected_larger_buffers():
    """Unexpected larger callback requests should grow buffers once and recover."""
    source = np.ones((256, 2), dtype=np.float32)
    outdata = np.zeros((256, 2), dtype=np.float32)
    audio = AudioOutput(buffer_size=64, audio_callback=lambda n: source[:n])
    audio._current_master_volume = 0.0
    audio._target_master_volume = 1.0
    audio._master_volume_smoothing_samples = 256

    audio._sounddevice_callback(outdata, 256, None, None)

    assert audio._ramp_buffer.shape[0] >= 256
    assert audio._ramp_index_buffer.shape[0] >= 256
    assert audio.callback_error_count == 0


def test_callback_has_bounded_python_allocations_in_steady_state():
    """Track Python allocations after warmup for the common callback path."""
    buffer_size = 512
    source = np.ones((buffer_size, 2), dtype=np.float32)
    outdata = np.zeros((buffer_size, 2), dtype=np.float32)
    audio = AudioOutput(buffer_size=buffer_size, audio_callback=lambda n: source[:n])
    audio._master_volume_smoothing_samples = 0

    for _ in range(32):
        audio._sounddevice_callback(outdata, buffer_size, None, None)

    tracemalloc.start()
    for _ in range(256):
        audio._sounddevice_callback(outdata, buffer_size, None, None)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak_bytes < 8192
