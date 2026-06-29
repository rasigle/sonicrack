"""Test cases for the AudioOutput module."""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.audio_io import (
    DEFAULT_FADEIN_DURATION_MS,
    DEFAULT_FADEOUT_DURATION_MS,
    AudioOutput,
)


class TestAudioOutputInitialization(unittest.TestCase):
    """Test AudioOutput initialization."""

    def test_default_initialization(self):
        """Test AudioOutput with default parameters."""
        audio = AudioOutput()

        self.assertEqual(audio.sample_rate, 44100)
        self.assertEqual(audio.buffer_size, 2048)
        self.assertIsNone(audio.audio_callback)
        self.assertFalse(audio.is_playing)
        self.assertIsNone(audio.stream)
        self.assertEqual(audio.master_volume, 0.7)

    def test_custom_initialization(self):
        """Test AudioOutput with custom parameters."""

        def dummy_callback(n):
            return np.zeros((n, 2))

        audio = AudioOutput(
            sample_rate=48000, buffer_size=1024, audio_callback=dummy_callback
        )

        self.assertEqual(audio.sample_rate, 48000)
        self.assertEqual(audio.buffer_size, 1024)
        self.assertIs(audio.audio_callback, dummy_callback)

    def test_fade_calculations(self):
        """Test that fade sample counts are calculated correctly."""
        audio = AudioOutput(sample_rate=44100)

        expected_fade_out = int(44100 * DEFAULT_FADEOUT_DURATION_MS / 1000)
        expected_fade_in = int(44100 * DEFAULT_FADEIN_DURATION_MS / 1000)

        self.assertEqual(audio.fade_out_total_samples, expected_fade_out)
        self.assertEqual(audio.fade_in_total_samples, expected_fade_in)

    def test_volume_smoothing_initialization(self):
        """Test that volume smoothing parameters are initialized."""
        audio = AudioOutput()

        self.assertEqual(audio._target_master_volume, 0.7)
        self.assertEqual(audio._current_master_volume, 0.7)
        self.assertEqual(audio._master_volume_smoothing_samples, 0)
        self.assertEqual(audio._master_volume_smoothing_duration, 441)  # 10ms @ 44.1kHz


class TestAudioOutputCallbacks(unittest.TestCase):
    """Test audio callback functionality."""

    def test_set_audio_callback(self):
        """Test setting audio callback."""
        audio = AudioOutput()

        def new_callback(num_frames: int):
            return np.ones((num_frames, 2))

        audio.set_audio_callback(new_callback)
        self.assertIs(audio.audio_callback, new_callback)

    def test_callback_execution(self):
        """Test that callback is executed correctly."""
        call_count = [0]

        def test_callback(num_frames: int):
            call_count[0] += 1
            return np.asarray(np.random.randn(num_frames, 2))

        audio = AudioOutput(audio_callback=test_callback)

        # Simulate callback execution
        result = audio.audio_callback(1024)

        self.assertEqual(call_count[0], 1)
        self.assertEqual(result.shape, (1024, 2))


class TestSampleRateAndBufferSize(unittest.TestCase):
    """Test sample rate and buffer size configuration."""

    def test_set_sample_rate_not_playing(self):
        """Test changing sample rate when not playing."""
        audio = AudioOutput(sample_rate=44100)

        audio.set_sample_rate(48000)

        self.assertEqual(audio.sample_rate, 48000)
        self.assertFalse(audio.is_playing)

    def test_set_buffer_size_not_playing(self):
        """Test changing buffer size when not playing."""
        audio = AudioOutput(buffer_size=2048)

        audio.set_buffer_size(1024)

        self.assertEqual(audio.buffer_size, 1024)
        self.assertFalse(audio.is_playing)

    @patch("src.audio_io.output.sd.OutputStream")
    def test_set_sample_rate_while_playing(self, mock_stream_class):
        """Test changing sample rate restarts stream if playing."""
        audio = AudioOutput(sample_rate=44100)

        # Start playback
        mock_stream = Mock()
        mock_stream_class.return_value = mock_stream
        audio.start_playback()

        # Change sample rate
        audio.set_sample_rate(48000)

        # Should have stopped and restarted
        self.assertTrue(mock_stream.stop.called)
        self.assertTrue(mock_stream.close.called)
        self.assertEqual(audio.sample_rate, 48000)

    @patch("src.audio_io.output.sd.OutputStream")
    def test_set_buffer_size_while_playing(self, mock_stream_class):
        """Test changing buffer size restarts stream if playing."""
        audio = AudioOutput(buffer_size=2048)

        # Start playback
        mock_stream = Mock()
        mock_stream_class.return_value = mock_stream
        audio.start_playback()

        # Change buffer size
        audio.set_buffer_size(1024)

        # Should have stopped and restarted
        self.assertTrue(mock_stream.stop.called)
        self.assertTrue(mock_stream.close.called)
        self.assertEqual(audio.buffer_size, 1024)

    def test_recalculate_fade_samples(self):
        """Test fade sample recalculation on sample rate change."""
        audio = AudioOutput(sample_rate=44100)

        original_fade_out = audio.fade_out_total_samples
        original_fade_in = audio.fade_in_total_samples

        audio.set_sample_rate(88200)  # Double sample rate

        # Fade samples should approximately double
        self.assertAlmostEqual(
            audio.fade_out_total_samples / original_fade_out, 2.0, places=1
        )
        self.assertAlmostEqual(
            audio.fade_in_total_samples / original_fade_in, 2.0, places=1
        )


class TestVolumeControl(unittest.TestCase):
    """Test master volume control."""

    def test_set_master_volume_normal(self):
        """Test setting master volume within range."""
        audio = AudioOutput()

        audio.set_master_volume(0.5)

        self.assertEqual(audio.master_volume, 0.5)
        self.assertEqual(audio._target_master_volume, 0.5)
        self.assertGreater(audio._master_volume_smoothing_samples, 0)

    def test_set_master_volume_clipping_high(self):
        """Test volume clipping at upper bound."""
        audio = AudioOutput()

        audio.set_master_volume(1.5)

        self.assertEqual(audio.master_volume, 1.0)
        self.assertEqual(audio._target_master_volume, 1.0)

    def test_set_master_volume_clipping_low(self):
        """Test volume clipping at lower bound."""
        audio = AudioOutput()

        audio.set_master_volume(-0.5)

        self.assertEqual(audio.master_volume, 0.0)
        self.assertEqual(audio._target_master_volume, 0.0)

    def test_volume_smoothing_activated(self):
        """Test that volume smoothing is activated on change."""
        audio = AudioOutput()
        audio._master_volume_smoothing_samples = 0

        audio.set_master_volume(0.3)

        self.assertEqual(
            audio._master_volume_smoothing_samples,
            audio._master_volume_smoothing_duration,
        )


class TestPlaybackControl(unittest.TestCase):
    """Test playback start/stop functionality."""

    @patch("src.audio_io.output.sd.OutputStream")
    def test_start_playback(self, mock_stream_class):
        """Test starting playback."""
        mock_stream = Mock()
        mock_stream_class.return_value = mock_stream

        audio = AudioOutput(sample_rate=44100, buffer_size=1024)
        audio.start_playback()

        self.assertTrue(audio.is_playing)
        self.assertTrue(audio.is_fading_in)
        self.assertGreater(audio.fade_in_samples_remaining, 0)
        self.assertFalse(audio.post_fade_silence)

        # Verify stream was created correctly
        mock_stream_class.assert_called_once_with(
            samplerate=44100,
            channels=2,
            blocksize=1024,
            callback=audio._sounddevice_callback,
            dtype=np.float32,
        )
        mock_stream.start.assert_called_once()

    @patch("src.audio_io.output.sd.OutputStream")
    def test_start_playback_already_playing(self, mock_stream_class):
        """Test starting playback when already playing."""
        mock_stream = Mock()
        mock_stream_class.return_value = mock_stream

        audio = AudioOutput()
        audio.start_playback()

        # Reset mock
        mock_stream_class.reset_mock()

        # Try to start again
        audio.start_playback()

        # Should not create new stream
        mock_stream_class.assert_not_called()

    @patch("src.audio_io.output.sd.OutputStream")
    @patch("time.sleep")
    def test_stop_playback(self, mock_sleep, mock_stream_class):
        """Test stopping playback."""
        mock_stream = Mock()
        mock_stream_class.return_value = mock_stream

        audio = AudioOutput()
        audio.start_playback()
        audio.stop_playback()

        self.assertFalse(audio.is_playing)
        self.assertIsNone(audio.stream)
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()

        # Should have waited for fade-out
        mock_sleep.assert_called_once()

    def test_stop_playback_not_playing(self):
        """Test stopping playback when not playing."""
        audio = AudioOutput()

        # Should not raise error
        audio.stop_playback()

        self.assertFalse(audio.is_playing)

    @patch("src.audio_io.output.sd.OutputStream")
    @patch("time.sleep")
    def test_cleanup(self, mock_sleep, mock_stream_class):
        """Test cleanup method."""
        _ = mock_sleep
        mock_stream = Mock()
        mock_stream_class.return_value = mock_stream

        audio = AudioOutput()
        audio.start_playback()
        audio.cleanup()

        self.assertFalse(audio.is_playing)
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()


class TestSoundDeviceCallback(unittest.TestCase):
    """Test the sounddevice callback implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.audio = AudioOutput(sample_rate=44100, buffer_size=1024)

    def test_callback_no_audio_callback(self):
        """Test callback when no audio_callback is set."""
        outdata = np.zeros((1024, 2), dtype=np.float32)

        self.audio._sounddevice_callback(outdata, 1024, None, None)

        # Should fill with silence
        np.testing.assert_array_equal(outdata, 0)

    def test_callback_with_audio_callback_returns_none(self):
        """Test callback when audio_callback returns None."""
        self.audio.audio_callback = lambda n: None
        outdata = np.zeros((1024, 2), dtype=np.float32)

        self.audio._sounddevice_callback(outdata, 1024, None, None)

        # Should fill with silence
        np.testing.assert_array_equal(outdata, 0)

    def test_callback_mono_to_stereo_conversion(self):
        """Test that mono signal is converted to stereo."""
        # Mono signal
        self.audio.audio_callback = lambda n: np.ones(n)
        outdata = np.zeros((1024, 2), dtype=np.float32)

        self.audio._sounddevice_callback(outdata, 1024, None, None)

        # Should have stereo output
        self.assertEqual(outdata.shape, (1024, 2))
        # Both channels should be equal
        np.testing.assert_array_almost_equal(outdata[:, 0], outdata[:, 1])

    def test_callback_stereo_passthrough(self):
        """Test that stereo signal passes through correctly."""
        # Stereo signal (scaled to stay within bounds after volume)
        test_signal = np.random.randn(1024, 2) * 0.5
        self.audio.audio_callback = lambda n: test_signal.copy()
        outdata = np.zeros((1024, 2), dtype=np.float32)

        self.audio._sounddevice_callback(outdata, 1024, None, None)

        # Should match (with volume applied and clipping)
        expected = test_signal * self.audio._target_master_volume
        expected = np.clip(expected, -1.0, 1.0)
        np.testing.assert_array_almost_equal(outdata, expected.astype(np.float32))

    def test_callback_applies_master_volume(self):
        """Test that master volume is applied."""
        self.audio.audio_callback = lambda n: np.ones((n, 2))
        self.audio.set_master_volume(0.5)
        self.audio._master_volume_smoothing_samples = 0  # Disable smoothing for test

        outdata = np.zeros((1024, 2), dtype=np.float32)
        self.audio._sounddevice_callback(outdata, 1024, None, None)

        # All samples should be scaled by volume
        np.testing.assert_array_almost_equal(outdata, 0.5)

    def test_callback_clipping(self):
        """Test that output is clipped to [-1, 1]."""
        # Generate signal exceeding bounds
        self.audio.audio_callback = lambda n: np.ones((n, 2)) * 2.0
        self.audio._master_volume_smoothing_samples = 0

        outdata = np.zeros((1024, 2), dtype=np.float32)
        self.audio._sounddevice_callback(outdata, 1024, None, None)

        # All values should be clipped to 1.0
        self.assertTrue(np.all(outdata <= 1.0))
        self.assertTrue(np.all(outdata >= -1.0))

    def test_callback_volume_smoothing(self):
        """Test volume smoothing in callback."""
        self.audio.audio_callback = lambda n: np.ones((n, 2))
        self.audio._current_master_volume = 0.0
        self.audio._target_master_volume = 1.0
        self.audio._master_volume_smoothing_samples = 100

        outdata = np.zeros((1024, 2), dtype=np.float32)
        self.audio._sounddevice_callback(outdata, 1024, None, None)

        # First samples should be quieter than last samples (fade in)
        self.assertLess(outdata[0, 0], outdata[99, 0])

        # Smoothing samples should be decremented
        self.assertEqual(self.audio._master_volume_smoothing_samples, 0)

    def test_callback_fade_in(self):
        """Test fade-in behavior."""
        self.audio.audio_callback = lambda n: np.ones((n, 2))
        self.audio.is_fading_in = True
        self.audio.fade_in_samples_remaining = 100
        self.audio.fade_in_total_samples = 100
        self.audio._master_volume_smoothing_samples = 0

        outdata = np.zeros((1024, 2), dtype=np.float32)
        self.audio._sounddevice_callback(outdata, 1024, None, None)

        # First samples should be quieter (fade in)
        self.assertLess(outdata[0, 0], outdata[99, 0])

        # Fade-in should be complete
        self.assertFalse(self.audio.is_fading_in)
        self.assertEqual(self.audio.fade_in_samples_remaining, 0)

    def test_callback_fade_out(self):
        """Test fade-out behavior."""
        self.audio.audio_callback = lambda n: np.ones((n, 2))
        self.audio.is_fading_out = True
        self.audio.fade_out_samples_remaining = 100
        self.audio.fade_out_total_samples = 100
        self.audio._master_volume_smoothing_samples = 0

        outdata = np.zeros((1024, 2), dtype=np.float32)
        self.audio._sounddevice_callback(outdata, 1024, None, None)

        # First samples should be louder (fade out)
        self.assertGreater(outdata[0, 0], outdata[99, 0])

        # Post-fade silence mode should be active
        self.assertTrue(self.audio.post_fade_silence)

        # Remaining samples should be silence
        np.testing.assert_array_equal(outdata[100:], 0)

    def test_callback_post_fade_silence(self):
        """Test post-fade silence mode."""
        self.audio.audio_callback = lambda n: np.ones((n, 2))
        self.audio.post_fade_silence = True

        outdata = np.ones((1024, 2), dtype=np.float32)  # Start with non-zero
        self.audio._sounddevice_callback(outdata, 1024, None, None)

        # Should be all silence
        np.testing.assert_array_equal(outdata, 0)

    def test_callback_exception_handling(self):
        """Test that exceptions in callback are handled gracefully."""

        def failing_callback(n):
            raise RuntimeError("Test error")

        self.audio.audio_callback = failing_callback
        outdata = np.ones((1024, 2), dtype=np.float32)

        # Should not raise, should fill with silence
        self.audio._sounddevice_callback(outdata, 1024, None, None)

        np.testing.assert_array_equal(outdata, 0)
        self.assertEqual(self.audio.callback_error_count, 1)
        self.assertIsInstance(self.audio.last_callback_error, RuntimeError)

    def test_callback_status_is_recorded(self):
        """Test that callback status diagnostics are recorded without raising."""
        status = object()
        self.audio.audio_callback = lambda n: np.zeros((n, 2), dtype=np.float32)
        outdata = np.ones((1024, 2), dtype=np.float32)

        self.audio._sounddevice_callback(outdata, 1024, None, status)

        self.assertEqual(self.audio.callback_status_count, 1)
        self.assertIs(self.audio.last_callback_status, status)

    def test_callback_reuses_ramp_buffer_for_steady_state_fades(self):
        """Test that fade/volume ramps reuse the preallocated ramp buffer."""
        self.audio.audio_callback = lambda n: np.ones((n, 2), dtype=np.float32)
        self.audio._current_master_volume = 0.0
        self.audio._target_master_volume = 1.0
        self.audio._master_volume_smoothing_samples = 64
        ramp_id = id(self.audio._ramp_buffer)

        outdata = np.zeros((1024, 2), dtype=np.float32)
        self.audio._sounddevice_callback(outdata, 1024, None, None)

        self.assertEqual(id(self.audio._ramp_buffer), ramp_id)

    def test_callback_copies_short_signal_into_output_buffer(self):
        """Test short callback output leaves the rest of the buffer silent."""
        self.audio.audio_callback = lambda n: np.ones((10, 2), dtype=np.float32)
        self.audio._master_volume_smoothing_samples = 0
        outdata = np.full((1024, 2), 99.0, dtype=np.float32)

        self.audio._sounddevice_callback(outdata, 1024, None, None)

        np.testing.assert_array_almost_equal(outdata[:10], 0.7)
        np.testing.assert_array_equal(outdata[10:], 0)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for common usage scenarios."""

    @patch("src.audio_io.output.sd.OutputStream")
    @patch("time.sleep")
    def test_full_playback_cycle(self, mock_sleep, mock_stream_class):
        """Test complete playback start -> change settings -> stop cycle."""
        _ = mock_sleep
        mock_stream = Mock()
        mock_stream_class.return_value = mock_stream

        def test_callback(n):
            return np.random.randn(n, 2)

        audio = AudioOutput(
            sample_rate=44100, buffer_size=2048, audio_callback=test_callback
        )

        # Start playback
        audio.start_playback()
        self.assertTrue(audio.is_playing)

        # Change volume
        audio.set_master_volume(0.5)
        self.assertEqual(audio.master_volume, 0.5)

        # Stop playback
        audio.stop_playback()
        self.assertFalse(audio.is_playing)

    @patch("src.audio_io.output.sd.OutputStream")
    def test_multiple_sample_rate_changes(self, mock_stream_class):
        """Test multiple sample rate changes."""
        mock_stream = Mock()
        mock_stream_class.return_value = mock_stream

        audio = AudioOutput(sample_rate=44100)

        rates = [48000, 88200, 96000, 22050, 44100]
        for rate in rates:
            audio.set_sample_rate(rate)
            self.assertEqual(audio.sample_rate, rate)

    def test_fade_sample_consistency(self):
        """Test that fade samples are consistent across sample rate changes."""
        audio = AudioOutput(sample_rate=44100)

        # Store duration
        fade_out_ms = audio.fade_out_duration_ms
        fade_in_ms = audio.fade_in_duration_ms

        # Change sample rate
        audio.set_sample_rate(48000)

        # Durations should be same, but sample counts different
        self.assertEqual(audio.fade_out_duration_ms, fade_out_ms)
        self.assertEqual(audio.fade_in_duration_ms, fade_in_ms)

        # Sample counts should scale with sample rate
        expected_fade_out = int(48000 * fade_out_ms / 1000)
        expected_fade_in = int(48000 * fade_in_ms / 1000)

        self.assertEqual(audio.fade_out_total_samples, expected_fade_out)
        self.assertEqual(audio.fade_in_total_samples, expected_fade_in)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_zero_buffer_size(self):
        """Test handling of zero buffer size."""
        audio = AudioOutput(buffer_size=0)
        self.assertEqual(audio.buffer_size, 0)

    def test_very_large_buffer_size(self):
        """Test handling of very large buffer size."""
        audio = AudioOutput(buffer_size=16384)
        self.assertEqual(audio.buffer_size, 16384)

    def test_low_sample_rate(self):
        """Test handling of low sample rate."""
        audio = AudioOutput(sample_rate=8000)
        self.assertEqual(audio.sample_rate, 8000)
        # Fade samples should still be calculated
        self.assertGreater(audio.fade_out_total_samples, 0)

    def test_high_sample_rate(self):
        """Test handling of high sample rate."""
        audio = AudioOutput(sample_rate=192000)
        self.assertEqual(audio.sample_rate, 192000)
        # Fade samples should scale appropriately
        self.assertGreater(audio.fade_out_total_samples, 0)

    @patch("src.audio_io.output.sd.OutputStream")
    def test_stream_creation_failure(self, mock_stream_class):
        """Test handling of stream creation failure."""
        mock_stream_class.side_effect = RuntimeError("Failed to create stream")

        audio = AudioOutput()

        with self.assertRaises(RuntimeError):
            audio.start_playback()

        # Should not be marked as playing
        self.assertFalse(audio.is_playing)


if __name__ == "__main__":
    unittest.main(verbosity=2)
