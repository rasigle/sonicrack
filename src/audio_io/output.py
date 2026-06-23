"""Audio output logic for real-time playback using sounddevice."""

from __future__ import annotations

import logging

import numpy as np
import sounddevice as sd

from src.audio_io.realtime import RealtimeAudioCallback

logger = logging.getLogger(__name__)

DEFAULT_FADEOUT_DURATION_MS = 50
DEFAULT_FADEIN_DURATION_MS = 10
STREAM_ERROR_TYPES = (sd.PortAudioError, OSError, RuntimeError)


class AudioOutput:
    """Manages audio output using sounddevice with configurable sample rate and buffer
    size.

    This class handles the actual audio streaming to the sound device, including:
    - Starting/stopping playback with fade in/out
    - Managing the audio callback
    - Volume control
    - Signal monitoring
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        buffer_size: int = 2048,
        audio_callback: RealtimeAudioCallback | None = None,
    ):
        """Initialize the audio output.

        Args:
            sample_rate: Sample rate in Hz (e.g., 44100, 48000)
            buffer_size: Audio buffer size in samples (e.g., 512, 1024, 2048)
            audio_callback: Callback function that generates audio samples.
                Should accept num_samples and return stereo array.
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.audio_callback = audio_callback

        # Audio state
        self.is_playing: bool = False
        self.stream: sd.OutputStream | None = None
        self.master_volume: float = 0.7

        # Master volume smoothing to prevent clicks
        self._target_master_volume = 0.7
        self._current_master_volume = 0.7
        self._master_volume_smoothing_samples = 0
        self._master_volume_smoothing_duration = 441  # 10ms @ 44.1kHz

        # Fade-out state to prevent clicks on stop
        self.is_fading_out: bool = False
        self.fade_out_samples_remaining: int = 0
        self.fade_out_duration_ms: float = DEFAULT_FADEOUT_DURATION_MS
        self.fade_out_total_samples: int = int(
            self.sample_rate * self.fade_out_duration_ms / 1000
        )

        # Fade-in state to prevent clicks on start
        self.is_fading_in: bool = False
        self.fade_in_samples_remaining: int = 0
        self.fade_in_duration_ms: float = DEFAULT_FADEIN_DURATION_MS
        self.fade_in_total_samples: int = int(
            self.sample_rate * self.fade_in_duration_ms / 1000
        )

        # Track if we're in post-fade silence mode
        self.post_fade_silence: bool = False
        self.callback_error_count: int = 0
        self.callback_status_count: int = 0
        self.last_callback_error: Exception | None = None
        self.last_callback_status = None

        callback_buffer_size = max(self.buffer_size, 1)
        self._ramp_buffer = np.empty(callback_buffer_size, dtype=np.float32)
        self._ramp_index_buffer = np.arange(callback_buffer_size, dtype=np.float32)

    def write(self, samples: np.ndarray):
        """Write audio samples directly to the output stream.

        Args:
            samples: Stereo audio samples to write
        """
        if not self.is_playing or self.stream is None:
            logger.warning("AudioOutput.write() called while not playing")
            return

        try:
            self.stream.write(samples.astype(np.float32))
        except STREAM_ERROR_TYPES as e:
            logger.error(f"Error writing audio samples: {e}", exc_info=True)

    def set_audio_callback(self, callback: RealtimeAudioCallback):
        """Set the audio generation callback.

        Args:
            callback: Function that generates audio samples
        """
        self.audio_callback = callback

    def set_sample_rate(self, sample_rate: int):
        """Change the sample rate (requires restart if playing).

        Args:
            sample_rate: New sample rate in Hz
        """
        was_playing = self.is_playing
        if was_playing:
            self.stop_playback()

        self.sample_rate = sample_rate
        self._recalculate_fade_samples()

        if was_playing:
            self.start_playback()

        logger.info(f"Sample rate changed to {sample_rate} Hz")

    def set_buffer_size(self, buffer_size: int):
        """Change the buffer size (requires restart if playing).

        Args:
            buffer_size: New buffer size in samples
        """
        was_playing = self.is_playing
        if was_playing:
            self.stop_playback()

        self.buffer_size = buffer_size
        self._ensure_callback_buffers(buffer_size)

        if was_playing:
            self.start_playback()

        logger.info(f"Buffer size changed to {buffer_size} samples")

    def _recalculate_fade_samples(self):
        """Recalculate fade sample counts based on current sample rate."""
        self.fade_out_total_samples = int(
            self.sample_rate * self.fade_out_duration_ms / 1000
        )
        self.fade_in_total_samples = int(
            self.sample_rate * self.fade_in_duration_ms / 1000
        )
        self._master_volume_smoothing_duration = int(
            self.sample_rate * 0.01
        )  # 10ms smoothing

    def set_master_volume(self, volume: float):
        """Set the master output volume with smoothing to prevent clicks.

        Args:
            volume: Volume level (0.0 to 1.0)
        """
        new_volume = np.clip(volume, 0.0, 1.0)
        self._target_master_volume = new_volume
        self._master_volume_smoothing_samples = self._master_volume_smoothing_duration
        self.master_volume = new_volume

    def _sounddevice_callback(
        self, outdata: np.ndarray, frames: int, time_info, status
    ):
        """Internal callback for sounddevice stream.

        Args:
            outdata: Output buffer to fill
            frames: Number of frames requested
            time_info: Time information
            status: Stream status
        """
        if status:
            self.callback_status_count += 1
            self.last_callback_status = status

        try:
            outdata.fill(0)

            if self.audio_callback is None:
                return

            samples = self.audio_callback(frames)

            if samples is None:
                return

            self._copy_callback_samples(outdata, samples, frames)

            # Apply master volume smoothing
            if self._master_volume_smoothing_samples > 0:
                num_smooth = min(self._master_volume_smoothing_samples, frames)
                volume_ramp = self._fill_ramp(
                    self._current_master_volume,
                    self._target_master_volume,
                    num_smooth,
                )

                outdata[:num_smooth] *= volume_ramp[:, np.newaxis]
                if frames > num_smooth:
                    outdata[num_smooth:] *= self._target_master_volume

                self._current_master_volume = self._target_master_volume
                self._master_volume_smoothing_samples -= num_smooth
            else:
                outdata *= self._target_master_volume

            # Apply fade-in
            if self.is_fading_in and self.fade_in_samples_remaining > 0:
                num_fade = min(self.fade_in_samples_remaining, frames)
                if num_fade > 0:
                    segment = outdata[:num_fade]
                    fade_start = 1.0 - (
                        self.fade_in_samples_remaining / self.fade_in_total_samples
                    )
                    fade_stop = 1.0 - (
                        (self.fade_in_samples_remaining - num_fade)
                        / self.fade_in_total_samples
                    )
                    fade_curve = self._fill_ramp(
                        fade_start,
                        fade_stop,
                        num_fade,
                    )
                    np.multiply(segment, fade_curve[:, np.newaxis], out=segment)
                    self.fade_in_samples_remaining -= num_fade

                    if self.fade_in_samples_remaining <= 0:
                        self.is_fading_in = False

            # Apply fade-out
            if self.is_fading_out and not self.post_fade_silence:
                if self.fade_out_samples_remaining > 0:
                    num_fade = min(self.fade_out_samples_remaining, frames)
                    if num_fade > 0:
                        segment = outdata[:num_fade]
                        fade_start = (
                            self.fade_out_samples_remaining
                            / self.fade_out_total_samples
                        )
                        fade_stop = (
                            (self.fade_out_samples_remaining - num_fade)
                            / self.fade_out_total_samples
                        )
                        fade_curve = self._fill_ramp(
                            fade_start,
                            fade_stop,
                            num_fade,
                        )
                        np.multiply(segment, fade_curve[:, np.newaxis], out=segment)
                        self.fade_out_samples_remaining -= num_fade

                        if self.fade_out_samples_remaining <= 0:
                            self.post_fade_silence = True

                        if frames > num_fade:
                            outdata[num_fade:] = 0
                else:
                    outdata.fill(0)
            elif self.post_fade_silence:
                outdata.fill(0)

            # Clip and copy to output
            np.clip(outdata, -1.0, 1.0, out=outdata)

        except Exception as e:
            self.callback_error_count += 1
            self.last_callback_error = e
            outdata.fill(0)

    def _copy_callback_samples(
        self, outdata: np.ndarray, samples: np.ndarray, frames: int
    ) -> None:
        """Copy callback samples into the stereo output buffer."""
        array = np.asarray(samples)
        if array.ndim == 0:
            outdata[:1, 0] = array
            outdata[:1, 1] = array
            return

        if array.ndim == 1:
            length = min(array.shape[0], frames)
            np.copyto(outdata[:length, 0], array[:length], casting="unsafe")
            np.copyto(outdata[:length, 1], array[:length], casting="unsafe")
            return

        length = min(array.shape[0], frames)
        if array.shape[1] == 1:
            mono = array[:length, 0]
            np.copyto(outdata[:length, 0], mono, casting="unsafe")
            np.copyto(outdata[:length, 1], mono, casting="unsafe")
            return

        np.copyto(outdata[:length, :2], array[:length, :2], casting="unsafe")

    def _fill_ramp(self, start: float, stop: float, length: int) -> np.ndarray:
        """Fill and return a reusable linear ramp buffer."""
        self._ensure_callback_buffers(length)
        ramp = self._ramp_buffer[:length]
        if length == 1:
            ramp[0] = stop
            return ramp

        indices = self._ramp_index_buffer[:length]
        scale = (stop - start) / (length - 1)
        np.multiply(indices, scale, out=ramp)
        ramp += start
        return ramp

    def _ensure_callback_buffers(self, frames: int) -> None:
        """Grow reusable callback buffers outside the steady-state hot path."""
        if frames > self._ramp_buffer.shape[0]:
            self._ramp_buffer = np.empty(frames, dtype=np.float32)
            self._ramp_index_buffer = np.arange(frames, dtype=np.float32)

    def start_playback(self):
        """Start audio playback."""
        if self.is_playing:
            logger.warning("Already playing")
            return

        try:
            # Enable fade-in to prevent startup click
            self.is_fading_out = False
            self.fade_out_samples_remaining = 0
            self.is_fading_in = True
            self.fade_in_samples_remaining = self.fade_in_total_samples
            self.post_fade_silence = False

            # Open audio stream
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=2,
                blocksize=self.buffer_size,
                callback=self._sounddevice_callback,
                dtype=np.float32,
            )

            self.stream.start()
            self.is_playing = True
            logger.info(
                f"Playback started (SR: {self.sample_rate} Hz, "
                f"Buffer: {self.buffer_size})"
            )

        except STREAM_ERROR_TYPES as e:
            self.stream = None
            self.is_playing = False
            self.is_fading_in = False
            self.fade_in_samples_remaining = 0
            logger.error(f"Failed to start playback: {e}", exc_info=True)
            raise

    def stop_playback(self, graceful: bool = True):
        """Stop audio playback.

        Args:
            graceful: If True, perform a short fade-out before closing the stream.
                If False, close the stream immediately.
        """
        if not self.is_playing:
            return

        if not graceful:
            logger.info("Stopping playback immediately")
            self._finalize_stop()
            return

        try:
            # Initiate fade-out
            self.is_fading_out = True
            self.fade_out_samples_remaining = self.fade_out_total_samples
            self.post_fade_silence = False

            logger.info(
                f"Stopping playback ({self.fade_out_duration_ms}ms fade-out...)"
            )

            # Wait for fade-out to complete
            import time

            fade_duration_sec = (self.fade_out_duration_ms + 100) / 1000.0
            time.sleep(fade_duration_sec)

            # Now stop the stream
            self._finalize_stop()

        except STREAM_ERROR_TYPES as e:
            logger.error(f"Failed to stop playback: {e}", exc_info=True)
            self._finalize_stop()

    def _finalize_stop(self):
        """Finalize playback stop after fade-out completes."""
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

            self.is_playing = False
            self.is_fading_out = False
            self.post_fade_silence = False
            logger.info("Playback stopped")

        except STREAM_ERROR_TYPES as e:
            logger.error(f"Failed to finalize stop: {e}", exc_info=True)

    def cleanup(self, graceful: bool = True):
        """Clean up resources.

        Args:
            graceful: If True, perform a short fade-out before closing playback.
        """
        if self.is_playing:
            self.stop_playback(graceful=graceful)
