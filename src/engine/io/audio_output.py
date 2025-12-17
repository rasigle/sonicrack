"""Audio output logic for real-time playback using sounddevice."""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

DEFAULT_FADEOUT_DURATION_MS = 50
DEFAULT_FADEIN_DURATION_MS = 10


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
        audio_callback: Callable[[int], np.ndarray | None] | None = None,
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
        except Exception as e:
            logger.error(f"Error writing audio samples: {e}", exc_info=True)

    def set_audio_callback(self, callback: Callable[[int], np.ndarray | None]):
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
            logger.warning(f"Audio callback status: {status}")

        try:
            # Generate samples using the provided callback
            if self.audio_callback is None:
                outdata.fill(0)
                return

            samples = self.audio_callback(frames)

            if samples is None:
                outdata.fill(0)
                return

            samples = np.asarray(samples)
            if samples.ndim == 0:
                samples = np.atleast_1d(samples)

            # Ensure stereo format
            if samples.ndim == 1:
                samples = np.column_stack((samples, samples))
            elif samples.ndim == 2 and samples.shape[1] == 1:
                samples = np.repeat(samples, 2, axis=1)
            else:
                samples = samples.copy()

            # Clip samples to expected frame length
            if samples.shape[0] != frames:
                resized = np.zeros((frames, 2), dtype=samples.dtype)
                length = min(samples.shape[0], frames)
                resized[:length] = samples[:length]
                samples = resized

            # Apply master volume smoothing
            if self._master_volume_smoothing_samples > 0:
                # Linear interpolation from current to target
                num_smooth = min(self._master_volume_smoothing_samples, frames)
                alpha = np.linspace(0, 1, num_smooth)
                volume_ramp = (
                    self._current_master_volume * (1 - alpha)
                    + self._target_master_volume * alpha
                )

                samples[:num_smooth] *= volume_ramp[:, np.newaxis]
                if frames > num_smooth:
                    samples[num_smooth:] *= self._target_master_volume

                self._current_master_volume = self._target_master_volume
                self._master_volume_smoothing_samples -= num_smooth
            else:
                samples *= self._target_master_volume

            # Apply fade-in
            if self.is_fading_in and self.fade_in_samples_remaining > 0:
                num_fade = min(self.fade_in_samples_remaining, frames)
                num_fade = min(num_fade, samples.shape[0])
                if num_fade > 0:
                    segment = samples[:num_fade]
                    actual_len = segment.shape[0]
                    if actual_len > 0:
                        fade_curve = np.linspace(
                            1.0 - (self.fade_in_samples_remaining / self.fade_in_total_samples),
                            1.0
                            - (
                                (self.fade_in_samples_remaining - actual_len)
                                / self.fade_in_total_samples
                            ),
                            actual_len,
                        )
                        fade_curve_stereo = fade_curve[:, np.newaxis]
                        np.multiply(segment, fade_curve_stereo, out=segment)
                        self.fade_in_samples_remaining -= actual_len

                        if self.fade_in_samples_remaining <= 0:
                            self.is_fading_in = False
                            logger.debug("Fade-in complete")

            # Apply fade-out
            if self.is_fading_out and not self.post_fade_silence:
                if self.fade_out_samples_remaining > 0:
                    num_fade = min(self.fade_out_samples_remaining, frames)
                    num_fade = min(num_fade, samples.shape[0])
                    if num_fade > 0:
                        segment = samples[:num_fade]
                        actual_len = segment.shape[0]
                        if actual_len > 0:
                            fade_curve = np.linspace(
                                self.fade_out_samples_remaining / self.fade_out_total_samples,
                                (self.fade_out_samples_remaining - actual_len)
                                / self.fade_out_total_samples,
                                actual_len,
                            )
                            fade_curve_stereo = fade_curve[:, np.newaxis]
                            np.multiply(segment, fade_curve_stereo, out=segment)
                            self.fade_out_samples_remaining -= actual_len

                            if self.fade_out_samples_remaining <= 0:
                                self.post_fade_silence = True
                                logger.debug("Fade-out complete, entering silence mode")

                        if frames > num_fade:
                            samples[num_fade:] = 0
                else:
                    samples.fill(0)
            elif self.post_fade_silence:
                samples.fill(0)

            # Clip and copy to output
            np.clip(samples, -1.0, 1.0, out=samples)
            outdata[:] = samples.astype(np.float32)

        except Exception as e:
            logger.error(f"Error in audio callback: {e}", exc_info=True)
            outdata.fill(0)

    def start_playback(self):
        """Start audio playback."""
        if self.is_playing:
            logger.warning("Already playing")
            return

        try:
            # Enable fade-in to prevent startup click
            self.is_fading_in = True
            self.fade_in_samples_remaining = self.fade_in_total_samples
            self.post_fade_silence = False

            # Open audio stream
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=2,
                blocksize=self.buffer_size,
                callback=self._sounddevice_callback if self.audio_callback else None,
                dtype=np.float32,
            )

            self.stream.start()
            self.is_playing = True
            logger.info(
                f"Playback started (SR: {self.sample_rate} Hz, "
                f"Buffer: {self.buffer_size})"
            )

        except Exception as e:
            logger.error(f"Failed to start playback: {e}", exc_info=True)
            raise

    def stop_playback(self):
        """Stop audio playback with fade-out to prevent clicks."""
        if not self.is_playing:
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

        except Exception as e:
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

        except Exception as e:
            logger.error(f"Failed to finalize stop: {e}", exc_info=True)

    def cleanup(self):
        """Clean up resources with graceful fade-out."""
        if self.is_playing:
            self.stop_playback()
