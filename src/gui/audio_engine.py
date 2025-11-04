"""Audio engine for real-time synthesis and playback."""

import logging
from typing import Any

import numpy as np
import sounddevice as sd
from PyQt6.QtCore import QObject, pyqtSignal

from src.constants import DEFAULT_SAMPLE_RATE

logger = logging.getLogger(__name__)

DEFAULT_FADEOUT_DURATION_MS = 50  # Default fade-out duration in milliseconds


class AudioEngine(QObject):
    """Audio engine for real-time synthesis and playback.

    Manages audio generation from the patch and streams it to the audio output.
    """

    # Signals
    samples_generated = pyqtSignal(np.ndarray)  # Emitted when new samples are generated
    playback_started = pyqtSignal()
    playback_stopped = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE, buffer_size: int = 2048):
        """Initialize the audio engine.

        Args:
            sample_rate: Sample rate in Hz
            buffer_size: Audio buffer size in samples
        """
        super().__init__()

        self.sample_rate: float = sample_rate
        self.buffer_size: float = buffer_size

        # Audio state
        self.is_playing: bool = False
        self.stream: sd.OutputStream | None = None
        self.patch: Any | None = None
        self.master_volume: float = 0.7

        # Master volume smoothing to prevent clicks
        self._target_master_volume = 0.7
        self._current_master_volume = 0.7
        self._master_volume_smoothing_samples = 0
        self._master_volume_smoothing_duration = 441  # 10ms @ 44.1kHz

        # Fade-out parameters
        self.current_buffer: np.ndarray | None = None

        # Fade-out state to prevent clicks on stop
        self.is_fading_out: bool = False
        self.fade_out_samples_remaining: int = 0
        self.fade_out_duration_ms: float = DEFAULT_FADEOUT_DURATION_MS  # 50 ms default
        self.fade_out_total_samples: int = int(self.sample_rate * self.fade_out_duration_ms / 1000)

        # Fade-in state to prevent clicks on start
        self.is_fading_in: bool = False
        self.fade_in_samples_remaining: int = 0
        self.fade_in_duration_ms: float = 10  # 10ms fade-in (shorter than fade-out)
        self.fade_in_total_samples: int = int(self.sample_rate * self.fade_in_duration_ms / 1000)

        # Track if we're in post-fade silence mode
        self.post_fade_silence: bool = False

    def set_patch(self, patch: Any):
        """Set the audio patch to play.

        Args:
            patch: Audio component (oscillator, chain, etc.)
        """
        self.patch = patch

        logger.debug(f"Patch set: {type(patch).__name__}")

    def set_master_volume(self, volume: float):
        """Set the master output volume with smoothing to prevent clicks.

        Args:
            volume: Volume level (0.0 to 1.0)
        """
        new_volume = np.clip(volume, 0.0, 1.0)
        # Trigger smooth transition
        self._target_master_volume = new_volume
        self._master_volume_smoothing_samples = self._master_volume_smoothing_duration
        self.master_volume = new_volume  # Update stored value

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status):
        """Audio callback function for sounddevice.

        Args:
            outdata: Output buffer to fill
            frames: Number of frames requested
            time_info: Time information
            status: Stream status
        """
        if status:
            logger.warning(f"Audio callback status: {status}")

        try:
            if self.patch is None:
                # Output silence
                outdata.fill(0)
                return

            # If in post-fade silence mode, just output silence
            if self.post_fade_silence:
                outdata.fill(0)
                return

            # Generate samples from patch
            samples = self.patch.get_samples(frames)

            # Handle different output formats
            if isinstance(samples, (list, tuple)) and len(samples) == 2:
                # Stereo output (left, right)
                left, right = samples
                if len(left.shape) == 1 and len(right.shape) == 1:
                    stereo = np.column_stack((left, right))
                else:
                    stereo = samples

            elif isinstance(samples, np.ndarray):
                if len(samples.shape) == 1:
                    # Mono, duplicate to stereo
                    stereo = np.column_stack((samples, samples))
                else:
                    # Already stereo
                    stereo = samples

            else:
                # Fallback to silence
                outdata.fill(0)
                return

            # Apply master volume with smoothing (prevents clicks!)
            if self._master_volume_smoothing_samples > 0:
                # Calculate how many samples to smooth
                smooth_count = min(frames, self._master_volume_smoothing_samples)

                # Create smooth volume envelope
                volume_envelope = np.linspace(
                    self._current_master_volume,
                    self._target_master_volume,
                    smooth_count
                )

                # Apply smoothed volume to first part
                if stereo.shape[0] >= smooth_count:
                    stereo[:smooth_count] = stereo[:smooth_count] * volume_envelope[:, np.newaxis]

                    # Apply target volume to rest
                    if smooth_count < frames:
                        stereo[smooth_count:] = stereo[smooth_count:] * self._target_master_volume
                else:
                    # Buffer smaller than smooth_count
                    stereo = stereo * volume_envelope[:stereo.shape[0], np.newaxis]

                # Update smoothing state
                self._master_volume_smoothing_samples -= smooth_count
                if self._master_volume_smoothing_samples <= 0:
                    self._current_master_volume = self._target_master_volume
                else:
                    self._current_master_volume = volume_envelope[-1]
            else:
                # No smoothing - direct multiplication
                stereo = stereo * self._target_master_volume

            # Apply fade-in if starting playback
            if self.is_fading_in and self.fade_in_samples_remaining > 0:
                # Calculate how many samples to fade in this buffer
                fade_samples = min(frames, self.fade_in_samples_remaining)

                # Create fade-in curve (linear)
                fade_start = 1.0 - (self.fade_in_samples_remaining / self.fade_in_total_samples)
                fade_end = 1.0 - (max(0, self.fade_in_samples_remaining - fade_samples) / self.fade_in_total_samples)
                fade_curve = np.linspace(fade_start, fade_end, fade_samples)

                # Apply fade to the samples
                stereo[:fade_samples] *= fade_curve[:, np.newaxis]

                self.fade_in_samples_remaining -= fade_samples

                # If fade-in complete, disable it
                if self.fade_in_samples_remaining <= 0:
                    self.is_fading_in = False

            # Apply fade-out if stopping playback
            if self.is_fading_out and self.fade_out_samples_remaining > 0:
                # Calculate how many samples to fade in this buffer
                fade_samples = min(frames, self.fade_out_samples_remaining)

                # Create fade-out curve (linear for simplicity, could use exponential)
                fade_curve = np.linspace(
                    self.fade_out_samples_remaining / self.fade_out_total_samples,
                    max(0.0, (self.fade_out_samples_remaining - fade_samples) / self.fade_out_total_samples),
                    fade_samples
                )

                # Apply fade to the samples
                stereo[:fade_samples] *= fade_curve[:, np.newaxis]

                # Remaining samples after fade are silent
                if fade_samples < frames:
                    stereo[fade_samples:] = 0

                self.fade_out_samples_remaining -= fade_samples

                # If fade-out complete, enter post-fade silence mode
                if self.fade_out_samples_remaining <= 0:
                    self.is_fading_out = False
                    self.post_fade_silence = True
                    # Fill rest with silence
                    if fade_samples < frames:
                        stereo[fade_samples:] = 0

            # Clip to valid range
            stereo = np.clip(stereo, -1.0, 1.0)

            # Copy to output buffer
            if len(stereo) >= frames:
                outdata[:] = stereo[:frames].astype(np.float32)
            else:
                outdata[: len(stereo)] = stereo.astype(np.float32)
                outdata[len(stereo) :] = 0

            # Store for visualization
            self.current_buffer = stereo[:frames].copy()
            self.samples_generated.emit(self.current_buffer)

        except Exception as e:
            logger.error(f"Error in audio callback: {e}", exc_info=True)
            self.error_occurred.emit(str(e))
            outdata.fill(0)

    def start_playback(self):
        """Start audio playback."""
        if self.is_playing:
            logger.warning("Already playing")
            return

        if self.patch is None:
            logger.warning("No patch set")
            self.error_occurred.emit("No patch configured")
            return

        try:
            # Reset patch components
            if hasattr(self.patch, "reset"):
                self.patch.reset()

            # Enable fade-in to prevent startup click
            self.is_fading_in = True
            self.fade_in_samples_remaining = self.fade_in_total_samples
            self.post_fade_silence = False  # Ensure we're not in silence mode

            # Open audio stream
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=2,
                blocksize=self.buffer_size,
                callback=self._audio_callback,
                dtype=np.float32,
            )

            self.stream.start()
            self.is_playing = True
            self.playback_started.emit()
            logger.info("Playback started")

        except Exception as e:
            logger.error(f"Failed to start playback: {e}", exc_info=True)
            self.error_occurred.emit(f"Failed to start playback: {e}")

    def stop_playback(self):
        """Stop audio playback with fade-out to prevent clicks."""
        if not self.is_playing:
            return

        try:
            # Initiate fade-out instead of immediate stop
            self.is_fading_out = True
            self.fade_out_samples_remaining = self.fade_out_total_samples
            self.post_fade_silence = False

            logger.info(f"Stopping playback  ({self.fade_out_duration_ms}ms fade-out...)")

            # Schedule the actual stream stop after fade-out completes
            # Use a timer to avoid blocking the GUI thread
            from PyQt6.QtCore import QTimer

            # Wait for fade + extra buffer time for safety
            fade_duration_ms = int(self.fade_out_duration_ms + 100)  # 50ms fade + 100ms buffer
            QTimer.singleShot(fade_duration_ms, self._finalize_stop)

        except Exception as e:
            logger.error(f"Failed to stop playback: {e}", exc_info=True)
            self._finalize_stop()  # Ensure cleanup happens

    def _finalize_stop(self):
        """Finalize playback stop after fade-out completes."""
        try:
            # Now actually stop the stream
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

            self.is_playing = False
            self.is_fading_out = False
            self.post_fade_silence = False
            self.playback_stopped.emit()
            logger.info("Playback stopped")

        except Exception as e:
            logger.error(f"Failed to finalize stop: {e}", exc_info=True)

    def generate_samples(self, num_samples: int) -> np.ndarray | None:
        """Generate samples from the patch without playback.

        Args:
            num_samples: Number of samples to generate

        Returns:
            Generated samples or None
        """
        if self.patch is None:
            return None

        try:
            samples = self.patch.get_samples(num_samples)

            # Convert to stereo if needed
            if isinstance(samples, (list, tuple)) and len(samples) == 2:
                left, right = samples
                stereo = np.column_stack((left, right))
            elif isinstance(samples, np.ndarray):
                if len(samples.shape) == 1:
                    stereo = np.column_stack((samples, samples))
                else:
                    stereo = samples
            else:
                return None

            # Apply master volume and clip
            stereo = np.clip(stereo * self._target_master_volume, -1.0, 1.0)

            return stereo

        except Exception as e:
            logger.error(f"Failed to generate samples: {e}", exc_info=True)
            return None

    def cleanup(self):
        """Clean up resources with graceful fade-out."""
        if self.is_playing:
            # Initiate fade-out
            self.is_fading_out = True
            self.fade_out_samples_remaining = self.fade_out_total_samples
            self.post_fade_silence = False

            logger.info("Cleanup: Fading out audio...")

            # Wait for fade-out to complete (blocking is OK during cleanup)
            import time
            fade_duration_sec = (self.fade_out_duration_ms + 100) / 1000.0
            time.sleep(fade_duration_sec)

            # Now stop immediately
            self._finalize_stop()
        else:
            # Not playing, just clean up
            self.stop_playback()
