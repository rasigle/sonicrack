"""Audio engine for real-time synthesis and playback."""

import logging
from typing import Any

import numpy as np
import sounddevice as sd
from PyQt6.QtCore import QObject, pyqtSignal

from src.constants import DEFAULT_SAMPLE_RATE

logger = logging.getLogger(__name__)


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

        # Buffer for visualization
        self.current_buffer: np.ndarray | None = None

    def set_patch(self, patch: Any):
        """Set the audio patch to play.

        Args:
            patch: Audio component (oscillator, chain, etc.)
        """
        self.patch = patch

        logger.info(f"Patch set: {type(patch).__name__}")

    def set_master_volume(self, volume: float):
        """Set the master output volume.

        Args:
            volume: Volume level (0.0 to 1.0)
        """
        self.master_volume = np.clip(volume, 0.0, 1.0)

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

            # Apply master volume
            stereo = stereo * self.master_volume

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
        """Stop audio playback."""
        if not self.is_playing:
            return

        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

            self.is_playing = False
            self.playback_stopped.emit()
            logger.info("Playback stopped")

        except Exception as e:
            logger.error(f"Failed to stop playback: {e}", exc_info=True)

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
            stereo = np.clip(stereo * self.master_volume, -1.0, 1.0)

            return stereo

        except Exception as e:
            logger.error(f"Failed to generate samples: {e}", exc_info=True)
            return None

    def cleanup(self):
        """Clean up resources."""
        self.stop_playback()
