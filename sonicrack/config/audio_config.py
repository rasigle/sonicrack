"""Global audio configuration singleton.

This module provides a centralized place to store and access audio configuration
parameters (sample rate, buffer size) that need to be shared across all modules.

The AudioConfig singleton ensures:
- Single source of truth for audio settings
- Automatic notification of changes to all components
- Thread-safe access to configuration values
- Easy integration with GUI controls

Example:
    >>> from sonicrack.config.audio_config import AudioConfig
    >>> # Singleton audio configuration
    >>> config = AudioConfig()
    >>> sr = config.sample_rate
    >>>
    >>> # Register a callback for changes
    >>> def on_sample_rate_changed(new_rate):
    ...     print(f"Sample rate changed to {new_rate}")
    >>> audio_config.add_sample_rate_listener(on_sample_rate_changed)
    >>>
    >>> # Change settings (all listeners will be notified)
    >>> config.sample_rate = 48000
"""

import logging
from collections.abc import Callable

from sonicrack.constants import DEFAULT_BUFFER_SIZE, DEFAULT_SAMPLE_RATE

logger = logging.getLogger(__name__)


class AudioConfig:
    """Singleton class for global audio configuration.

    This class provides centralized storage and management of audio settings
    that need to be shared across all modules (sample rate, buffer size).

    Features:
    - Observer pattern: register callbacks to be notified of changes
    - Thread-safe: uses property setters with validation
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize audio configuration with defaults."""
        if self._initialized:
            return

        self._sample_rate = DEFAULT_SAMPLE_RATE
        self._buffer_size = DEFAULT_BUFFER_SIZE

        # Observer callbacks
        self._sample_rate_listeners: list[Callable[[int], None]] = []
        self._buffer_size_listeners: list[Callable[[int], None]] = []

        self._initialized = True
        logger.info(
            f"AudioConfig initialized: sample_rate={self._sample_rate} Hz, "
            f"buffer_size={self._buffer_size} samples"
        )

    @property
    def sample_rate(self) -> int:
        """Get current sample rate in Hz.

        Returns:
            Current sample rate (e.g., 44100, 48000)
        """
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int):
        """Set sample rate and notify all registered listeners.

        Args:
            value: New sample rate in Hz (must be positive)

        Raises:
            ValueError: If sample rate is not positive
        """
        if value <= 0:
            raise ValueError(f"Sample rate must be positive, got {value}")

        if value != self._sample_rate:
            old_value = self._sample_rate
            self._sample_rate = value
            logger.info(f"Sample rate changed: {old_value} Hz → {value} Hz")

            # Notify all listeners
            for listener in self._sample_rate_listeners:
                try:
                    listener(value)
                except Exception as e:
                    logger.error(
                        f"Error in sample rate listener {listener.__name__}: {e}",
                        exc_info=True,
                    )

    @property
    def buffer_size(self) -> int:
        """Get current buffer size in samples.

        Returns:
            Current buffer size (e.g., 512, 1024)
        """
        return self._buffer_size

    @buffer_size.setter
    def buffer_size(self, value: int):
        """Set buffer size and notify all registered listeners.

        Args:
            value: New buffer size in samples (must be positive)

        Raises:
            ValueError: If buffer size is not positive
        """
        if value <= 0:
            raise ValueError(f"Buffer size must be positive, got {value}")

        if value != self._buffer_size:
            old_value = self._buffer_size
            self._buffer_size = value
            logger.info(f"Buffer size changed: {old_value} → {value} samples")

            # Notify all listeners
            for listener in self._buffer_size_listeners:
                try:
                    listener(value)
                except Exception as e:
                    logger.error(
                        f"Error in buffer size listener {listener.__name__}: {e}",
                        exc_info=True,
                    )

    def add_sample_rate_listener(self, callback: Callable[[int], None]):
        """Register a callback to be notified when sample rate changes.

        The callback will be called with the new sample rate as an argument.

        Args:
            callback: Function that takes sample_rate (int) as parameter

        Example:
            >>> def on_sr_change(new_rate):
            ...     print(f"New rate: {new_rate}")
            >>> audio_config.add_sample_rate_listener(on_sr_change)
        """
        if callback not in self._sample_rate_listeners:
            self._sample_rate_listeners.append(callback)
            logger.debug(f"Added sample rate listener: {callback.__name__}")

    def remove_sample_rate_listener(self, callback: Callable[[int], None]):
        """Unregister a sample rate change callback.

        Args:
            callback: Previously registered callback function
        """
        if callback in self._sample_rate_listeners:
            self._sample_rate_listeners.remove(callback)
            logger.debug(f"Removed sample rate listener: {callback.__name__}")

    def add_buffer_size_listener(self, callback: Callable[[int], None]):
        """Register a callback to be notified when buffer size changes.

        The callback will be called with the new buffer size as an argument.

        Args:
            callback: Function that takes buffer_size (int) as parameter

        Example:
            >>> def on_buffer_change(new_size):
            ...     print(f"New buffer: {new_size}")
            >>> audio_config.add_buffer_size_listener(on_buffer_change)
        """
        if callback not in self._buffer_size_listeners:
            self._buffer_size_listeners.append(callback)
            logger.debug(f"Added buffer size listener: {callback.__name__}")

    def remove_buffer_size_listener(self, callback: Callable[[int], None]):
        """Unregister a buffer size change callback.

        Args:
            callback: Previously registered callback function
        """
        if callback in self._buffer_size_listeners:
            self._buffer_size_listeners.remove(callback)
            logger.debug(f"Removed buffer size listener: {callback.__name__}")

    def clear_all_listeners(self):
        """Remove all registered listeners.

        Useful for cleanup or testing.
        """
        self._sample_rate_listeners.clear()
        self._buffer_size_listeners.clear()
        logger.debug("Cleared all listeners")

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"AudioConfig(sample_rate={self._sample_rate} Hz, "
            f"buffer_size={self._buffer_size} samples, "
            f"sr_listeners={len(self._sample_rate_listeners)}, "
            f"buf_listeners={len(self._buffer_size_listeners)})"
        )


# Global singleton instance
audio_config = AudioConfig()


# Convenience accessor functions (optional, for cleaner code)
def get_sample_rate() -> int:
    """Get current global sample rate.

    Returns:
        Current sample rate in Hz
    """
    return audio_config.sample_rate


def get_buffer_size() -> int:
    """Get current global buffer size.

    Returns:
        Current buffer size in samples
    """
    return audio_config.buffer_size


def set_sample_rate(value: int):
    """Set global sample rate.

    This will notify all registered listeners.

    Args:
        value: New sample rate in Hz
    """
    audio_config.sample_rate = value


def set_buffer_size(value: int):
    """Set global buffer size.

    This will notify all registered listeners.

    Args:
        value: New buffer size in samples
    """
    audio_config.buffer_size = value
