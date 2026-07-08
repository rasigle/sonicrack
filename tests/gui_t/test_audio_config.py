"""Test script for centralized audio configuration.

Tests that audio_config properly manages sample rate and buffer size
and notifies all registered listeners.
"""

import pytest
from soniclab.generators.oscillators.oscillator import SineOscillator

from src.gui.audio_config import AudioConfig, audio_config


def test_singleton():
    """Test that AudioConfig is a proper singleton."""
    config1 = AudioConfig()
    config2 = AudioConfig()
    config3 = audio_config

    assert config1 is config2, "AudioConfig should be a singleton"
    assert config1 is config3, "audio_config should be the same instance"


def test_sample_rate():
    """Test sample rate property and notifications."""
    # Reset to default
    audio_config.sample_rate = 44100

    # Test getter
    assert audio_config.sample_rate == 44100, "Initial sample rate should be 44100"

    # Test setter
    audio_config.sample_rate = 48000
    assert audio_config.sample_rate == 48000, "Sample rate should be 48000"

    # Test listener notification
    notifications = []

    def listener(new_rate):
        notifications.append(new_rate)

    audio_config.add_sample_rate_listener(listener)
    audio_config.sample_rate = 96000

    assert len(notifications) == 1, "Listener should be called once"
    assert notifications[0] == 96000, "Listener should receive new rate"

    # Test multiple listeners
    notifications2 = []

    def listener2(new_rate):
        notifications2.append(new_rate)

    audio_config.add_sample_rate_listener(listener2)
    audio_config.sample_rate = 22050

    assert len(notifications) == 2, "First listener should be called again"
    assert len(notifications2) == 1, "Second listener should be called once"
    assert notifications[1] == 22050, "First listener should receive new rate"
    assert notifications2[0] == 22050, "Second listener should receive new rate"

    # Cleanup
    audio_config.remove_sample_rate_listener(listener)
    audio_config.remove_sample_rate_listener(listener2)


def test_buffer_size():
    """Test buffer size property and notifications."""
    # Reset to default
    audio_config.buffer_size = 512

    # Test getter
    assert audio_config.buffer_size == 512, "Initial buffer size should be 512"

    # Test setter
    audio_config.buffer_size = 1024
    assert audio_config.buffer_size == 1024, "Buffer size should be 1024"

    # Test listener notification
    notifications = []

    def listener(new_size):
        notifications.append(new_size)

    audio_config.add_buffer_size_listener(listener)
    audio_config.buffer_size = 2048

    assert len(notifications) == 1, "Listener should be called once"
    assert notifications[0] == 2048, "Listener should receive new size"

    # Cleanup
    audio_config.remove_buffer_size_listener(listener)


def test_oscillator_integration():
    """Test that oscillators can register with audio_config."""

    # Create oscillator at default sample rate
    audio_config.sample_rate = 44100
    osc = SineOscillator(frequency=440, sample_rate=audio_config.sample_rate)

    # Register callback to update oscillator
    def update_osc_sample_rate(new_rate):
        osc.sample_rate = new_rate

    audio_config.add_sample_rate_listener(update_osc_sample_rate)

    # Change global sample rate
    audio_config.sample_rate = 48000

    # Verify oscillator was updated
    assert osc.sample_rate == 48000, "Oscillator should be updated"

    # Cleanup
    audio_config.remove_sample_rate_listener(update_osc_sample_rate)


def test_validation():
    """Test input validation."""
    # Test invalid sample rate - should raise ValueError
    with pytest.raises(ValueError, match="Sample rate must be positive"):
        audio_config.sample_rate = 0

    with pytest.raises(ValueError, match="Sample rate must be positive"):
        audio_config.sample_rate = -44100

    # Test invalid buffer size - should raise ValueError
    with pytest.raises(ValueError, match="Buffer size must be positive"):
        audio_config.buffer_size = 0

    with pytest.raises(ValueError, match="Buffer size must be positive"):
        audio_config.buffer_size = -512


def test_no_notification_on_same_value():
    """Test that listeners are not notified if value doesn't change."""
    notifications = []

    def listener(new_rate):
        notifications.append(new_rate)

    audio_config.add_sample_rate_listener(listener)
    audio_config.sample_rate = 44100

    # Setting to same value should not notify
    initial_count = len(notifications)
    audio_config.sample_rate = 44100

    assert len(notifications) == initial_count, "Should not notify on same value"

    # Cleanup
    audio_config.remove_sample_rate_listener(listener)


def test_listener_exception_handling():
    """Test that exceptions in listeners don't break the system."""
    notifications = []

    def bad_listener(new_rate):
        raise RuntimeError("Test error")

    def good_listener(new_rate):
        notifications.append(new_rate)

    audio_config.add_sample_rate_listener(bad_listener)
    audio_config.add_sample_rate_listener(good_listener)

    # Should not raise, and good listener should still be called
    audio_config.sample_rate = 96000

    assert len(notifications) == 1, (
        "Good listener should be called despite bad listener"
    )
    assert notifications[0] == 96000

    # Cleanup
    audio_config.remove_sample_rate_listener(bad_listener)
    audio_config.remove_sample_rate_listener(good_listener)


def test_remove_nonexistent_listener():
    """Test removing a listener that was never added."""

    def listener(new_rate):
        _ = new_rate

    # Should not raise
    audio_config.remove_sample_rate_listener(listener)
    audio_config.remove_buffer_size_listener(listener)


def test_clear_all_listeners():
    """Test clearing all listeners."""
    notifications1 = []
    notifications2 = []

    def listener1(new_rate):
        notifications1.append(new_rate)

    def listener2(new_rate):
        notifications2.append(new_rate)

    audio_config.add_sample_rate_listener(listener1)
    audio_config.add_sample_rate_listener(listener2)

    # Clear all
    audio_config.clear_all_listeners()

    # Change value - no listeners should be called
    audio_config.sample_rate = 22050

    assert len(notifications1) == 0, "Listener 1 should not be called after clear"
    assert len(notifications2) == 0, "Listener 2 should not be called after clear"


def test_typical_sample_rates():
    """Test that typical sample rates are accepted."""
    typical_rates = [22050, 44100, 48000, 88200, 96000, 192000]

    for rate in typical_rates:
        audio_config.sample_rate = rate
        assert audio_config.sample_rate == rate, f"Should accept {rate} Hz"


def test_typical_buffer_sizes():
    """Test that typical buffer sizes are accepted."""
    typical_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]

    for size in typical_sizes:
        audio_config.buffer_size = size
        assert audio_config.buffer_size == size, f"Should accept {size} samples"


def test_multiple_oscillators():
    """Test updating multiple oscillators simultaneously."""
    audio_config.sample_rate = 44100

    # Create multiple oscillators
    oscs = [
        SineOscillator(frequency=440, sample_rate=audio_config.sample_rate),
        SineOscillator(frequency=880, sample_rate=audio_config.sample_rate),
        SineOscillator(frequency=220, sample_rate=audio_config.sample_rate),
    ]

    # Register callback to update all oscillators
    def update_all_oscs(new_rate):
        for osc in oscs:
            osc.sample_rate = new_rate

    audio_config.add_sample_rate_listener(update_all_oscs)

    # Change sample rate
    audio_config.sample_rate = 48000

    # Verify all oscillators were updated
    for osc in oscs:
        assert osc.sample_rate == 48000, "All oscillators should be updated"

    # Cleanup
    audio_config.remove_sample_rate_listener(update_all_oscs)


def test_repr():
    """Test string representation."""
    repr_str = repr(audio_config)
    assert "AudioConfig" in repr_str
    assert "sample_rate" in repr_str
    assert "buffer_size" in repr_str


# Cleanup fixture to reset state after all tests
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up audio_config state after each test."""
    yield
    # Reset to defaults
    audio_config.clear_all_listeners()
    audio_config.sample_rate = 44100
    audio_config.buffer_size = 512
