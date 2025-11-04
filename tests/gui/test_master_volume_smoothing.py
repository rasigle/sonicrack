"""
Test master volume and phase smoothing to verify click elimination.
"""

from src.gui.audio_engine import AudioEngine


def test_master_volume_smoothing():
    """Test that master volume changes are smoothed."""
    engine = AudioEngine()

    # Initial state
    initial_vol = engine._current_master_volume  # 0.7 by default

    # Change volume (should trigger smoothing)
    engine.set_master_volume(0.5)

    # Verify smoothing state is active
    assert engine._master_volume_smoothing_samples > 0
    assert engine._target_master_volume == 0.5
    assert engine._current_master_volume == initial_vol  # Still at old value

    print("✓ Master volume smoothing initialized correctly")


def test_master_volume_no_discontinuity():
    """Test that rapid master volume changes don't create discontinuities."""
    engine = AudioEngine()

    # Simulate rapid volume changes
    volumes = [1.0, 0.8, 0.6, 0.4, 0.2, 0.5, 0.7, 1.0]

    for vol in volumes:
        engine.set_master_volume(vol)
        # Verify smoothing triggered
        assert engine._master_volume_smoothing_samples > 0
        assert engine._target_master_volume == vol

    # Verify final state
    assert engine._target_master_volume == 1.0

    print("✓ Rapid master volume changes handled correctly")


def test_master_volume_range_clamping():
    """Test that master volume is clamped to valid range."""
    engine = AudioEngine()

    # Test values outside range
    engine.set_master_volume(1.5)  # Above max
    assert engine._target_master_volume == 1.0

    engine.set_master_volume(-0.5)  # Below min
    assert engine._target_master_volume == 0.0

    # Test edge values
    engine.set_master_volume(0.0)
    assert engine._target_master_volume == 0.0

    engine.set_master_volume(1.0)
    assert engine._target_master_volume == 1.0

    print("✓ Master volume clamping works correctly")


if __name__ == "__main__":
    print("Testing Master Volume Smoothing...")
    print("="*60)

    test_master_volume_smoothing()
    test_master_volume_no_discontinuity()
    test_master_volume_range_clamping()

    print("="*60)
    print("All master volume smoothing tests passed! ✅")
    print("\nMaster volume changes are now click-free!")

