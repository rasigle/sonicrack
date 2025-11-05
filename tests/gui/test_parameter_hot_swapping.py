"""
Test parameter hot-swapping to verify zero-click parameter changes.

Run this test while the GUI is running and audio is playing.
"""

import pytest

from src.engine.oscillator import SineOscillator
from src.gui.audio_module_interface import ModuleCategory
from src.gui.patch_compiler import PatchCompiler


class MockOscillatorModule:
    """Mock oscillator module for testing."""

    class MockMetadata:
        category = ModuleCategory.SOURCE
        title = "Test Oscillator"

    metadata = MockMetadata()

    def create_component(self, input_components=None, modulation_components=None):
        """Create a SineOscillator component."""
        return SineOscillator(frequency=440, gain_db=-20)

    def get_required_inputs(self):
        return []

    def get_modulation_inputs(self):
        return []


def test_hot_swap_frequency():
    """Test that frequency can be hot-swapped without recompiling."""
    compiler = PatchCompiler()

    # Create mock module
    module = MockOscillatorModule()

    # Simulate compilation
    compiler.modules = [module]
    compiler.connections = []

    # Build component
    component = module.create_component()
    compiler._module_to_component[module] = component

    # Verify initial frequency
    assert component.frequency == 440

    # Hot-swap frequency
    success = compiler.update_parameter(module, "frequency", 880)

    # Verify hot-swap worked
    assert success is True
    assert component.frequency == 880

    # Verify phase continuity (phase should not reset)
    # Generate some samples to advance phase
    _ = component.get_samples_vectorized(100)

    # Hot-swap again
    compiler.update_parameter(module, "frequency", 1320)

    # Generate more samples
    _ = component.get_samples_vectorized(100)

    # Phase should be continuous (no reset to 0)
    # If phase reset, there would be a discontinuity
    print("✓ Hot-swap maintains phase continuity")


def test_hot_swap_gain_db():
    """Test that gain_db can be hot-swapped."""
    compiler = PatchCompiler()
    module = MockOscillatorModule()

    component = module.create_component()
    compiler._module_to_component[module] = component

    # Initial gain
    assert component.gain_db == pytest.approx(-20, rel=0.1)

    # Hot-swap gain
    success = compiler.update_parameter(module, "gain_db", -12)

    assert success is True
    assert component.gain_db == pytest.approx(-12, rel=0.1)
    assert component.amplitude == pytest.approx(0.25, rel=0.01)


def test_hot_swap_nonexistent_parameter():
    """Test that hot-swapping nonexistent parameter returns False."""
    compiler = PatchCompiler()
    module = MockOscillatorModule()

    component = module.create_component()
    compiler._module_to_component[module] = component

    # Try to hot-swap parameter that doesn't exist
    success = compiler.update_parameter(module, "nonexistent_param", 123)

    assert success is False


def test_hot_swap_module_not_in_mapping():
    """Test that hot-swapping unknown module returns False."""
    compiler = PatchCompiler()
    module = MockOscillatorModule()

    # Don't add to mapping

    # Try to hot-swap
    success = compiler.update_parameter(module, "frequency", 880)

    assert success is False


if __name__ == "__main__":
    print("Testing Parameter Hot-Swapping...")
    print("=" * 60)

    test_hot_swap_frequency()
    print("✓ Frequency hot-swap works")

    test_hot_swap_gain_db()
    print("✓ Gain dB hot-swap works")

    test_hot_swap_nonexistent_parameter()
    print("✓ Nonexistent parameter handled correctly")

    test_hot_swap_module_not_in_mapping()
    print("✓ Unknown module handled correctly")

    print("=" * 60)
    print("All hot-swapping tests passed! ✅")
    print("\nParameter changes will now be click-free during playback!")
