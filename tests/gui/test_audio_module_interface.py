"""Test the generic audio module interface."""

import pytest
from PyQt6.QtWidgets import QApplication

from src.engine import SineOscillator
from src.gui.audio_module_interface import ModuleCategory
from src.gui.modules.mixer import MixerModule
from src.gui.modules.oscillator import OscillatorModule
from src.gui.modules.output import OutputModule
from src.gui.modules.volume_mod import VolumeModule


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication for tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_oscillator_interface(qapp):
    """Test oscillator implements interface correctly."""
    module = OscillatorModule()

    # Check module type (property, not method)
    assert module.metadata.category == ModuleCategory.SOURCE

    # Check no required inputs
    assert module.get_required_inputs() == []
    assert module.get_modulation_inputs() == []

    # Check component creation
    component = module.create_engine_component(
        input_components=None, modulation_components=None
    )
    assert component is not None
    assert isinstance(component, SineOscillator)


def test_volume_interface(qapp):
    """Test volume modifier implements interface correctly."""
    module = VolumeModule()

    # Check module type (property, not method)
    assert module.metadata.category == ModuleCategory.MODIFIER

    # Check required inputs
    assert "In" in module.get_required_inputs()
    assert "Mod" in module.get_modulation_inputs()

    # Check component creation without modulation
    component = module.create_engine_component(
        input_components=None, modulation_components=None
    )
    assert component is not None


def test_volume_with_modulation(qapp):
    """Test volume with modulation input."""
    module = VolumeModule()

    # Create modulation source
    lfo = SineOscillator(1.0, wave_range=(-1, 1))

    # Create component with modulation
    component = module.create_engine_component(
        input_components=None, modulation_components={"Mod": lfo}
    )
    assert component is not None


def test_mixer_interface(qapp):
    """Test mixer implements interface correctly."""
    module = MixerModule()

    # Check module type (property, not method)
    assert module.metadata.category == ModuleCategory.MIXER

    # Check no required inputs (accepts multiple)
    assert module.get_required_inputs() == []

    # Check component creation with multiple inputs
    osc1 = SineOscillator(440)
    osc2 = SineOscillator(550)

    component = module.create_engine_component(
        input_components=[osc1, osc2], modulation_components=None
    )
    assert component is not None


def test_output_interface(qapp):
    """Test output implements interface correctly."""
    module = OutputModule()

    # Check module type (property, not method)
    assert module.metadata.category == ModuleCategory.OUTPUT

    # Check required inputs
    assert "In" in module.get_required_inputs()

    # Check component creation (output returns the input)
    osc = SineOscillator(440)
    component = module.create_engine_component(
        input_components=[osc], modulation_components=None
    )
    assert component == osc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
