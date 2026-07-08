"""Test the generic audio module interface."""

import logging

import pytest
from soniclab.generators.oscillators.oscillator import SineOscillator

from sonicrack.gui.modules.input.midi_keyboard import MIDIKeyboardModule
from sonicrack.gui.modules.mixer import MixerModule
from sonicrack.gui.modules.modifier.volume_mod import VolumeModule
from sonicrack.gui.modules.sequencing.step_sequencer import StepSequencerModule
from sonicrack.gui.modules.source.lfo import LFOModule
from sonicrack.gui.modules.source.oscillator import OscillatorModule
from sonicrack.gui.modules.source.vco import ModulatedOscillatorModule
from sonicrack.gui.modules.voice.tb303_voice import TB303VoiceModule
from sonicrack.patching.module import ModuleCategory, infer_port_signal
from sonicrack.patching.port import PortSignal


def test_oscillator_interface():
    """Test oscillator implements interface correctly."""
    module = OscillatorModule()

    # Check module type (property, not method)
    assert module.metadata.category == ModuleCategory.SOURCE

    # Check no required inputs (oscillator is a source)
    assert not module.get_required_inputs()
    assert not module.get_modulation_inputs()

    # Check component creation with no ports connected returns None
    component = module.create_engine_component(
        input_components=None, modulation_components=None
    )
    assert component is None

    # Check that get_output_component returns oscillators for each port
    # Note: These create new oscillator instances each time
    sine_component = module.get_output_component("Sine")
    assert sine_component is None or isinstance(sine_component, SineOscillator)

    triangle_component = module.get_output_component("Triangle")
    assert triangle_component is None  # No cables connected

    square_component = module.get_output_component("Square")
    assert square_component is None  # No cables connected

    # Check that module has the expected output ports
    assert hasattr(module, "sine_port")
    assert hasattr(module, "triangle_port")
    assert hasattr(module, "sawtooth_port")
    assert hasattr(module, "square_port")


def test_volume_interface():
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


def test_volume_with_modulation(caplog):
    """Test volume with modulation input."""
    module = VolumeModule()

    # Create modulation source
    lfo = SineOscillator(1.0, wave_range=(-1, 1))
    assert str(lfo) == "AudioComponent Sine"

    # Create component with modulation
    with caplog.at_level(logging.DEBUG, logger="sonicrack.gui.modules._modulated_base"):
        component = module.create_engine_component(
            input_components=None, modulation_components={"Mod": lfo}
        )

    assert component is not None
    assert "VolumeModule modulator: SineOscillator(Sine)" in caplog.text


def test_mixer_interface():
    """Test mixer implements interface correctly."""
    module = MixerModule()

    # Check module type (property, not method)
    assert module.metadata.category == ModuleCategory.MIXER

    # Check no required inputs (accepts multiple)
    assert not module.get_required_inputs()

    # Check component creation with multiple inputs
    osc1 = SineOscillator(440)
    osc2 = SineOscillator(550)

    component = module.create_engine_component(
        input_components=[osc1, osc2], modulation_components=None
    )
    assert component is not None


def test_infer_port_signal_treats_ambiguous_freq_as_hz():
    assert infer_port_signal("Freq", "input") == PortSignal.FREQUENCY_HZ
    assert infer_port_signal("Freq", "output") == PortSignal.FREQUENCY_HZ
    assert infer_port_signal("V/Oct", "input") == PortSignal.PITCH_CV


def test_modules_explicitly_mark_pitch_cv_ports(qapp):
    del qapp

    vco = ModulatedOscillatorModule()
    tb303 = TB303VoiceModule()
    midi_keyboard = MIDIKeyboardModule()
    sequencer = StepSequencerModule()
    lfo = LFOModule()

    assert vco.freq_input.signal == PortSignal.PITCH_CV
    assert tb303.freq_input.signal == PortSignal.PITCH_CV
    assert midi_keyboard.freq_port.signal == PortSignal.PITCH_CV
    assert sequencer.freq_port.signal == PortSignal.PITCH_CV
    assert lfo.sine_port.signal == PortSignal.CONTROL_CV


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
