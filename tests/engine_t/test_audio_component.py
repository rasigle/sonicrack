"""Unit tests for ComponentRegistry system.

Tests cover:
- Component registration
- Component retrieval
- Component categories
- Component descriptors
- Registry operations
"""

import unittest

from src.engine.audio_component import (
    ComponentCategory,
    ComponentDescriptor,
    ParameterDescriptor,
)
from src.engine.composer import WaveAdder
from src.engine.effects import Delay, Distortion, Reverb
from src.engine.modifier import (
    Frequency,
    ModulatedClipper,
    ModulatedPanner,
    ModulatedVolume,
    Panner,
    Volume,
)
from src.engine.oscillator_modulated import ModulatedFrequency, ModulatedOscillator
from src.engine.oscillator_square import SquareOscillator


class TestComponentDescriptor(unittest.TestCase):
    """Tests for ComponentDescriptor."""

    def test_descriptor_creation(self):
        """Test creating a component descriptor."""
        descriptor = ComponentDescriptor(
            name="test_component",
            category=ComponentCategory.OSCILLATOR,
            parameters={
                "param1": ParameterDescriptor(
                    name="param1",
                    default=0.5,
                    minimum=0.0,
                    maximum=1.0,
                    clamp=True,
                ),
                "param2": ParameterDescriptor(name="param2", default=1.0),
            },
            description="Test component",
        )

        self.assertEqual(descriptor.name, "test_component")
        self.assertEqual(descriptor.category, ComponentCategory.OSCILLATOR)
        self.assertEqual(descriptor.parameter_names, ["param1", "param2"])
        assert descriptor.parameters is not None
        self.assertTrue(descriptor.parameters["param1"].clamp)
        self.assertEqual(descriptor.description, "Test component")

    def test_to_config(self):
        """Test converting to config dictionary."""
        descriptor = ComponentDescriptor(
            name="test_component",
            category=ComponentCategory.OSCILLATOR,
            parameters={
                "frequency": ParameterDescriptor("frequency", 440.0),
                "amplitude": ParameterDescriptor("amplitude", 1.0),
            },
        )

        config = descriptor.to_config(440, amplitude=0.8)

        self.assertEqual(config["name"], "test_component")
        self.assertEqual(config["category"], "oscillator")
        self.assertEqual(config["frequency"], 440)
        self.assertEqual(config["amplitude"], 0.8)

    def test_to_config_with_kwargs(self):
        """Test config generation with keyword arguments."""
        descriptor = ComponentDescriptor(
            name="test",
            category=ComponentCategory.OSCILLATOR,
            parameters={
                "frequency": ParameterDescriptor("frequency", 440.0),
                "amplitude": ParameterDescriptor("amplitude", 1.0),
                "phase": ParameterDescriptor("phase", 0.0),
            },
        )

        config = descriptor.to_config(frequency=440, amplitude=0.5, phase=90)

        self.assertEqual(config["name"], "test")
        self.assertEqual(config["category"], "oscillator")
        self.assertEqual(config["frequency"], 440)
        self.assertEqual(config["amplitude"], 0.5)
        self.assertEqual(config["phase"], 90)


def test_effect_descriptors_expose_strict_parameter_policy():
    """Effects should publish strict engine-level range policy as metadata."""
    expected = {
        Distortion: {
            "drive": (0.0, 10.0),
            "mix": (0.0, 1.0),
            "output_gain": (0.0, 2.0),
        },
        Delay: {
            "delay_time": (0.001, 2.0),
            "feedback": (0.0, 0.95),
            "mix": (0.0, 1.0),
        },
        Reverb: {
            "room_size": (0.0, 1.0),
            "damping": (0.0, 1.0),
            "mix": (0.0, 1.0),
        },
        Panner: {
            "position": (-1.0, 1.0),
        },
    }

    for component, parameters in expected.items():
        descriptor_parameters = component.descriptor.parameters
        assert descriptor_parameters is not None

        for name, (minimum, maximum) in parameters.items():
            parameter = descriptor_parameters[name]
            assert parameter.minimum == minimum
            assert parameter.maximum == maximum
            assert parameter.clamp is False


def test_volume_descriptor_exposes_sample_rate_smoothing_contract():
    parameters = Volume.descriptor.parameters
    assert parameters is not None

    assert parameters["amplitude"].minimum == 0.0
    assert parameters["sample_rate"].unit == "Hz"
    assert parameters["smoothing_time_ms"].default == 10.0
    assert parameters["smoothing_time_ms"].unit == "ms"


def test_remaining_registry_descriptors_expose_ui_preset_limits():
    expected_names = {
        WaveAdder: ["generators", "stereo", "mix_mode"],
        ModulatedPanner: ["modulator", "sample_rate", "smoothing_time_ms"],
        ModulatedVolume: [
            "modulator",
            "modulation_target",
            "sample_rate",
            "smoothing_time_ms",
        ],
        Frequency: ["frequency"],
        ModulatedClipper: ["modulator"],
        ModulatedOscillator: [
            "oscillator",
            "modulators",
            "amp_mod",
            "freq_mod",
            "phase_mod",
        ],
        ModulatedFrequency: ["oscillator", "modulator", "freq_mod_func"],
        SquareOscillator: [
            "frequency",
            "gain_db",
            "amplitude",
            "phase",
            "sample_rate",
            "wave_range",
            "pulsewidth",
            "mode",
        ],
    }

    for component, parameter_names in expected_names.items():
        assert component.descriptor.parameter_names == parameter_names

    assert WaveAdder.descriptor.parameters is not None
    assert WaveAdder.descriptor.parameters["stereo"].choices == (False, True)
    assert WaveAdder.descriptor.parameters["mix_mode"].choices == ("average", "sum")

    assert ModulatedPanner.descriptor.parameters is not None
    assert ModulatedPanner.descriptor.parameters["smoothing_time_ms"].minimum == 0.0
    assert ModulatedPanner.descriptor.parameters["sample_rate"].unit == "Hz"

    assert ModulatedVolume.descriptor.parameters is not None
    assert ModulatedVolume.descriptor.parameters["modulation_target"].choices == (
        "amplitude",
        "gain_db",
    )

    assert Frequency.descriptor.parameters is not None
    assert Frequency.descriptor.parameters["frequency"].minimum == 0.0

    assert ModulatedClipper.descriptor.parameters is not None
    clipper_modulator = ModulatedClipper.descriptor.parameters["modulator"]
    assert clipper_modulator.minimum == 0.0
    assert clipper_modulator.maximum == 1.0
    assert clipper_modulator.clamp is True

    assert SquareOscillator.descriptor.parameters is not None
    assert SquareOscillator.descriptor.parameters["mode"].choices == (
        "ideal",
        "ideal_smooth",
        "bandlimited",
        "vcv",
        "soft",
        "comparator",
    )
