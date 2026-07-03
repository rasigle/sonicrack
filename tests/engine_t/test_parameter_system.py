"""Unit tests for the enhanced parameter system.

Tests cover:
- ParameterDescriptor validation
- RuntimeParameter smoothing and interpolation
- ParameterRegistry management
- Automation semantics
- Smoothing policies
"""

import math
import unittest

import numpy as np
import pytest

from src.engine.core.parameter import (
    AutomationMode,
    ParameterDescriptor,
    ParameterRegistry,
    RuntimeParameter,
    SmoothingPolicy,
)


class TestParameterDescriptor(unittest.TestCase):
    """Tests for ParameterDescriptor validation."""

    def test_validate_numeric_range(self):
        """Test numeric range validation."""
        descriptor = ParameterDescriptor(
            name="test_param",
            default=0.5,
            minimum=0.0,
            maximum=1.0,
        )

        # Valid values
        self.assertEqual(descriptor.validate(0.0), 0.0)
        self.assertEqual(descriptor.validate(0.5), 0.5)
        self.assertEqual(descriptor.validate(1.0), 1.0)

        # Out of range
        with self.assertRaises(ValueError):
            descriptor.validate(-0.1)
        with self.assertRaises(ValueError):
            descriptor.validate(1.1)

    def test_validate_with_clamping(self):
        """Test clamping mode validation."""
        descriptor = ParameterDescriptor(
            name="test_param",
            default=0.5,
            minimum=0.0,
            maximum=1.0,
            clamp=True,
        )

        # Out of range values should be clamped
        self.assertEqual(descriptor.validate(-0.5), 0.0)
        self.assertEqual(descriptor.validate(1.5), 1.0)

    def test_validate_choices(self):
        """Test discrete choice validation."""
        descriptor = ParameterDescriptor(
            name="mode",
            default="linear",
            choices=("linear", "exponential", "logarithmic"),
        )

        # Valid choices
        self.assertEqual(descriptor.validate("linear"), "linear")
        self.assertEqual(descriptor.validate("exponential"), "exponential")

        # Invalid choice
        with self.assertRaises(ValueError) as ctx:
            descriptor.validate("invalid")
        self.assertIn("must be one of", str(ctx.exception))

    def test_validate_rejects_nan_inf(self):
        """Test that NaN and Inf are rejected."""
        descriptor = ParameterDescriptor(
            name="test_param",
            default=0.5,
            minimum=0.0,
            maximum=1.0,
        )

        with self.assertRaises(ValueError):
            descriptor.validate(math.nan)
        with self.assertRaises(ValueError):
            descriptor.validate(math.inf)
        with self.assertRaises(ValueError):
            descriptor.validate(-math.inf)

    def test_validate_type_checking(self):
        """Test type validation for numeric parameters."""
        descriptor = ParameterDescriptor(
            name="test_param",
            default=0.5,
            minimum=0.0,
            maximum=1.0,
        )

        # Should accept numeric types
        self.assertEqual(descriptor.validate(0.5), 0.5)
        self.assertEqual(descriptor.validate(1), 1.0)

        # Should reject non-numeric
        with self.assertRaises(TypeError):
            descriptor.validate("0.5")
        with self.assertRaises(TypeError):
            descriptor.validate(True)

    def test_smoothing_policy_attributes(self):
        """Test smoothing policy configuration."""
        descriptor = ParameterDescriptor(
            name="frequency",
            default=440.0,
            smoothing_policy=SmoothingPolicy.EXPONENTIAL,
            smoothing_duration_ms=10.0,
        )

        self.assertTrue(descriptor.needs_smoothing())
        self.assertEqual(descriptor.smoothing_policy, SmoothingPolicy.EXPONENTIAL)
        self.assertEqual(descriptor.smoothing_duration_ms, 10.0)

        # Non-smoothed parameter
        descriptor_no_smooth = ParameterDescriptor(
            name="mode",
            default="linear",
            smoothing_policy=SmoothingPolicy.NONE,
        )
        self.assertFalse(descriptor_no_smooth.needs_smoothing())

    def test_automation_mode_attributes(self):
        """Test automation mode configuration."""
        # Audio rate automation
        descriptor_audio = ParameterDescriptor(
            name="amplitude",
            default=1.0,
            automation_mode=AutomationMode.AUDIO_RATE,
        )
        self.assertTrue(descriptor_audio.is_automatable())
        self.assertEqual(descriptor_audio.automation_mode, AutomationMode.AUDIO_RATE)

        # No automation
        descriptor_none = ParameterDescriptor(
            name="sample_rate",
            default=44100,
            automation_mode=AutomationMode.NONE,
        )
        self.assertFalse(descriptor_none.is_automatable())


class TestRuntimeParameter(unittest.TestCase):
    """Tests for RuntimeParameter wrapper."""

    def test_initialization(self):
        """Test runtime parameter initialization."""
        descriptor = ParameterDescriptor(
            name="gain",
            default=0.5,
            minimum=0.0,
            maximum=1.0,
        )

        param = RuntimeParameter(descriptor, sample_rate=44100)
        self.assertEqual(param.value, 0.5)
        self.assertEqual(param.target, 0.5)
        self.assertFalse(param.is_smoothing)

    def test_instant_parameter_change(self):
        """Test parameter change without smoothing."""
        descriptor = ParameterDescriptor(
            name="mode",
            default="a",
            choices=("a", "b", "c"),
            smoothing_policy=SmoothingPolicy.NONE,
        )

        param = RuntimeParameter(descriptor)
        param.value = "b"

        self.assertEqual(param.value, "b")
        self.assertFalse(param.is_smoothing)

    def test_smoothed_parameter_change(self):
        """Test parameter change with smoothing."""
        descriptor = ParameterDescriptor(
            name="gain",
            default=0.0,
            minimum=0.0,
            maximum=1.0,
            smoothing_policy=SmoothingPolicy.LINEAR,
            smoothing_duration_ms=10.0,
        )

        param = RuntimeParameter(descriptor, sample_rate=44100)
        param.value = 1.0

        # Should trigger smoothing
        self.assertTrue(param.is_smoothing)
        self.assertEqual(param.target, 1.0)

        # Current value should still be near 0.0
        self.assertLess(param.value, 0.1)

    def test_linear_smoothing(self):
        """Test linear smoothing interpolation."""
        descriptor = ParameterDescriptor(
            name="gain",
            default=0.0,
            minimum=0.0,
            maximum=1.0,
            smoothing_policy=SmoothingPolicy.LINEAR,
            smoothing_duration_ms=10.0,
        )

        param = RuntimeParameter(descriptor, sample_rate=1000)  # Simple rate
        param.value = 1.0

        # 10ms at 1000 Hz = 10 samples
        self.assertEqual(param._smoothing_duration_samples, 10)

        # Advance halfway
        param.advance_smoothing(5)
        self.assertAlmostEqual(param.value, 0.5, places=6)

        # Complete smoothing
        param.advance_smoothing(5)
        self.assertAlmostEqual(param.value, 1.0, places=6)
        self.assertFalse(param.is_smoothing)

    def test_exponential_smoothing(self):
        """Test exponential smoothing for frequency-like parameters."""
        descriptor = ParameterDescriptor(
            name="frequency",
            default=100.0,
            minimum=20.0,
            maximum=20000.0,
            smoothing_policy=SmoothingPolicy.EXPONENTIAL,
            smoothing_duration_ms=10.0,
        )

        param = RuntimeParameter(descriptor, sample_rate=1000)
        param.value = 1000.0

        # Advance partway
        param.advance_smoothing(5)  # Halfway

        # Exponential: should be geometric mean of 100 and 1000
        # sqrt(100 * 1000) = 316.227...
        expected = 100.0 * (10.0**0.5)  # 316.227...
        self.assertAlmostEqual(param.value, expected, places=1)

    def test_validation_on_set(self):
        """Test that validation happens when setting value."""
        descriptor = ParameterDescriptor(
            name="gain",
            default=0.5,
            minimum=0.0,
            maximum=1.0,
        )

        param = RuntimeParameter(descriptor)

        # Valid value
        param.value = 0.8
        self.assertEqual(param.target, 0.8)

        # Invalid value
        with self.assertRaises(ValueError):
            param.value = 1.5

    def test_get_interpolated_buffer(self):
        """Test generating interpolated buffer for automation."""
        descriptor = ParameterDescriptor(
            name="gain",
            default=0.0,
            minimum=0.0,
            maximum=1.0,
            smoothing_policy=SmoothingPolicy.LINEAR,
            smoothing_duration_ms=10.0,
        )

        param = RuntimeParameter(descriptor, sample_rate=1000)
        param.value = 1.0

        # Get buffer of 10 samples (full smoothing duration)
        buffer = param.get_interpolated_buffer(10)

        self.assertEqual(len(buffer), 10)
        self.assertLess(buffer[0], 0.2)  # Start near 0
        self.assertGreater(buffer[-1], 0.8)  # End near 1

        # Should have completed smoothing
        self.assertFalse(param.is_smoothing)

    def test_constant_buffer_when_not_smoothing(self):
        """Test that constant buffer is returned when not smoothing."""
        descriptor = ParameterDescriptor(
            name="gain",
            default=0.5,
            smoothing_policy=SmoothingPolicy.LINEAR,
        )

        param = RuntimeParameter(descriptor)
        # No value change, so no smoothing

        buffer = param.get_interpolated_buffer(100)
        self.assertTrue(np.allclose(buffer, 0.5))


class TestParameterRegistry(unittest.TestCase):
    """Tests for ParameterRegistry."""

    def test_initialization(self):
        """Test parameter registry initialization."""
        descriptors = {
            "gain": ParameterDescriptor(
                name="gain",
                default=0.5,
                minimum=0.0,
                maximum=1.0,
            ),
            "frequency": ParameterDescriptor(
                name="frequency",
                default=440.0,
                minimum=20.0,
                maximum=20000.0,
            ),
        }

        registry = ParameterRegistry(descriptors, sample_rate=44100)

        # Should have created runtime parameters
        self.assertEqual(registry.get("gain"), 0.5)
        self.assertEqual(registry.get("frequency"), 440.0)

    def test_set_and_get(self):
        """Test setting and getting parameter values."""
        descriptors = {
            "gain": ParameterDescriptor(
                name="gain",
                default=0.5,
                minimum=0.0,
                maximum=1.0,
            ),
        }

        registry = ParameterRegistry(descriptors)

        registry.set("gain", 0.8)
        self.assertEqual(registry.get("gain"), 0.8)

    def test_validation_through_registry(self):
        """Test that registry enforces validation."""
        descriptors = {
            "gain": ParameterDescriptor(
                name="gain",
                default=0.5,
                minimum=0.0,
                maximum=1.0,
            ),
        }

        registry = ParameterRegistry(descriptors)

        # Valid value
        registry.set("gain", 0.8)

        # Invalid value
        with self.assertRaises(ValueError):
            registry.set("gain", 1.5)

    def test_unknown_parameter_error(self):
        """Test that accessing unknown parameter raises error."""
        descriptors = {
            "gain": ParameterDescriptor(name="gain", default=0.5),
        }

        registry = ParameterRegistry(descriptors)

        with self.assertRaises(KeyError):
            registry.get("unknown")

        with self.assertRaises(KeyError):
            registry.set("unknown", 0.5)

    def test_sample_rate_propagation(self):
        """Test that sample rate is propagated to all parameters."""
        descriptors = {
            "param1": ParameterDescriptor(
                name="param1",
                default=0.0,
                smoothing_policy=SmoothingPolicy.LINEAR,
                smoothing_duration_ms=10.0,
            ),
            "param2": ParameterDescriptor(
                name="param2",
                default=0.0,
                smoothing_policy=SmoothingPolicy.LINEAR,
                smoothing_duration_ms=10.0,
            ),
        }

        registry = ParameterRegistry(descriptors, sample_rate=44100)
        param1 = registry.get_parameter("param1")
        self.assertEqual(param1._smoothing_duration_samples, 441)  # 10ms at 44.1kHz

        # Change sample rate
        registry.set_sample_rate(48000)
        self.assertEqual(param1._smoothing_duration_samples, 480)  # 10ms at 48kHz

    def test_has_any_smoothing(self):
        """Test detecting if any parameter is smoothing."""
        descriptors = {
            "gain": ParameterDescriptor(
                name="gain",
                default=0.0,
                smoothing_policy=SmoothingPolicy.LINEAR,
                smoothing_duration_ms=10.0,
            ),
        }

        registry = ParameterRegistry(descriptors, sample_rate=1000)

        # No smoothing initially
        self.assertFalse(registry.has_any_smoothing())

        # Trigger smoothing
        registry.set("gain", 1.0)
        self.assertTrue(registry.has_any_smoothing())

        # Advance to completion
        registry.advance_all_smoothing(10)
        self.assertFalse(registry.has_any_smoothing())


@pytest.mark.parametrize(
    "policy,start,end,midpoint_ratio",
    [
        (SmoothingPolicy.LINEAR, 0.0, 1.0, 0.5),
        (SmoothingPolicy.EXPONENTIAL, 100.0, 1000.0, 3.16),  # ~sqrt(10)
        (SmoothingPolicy.LOGARITHMIC, 0.1, 10.0, 1.0),  # Geometric mean
    ],
)
def test_smoothing_policies(policy, start, end, midpoint_ratio):
    """Test different smoothing policies produce expected curves."""
    descriptor = ParameterDescriptor(
        name="test",
        default=start,
        smoothing_policy=policy,
        smoothing_duration_ms=10.0,
    )

    param = RuntimeParameter(descriptor, sample_rate=1000)
    param.value = end

    # Advance halfway
    param.advance_smoothing(5)

    # Check that midpoint is approximately correct
    if policy == SmoothingPolicy.LINEAR:
        expected = start + (end - start) * 0.5
    elif policy == SmoothingPolicy.EXPONENTIAL:
        expected = start * midpoint_ratio
    else:  # LOGARITHMIC
        expected = start * midpoint_ratio

    assert abs(param.value - expected) < abs(end - start) * 0.2


if __name__ == "__main__":
    unittest.main()
