"""Enhanced parameter system with runtime validation, smoothing, and automation support.

This module provides a single source of truth for parameter behavior across UI, presets,
and runtime engine components. All parameter constraints, smoothing policies, and
automation semantics are centralized in ParameterDescriptor.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Any

import numpy as np

from src.engine.utils.ramping import consume_linear_ramp, duration_ms_to_samples


class SmoothingPolicy(Enum):
    """Defines how parameter changes should be smoothed to prevent clicks/pops."""

    NONE = "none"  # Instant parameter change (e.g., mode switches, sample_rate)
    LINEAR = "linear"  # Linear ramp over smoothing_duration_ms
    EXPONENTIAL = "exponential"  # Exponential curve (for frequency, amplitude)
    LOGARITHMIC = "logarithmic"  # Logarithmic curve (for dB values)


class AutomationMode(Enum):
    """Defines parameter automation/modulation capabilities."""

    NONE = "none"  # Cannot be automated (e.g., sample_rate, mode switches)
    CONTROL_RATE = "control_rate"  # Updated per-buffer (typical UI parameters)
    AUDIO_RATE = "audio_rate"  # Can be modulated per-sample (e.g., FM, AM)


@dataclass(frozen=True)
class ParameterDescriptor:
    """Complete metadata describing a component parameter.

    This descriptor serves as the single source of truth for all parameter behavior:
    - Runtime validation (type, range, choices)
    - UI presentation (unit, description)
    - Preset serialization
    - Smoothing policy (anti-click behavior)
    - Automation semantics (control vs audio rate)

    Attributes:
        name: Parameter identifier
        default: Default value
        minimum: Minimum allowed value (None = no minimum)
        maximum: Maximum allowed value (None = no maximum)
        unit: Physical unit (Hz, dB, ms, s, etc.)
        clamp: Whether to clamp out-of-range values instead of raising error
        choices: Valid discrete values (for enums, modes)
        description: Human-readable description
        smoothing_policy: How parameter changes should be smoothed
        smoothing_duration_ms: Default smoothing duration in milliseconds
        automation_mode: Whether/how the parameter can be automated
        is_required: Whether parameter must be provided during construction
        validator: Optional custom validation function
    """

    name: str
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    unit: str | None = None
    clamp: bool = False
    choices: tuple[Any, ...] | None = None
    description: str = ""
    smoothing_policy: SmoothingPolicy = SmoothingPolicy.NONE
    smoothing_duration_ms: float = 10.0
    automation_mode: AutomationMode = AutomationMode.CONTROL_RATE
    is_required: bool = False
    validator: Callable[[Any], Any] | None = None

    def validate(self, value: Any, *, param_name: str | None = None) -> Any:
        """Validate a parameter value according to descriptor constraints.

        Args:
            value: The value to validate
            param_name: Optional parameter name for error messages
                (defaults to self.name)

        Returns:
            Validated (and potentially coerced) value

        Raises:
            TypeError: If value is wrong type
            ValueError: If value is out of range or not in choices
        """
        param_name = param_name or self.name

        # Custom validator takes precedence
        if self.validator is not None:
            return self.validator(value)

        # Check for required parameters
        if self.is_required and value is None:
            raise ValueError(f"{param_name} is required but got None")

        # None is allowed for optional parameters
        if value is None:
            return self.default

        # Handle choices (discrete values)
        if self.choices is not None:
            if value not in self.choices:
                choices_str = ", ".join(repr(c) for c in self.choices)
                raise ValueError(
                    f"{param_name} must be one of [{choices_str}], got {value!r}"
                )
            return value

        # Numeric validation
        if self.minimum is not None or self.maximum is not None:
            return self._validate_numeric(value, param_name)

        return value

    def _validate_numeric(self, value: Any, param_name: str) -> float:
        """Validate numeric parameter with range checking."""
        # Type check
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                f"{param_name} must be a real number, got {type(value).__name__}"
            )

        result = float(value)

        # Check for NaN/Inf
        if not math.isfinite(result):
            raise ValueError(f"{param_name} must be finite, got {value!r}")

        # Range check with optional clamping
        has_min = self.minimum is not None
        has_max = self.maximum is not None

        # Use a small epsilon for floating-point comparisons to handle numerical
        # precision
        epsilon = 1e-10

        if has_min and has_max:
            if self.clamp:
                result = max(self.minimum, min(self.maximum, result))
            elif not (self.minimum - epsilon <= result <= self.maximum + epsilon):
                raise ValueError(
                    f"{param_name} must be between {self.minimum} and {self.maximum}, "
                    f"got {value!r}"
                )
        elif has_min:
            if self.clamp:
                result = max(self.minimum, result)
            elif result < self.minimum - epsilon:
                raise ValueError(
                    f"{param_name} must be >= {self.minimum}, got {value!r}"
                )
            # Clamp to minimum if within epsilon tolerance
            if result < self.minimum:
                result = self.minimum
        elif has_max:
            if self.clamp:
                result = min(self.maximum, result)
            elif result > self.maximum + epsilon:
                raise ValueError(
                    f"{param_name} must be <= {self.maximum}, got {value!r}"
                )
            # Clamp to maximum if within epsilon tolerance
            if result > self.maximum:
                result = self.maximum

        return result

    def needs_smoothing(self) -> bool:
        """Check if this parameter requires smoothing for changes."""
        return self.smoothing_policy != SmoothingPolicy.NONE

    def is_automatable(self) -> bool:
        """Check if this parameter can be automated/modulated."""
        return self.automation_mode != AutomationMode.NONE


@dataclass
class RuntimeParameter:
    """Runtime parameter wrapper with automatic validation and smoothing.

    This class wraps a parameter value and enforces the policies defined in
    its ParameterDescriptor. It handles:
    - Automatic validation on value changes
    - Smooth interpolation between values (anti-click)
    - Sample-accurate automation support

    Attributes:
        descriptor: The parameter descriptor defining behavior
        current_value: The current smoothed value
        target_value: The target value being interpolated toward
        _smoothing_samples_remaining: Samples left in current smooth transition
        _smoothing_duration_samples: Total smoothing duration in samples
        _sample_rate: Current sample rate
    """

    descriptor: ParameterDescriptor
    _sample_rate: float = field(default=44100.0)
    _current_value: Any = field(init=False, default=None)
    _target_value: Any = field(init=False, default=None)
    _smoothing_start_value: Any = field(init=False, default=None)
    _smoothing_samples_remaining: int = field(init=False, default=0)
    _smoothing_duration_samples: int = field(init=False, default=0)

    def __init__(self, descriptor: ParameterDescriptor, sample_rate: float = 44100.0):
        """Initialize runtime parameter.

        Args:
            descriptor: Parameter descriptor defining behavior
            sample_rate: Sample rate for smoothing calculations
        """
        self.descriptor = descriptor
        self._sample_rate = sample_rate

        # Initialize with default value
        validated_default = self.descriptor.validate(self.descriptor.default)
        self._current_value = validated_default
        self._target_value = validated_default
        self._smoothing_start_value = validated_default
        self._smoothing_samples_remaining = 0
        self._smoothing_duration_samples = 0
        self._update_smoothing_duration()

    def set_sample_rate(self, sample_rate: float) -> None:
        """Update sample rate and recalculate smoothing duration."""
        self._sample_rate = sample_rate
        self._update_smoothing_duration()

    def _update_smoothing_duration(self) -> None:
        """Recalculate smoothing duration in samples."""
        if self.descriptor.needs_smoothing():
            self._smoothing_duration_samples = duration_ms_to_samples(
                self._sample_rate,
                self.descriptor.smoothing_duration_ms,
                name=f"{self.descriptor.name}_smoothing",
            )
        else:
            self._smoothing_duration_samples = 0

    @property
    def value(self) -> Any:
        """Get the current value (target for non-smoothed, current for smoothed)."""
        if self.descriptor.needs_smoothing():
            return self._current_value
        return self._target_value

    @value.setter
    def value(self, new_value: Any) -> None:
        """Set parameter value with validation and optional smoothing."""
        validated = self.descriptor.validate(new_value)

        if self.descriptor.needs_smoothing() and isinstance(validated, Real):
            # Trigger smooth transition
            self._target_value = validated
            if self._target_value != self._current_value:
                self._smoothing_start_value = self._current_value
                self._smoothing_samples_remaining = self._smoothing_duration_samples
        else:
            # Instant change
            self._target_value = validated
            self._current_value = validated
            self._smoothing_start_value = validated

    @property
    def target(self) -> Any:
        """Get the target value (what it's interpolating toward)."""
        return self._target_value

    @property
    def is_smoothing(self) -> bool:
        """Check if currently smoothing between values."""
        return self._smoothing_samples_remaining > 0

    def advance_smoothing(self, num_samples: int = 1) -> None:
        """Advance smoothing by num_samples.

        Args:
            num_samples: Number of samples to advance
        """
        if not self.is_smoothing:
            return

        if not isinstance(self._current_value, Real) or not isinstance(
            self._target_value, Real
        ):
            # Can't smooth non-numeric values
            self._current_value = self._target_value
            self._smoothing_samples_remaining = 0
            return

        # Calculate how much to advance
        samples_to_process = min(num_samples, self._smoothing_samples_remaining)

        if samples_to_process >= self._smoothing_samples_remaining:
            # Smoothing complete
            self._current_value = self._target_value
            self._smoothing_samples_remaining = 0
        else:
            # Interpolate based on policy
            self._current_value = self._interpolate(samples_to_process)
            self._smoothing_samples_remaining -= samples_to_process

    def _interpolate(self, samples_advanced: int) -> float:
        """Interpolate value based on smoothing policy."""
        start = float(self._smoothing_start_value)
        target = float(self._target_value)

        # Calculate how far through the smoothing we are
        total_duration = self._smoothing_duration_samples
        if total_duration <= 0:
            return target

        # Progress from 0.0 (start) to 1.0 (target)
        samples_elapsed = (
            total_duration - self._smoothing_samples_remaining + samples_advanced
        )
        progress = min(1.0, samples_elapsed / total_duration)

        policy = self.descriptor.smoothing_policy

        if policy == SmoothingPolicy.LINEAR:
            # Linear interpolation
            return start + (target - start) * progress

        elif policy == SmoothingPolicy.EXPONENTIAL:
            # Exponential curve (good for frequency, amplitude)
            if start <= 0:
                return target * progress
            ratio = target / start if start > 0 else 1.0
            return start * (ratio**progress)

        elif policy == SmoothingPolicy.LOGARITHMIC:
            # Logarithmic curve (good for dB)
            if start <= 0 or target <= 0:
                return start + (target - start) * progress
            return start * math.exp(math.log(target / start) * progress)

        else:
            # Fallback to linear
            return start + (target - start) * progress

    def get_interpolated_buffer(self, num_samples: int) -> np.ndarray:
        """Get a buffer of interpolated values for sample-accurate automation.

        Args:
            num_samples: Number of samples to generate

        Returns:
            Array of interpolated values
        """
        if not self.is_smoothing:
            # No smoothing needed, return constant
            return np.full(num_samples, self._current_value, dtype=np.float32)

        # Use efficient vectorized ramping
        policy = self.descriptor.smoothing_policy
        start = float(self._smoothing_start_value)
        target = float(self._target_value)

        if policy == SmoothingPolicy.LINEAR:
            # Use the efficient consume_linear_ramp for LINEAR policy
            buffer, self._current_value, self._smoothing_samples_remaining = (
                consume_linear_ramp(
                    self._current_value,
                    target,
                    self._smoothing_samples_remaining,
                    num_samples,
                )
            )
            return buffer

        # For EXPONENTIAL and LOGARITHMIC, generate full buffer at once
        samples_to_process = min(num_samples, self._smoothing_samples_remaining)
        buffer = np.empty(num_samples, dtype=np.float32)

        if samples_to_process > 0:
            # Calculate progress for each sample
            total_duration = self._smoothing_duration_samples
            samples_before = total_duration - self._smoothing_samples_remaining
            positions = np.arange(
                samples_before + 1,
                samples_before + samples_to_process + 1,
                dtype=np.float64,
            )
            progress = positions / total_duration

            if policy == SmoothingPolicy.EXPONENTIAL:
                # Exponential curve
                if start <= 0:
                    buffer[:samples_to_process] = target * progress
                else:
                    ratio = target / start if start > 0 else 1.0
                    buffer[:samples_to_process] = start * (ratio**progress)

            elif policy == SmoothingPolicy.LOGARITHMIC:
                # Logarithmic curve
                if start <= 0 or target <= 0:
                    buffer[:samples_to_process] = start + (target - start) * progress
                else:
                    buffer[:samples_to_process] = start * np.exp(
                        np.log(target / start) * progress
                    )

            # Update state
            self._current_value = float(buffer[samples_to_process - 1])
            self._smoothing_samples_remaining -= samples_to_process

            if self._smoothing_samples_remaining <= 0:
                self._current_value = target

        # Fill remaining with target value if any
        if samples_to_process < num_samples:
            buffer[samples_to_process:] = target

        return buffer.astype(np.float32)


class ParameterRegistry:
    """Registry for managing runtime parameters for a component.

    This class helps components manage multiple parameters with consistent
    validation and smoothing behavior.
    """

    def __init__(
        self, descriptors: dict[str, ParameterDescriptor], sample_rate: float = 44100.0
    ):
        """Initialize parameter registry.

        Args:
            descriptors: Dictionary mapping parameter names to descriptors
            sample_rate: Sample rate for smoothing calculations
        """
        self._descriptors = descriptors
        self._sample_rate = sample_rate
        self._parameters: dict[str, RuntimeParameter] = {}

        # Initialize all parameters
        for name, descriptor in descriptors.items():
            param = RuntimeParameter(descriptor, sample_rate)
            self._parameters[name] = param

    def set(self, name: str, value: Any) -> None:
        """Set a parameter value with validation."""
        if name not in self._parameters:
            raise KeyError(f"Unknown parameter: {name}")
        self._parameters[name].value = value

    def get(self, name: str) -> Any:
        """Get current parameter value."""
        if name not in self._parameters:
            raise KeyError(f"Unknown parameter: {name}")
        return self._parameters[name].value

    def get_parameter(self, name: str) -> RuntimeParameter:
        """Get the RuntimeParameter wrapper."""
        if name not in self._parameters:
            raise KeyError(f"Unknown parameter: {name}")
        return self._parameters[name]

    def set_sample_rate(self, sample_rate: float) -> None:
        """Update sample rate for all parameters."""
        self._sample_rate = sample_rate
        for param in self._parameters.values():
            param.set_sample_rate(sample_rate)

    def advance_all_smoothing(self, num_samples: int = 1) -> None:
        """Advance smoothing for all parameters."""
        for param in self._parameters.values():
            param.advance_smoothing(num_samples)

    def has_any_smoothing(self) -> bool:
        """Check if any parameter is currently smoothing."""
        return any(param.is_smoothing for param in self._parameters.values())
