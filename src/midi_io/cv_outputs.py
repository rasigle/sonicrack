"""MIDI CV output adapters for different signal types.

These adapters wrap the MIDIToCV converter to provide specific outputs
(frequency, gate, velocity) that can be connected to other modules.
"""

from typing import Any

import numpy as np

from src.engine.core.component import AudioComponent
from src.midi_io.midi_to_cv import MIDIToCV


class CVFrequencyOutput(AudioComponent):
    """Outputs frequency CV from MIDI converter."""

    def __init__(self, cv_converter: MIDIToCV):
        """Initialize frequency output.

        Args:
            cv_converter: The MIDIToCV converter to read from
        """
        super().__init__()
        self.cv_converter = cv_converter

    def get_samples(self, n: int, *args: Any, **kwargs: Any) -> np.ndarray:
        """Get frequency samples.

        Args:
            n: Number of samples to generate
            *args: Additional positional arguments (ignored, for compatibility)
            **kwargs: Additional arguments (ignored, for compatibility)

        Returns:
            Array of frequency values in Hz
        """
        _ = args, kwargs
        return self.cv_converter.get_samples(n)

    def __iter__(self):
        """Make component iterable for ModulatedOscillator."""
        return self

    def __next__(self):
        """Return next CV value (constant)."""
        return self.cv_converter.frequency


class CVGateOutput(AudioComponent):
    """Outputs gate CV from MIDI converter."""

    def __init__(self, cv_converter: MIDIToCV):
        """Initialize gate output.

        Args:
            cv_converter: The MIDIToCV converter to read from
        """
        super().__init__()
        self.cv_converter = cv_converter

    def get_samples(self, n: int, *args: Any, **kwargs: Any) -> np.ndarray:
        """Get gate samples.

        Args:
            n: Number of samples to generate
            *args: Additional positional arguments (ignored, for compatibility)
            **kwargs: Additional arguments (ignored, for compatibility)

        Returns:
            Array of gate values (0.0 or 1.0)
        """
        _ = args, kwargs
        return self.cv_converter.get_gate_samples(n)

    def __iter__(self):
        """Make component iterable for ModulatedOscillator."""
        return self

    def __next__(self):
        """Return next CV value (constant)."""
        return self.cv_converter.gate


class CVVelocityOutput(AudioComponent):
    """Outputs velocity CV from MIDI converter."""

    def __init__(self, cv_converter: MIDIToCV):
        """Initialize velocity output.

        Args:
            cv_converter: The MIDIToCV converter to read from
        """
        super().__init__()
        self.cv_converter = cv_converter

    def get_samples(self, n: int, *args: Any, **kwargs: Any) -> np.ndarray:
        """Get velocity samples.

        Args:
            n: Number of samples to generate
            *args: Additional positional arguments (ignored, for compatibility)
            **kwargs: Additional arguments (ignored, for compatibility)

        Returns:
            Array of velocity values (0.0 to 1.0)
        """
        _ = args, kwargs
        return self.cv_converter.get_velocity_samples(n)

    def __iter__(self):
        """Make component iterable for ModulatedOscillator."""
        return self

    def __next__(self):
        """Return next CV value (constant)."""
        return self.cv_converter.velocity
