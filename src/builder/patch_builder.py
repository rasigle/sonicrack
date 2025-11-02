"""Fluent API builder for creating audio synthesis patches.

This module provides the PatchBuilder class that enables declarative, chainable
patch construction with support for preset saving and loading.
"""

from __future__ import annotations
from typing import Any
import json
from pathlib import Path

from src.engine.oscillator import (
    Oscillator,
    SineOscillator,
    SquareOscillator,
    SawtoothOscillator,
    TriangleOscillator,
)
from src.engine.modulator import ADSREnvelope, Modulator
from src.engine.modulated_oscillator import ModulatedOscillator
from src.engine.composer import Chain, WaveAdder
from src.engine.modifier import Volume, Panner, Clipper, Modifier
from src.constants import DEFAULT_SAMPLE_RATE
from src.utils.logging_config import get_logger

logger = get_logger("builder.patch_builder")


class PatchBuilder:
    """Fluent API builder for creating audio synthesis patches.

    This class provides a chainable interface for constructing complex
    audio synthesis patches. Patches can be built programmatically,
    saved as builder, and loaded from builder.

    Attributes:
        _source: The current signal source (oscillator or composer)
        _modifiers: List of modifiers to apply in chain
        _modulators: Dictionary of modulators for modulation
        _config: Configuration dictionary for preset saving
        _name: Name of the patch
        _description: Description of the patch

    Example:
        >>> from src.builder import PatchBuilder
        >>>
        >>> # Simple sine wave with envelope
        >>> patch = (PatchBuilder("My Synth")
        ...     .sine(440)
        ...     .adsr(0.1, 0.2, 0.7, 0.3)
        ...     .volume(0.5)
        ...     .build())
    """

    def __init__(self, name: str = "Untitled Patch", description: str = ""):
        """Initialize an empty patch builder.

        Args:
            name: Name for this patch (default: "Untitled Patch")
            description: Optional description of the patch
        """
        self._source: Any | None = None
        self._modifiers: list[Modifier] = []
        self._modulators: dict[str, Modulator] = {}
        self._config: dict[str, Any] = {
            "version": "1.0",
            "name": name,
            "description": description,
            "components": []
        }
        self._sample_rate = DEFAULT_SAMPLE_RATE
        self._name = name
        self._description = description

    # ========================================================================
    # Oscillator Creation Methods
    # ========================================================================

    def sine(self, frequency: float, amplitude: float = 1.0, phase: float = 0.0) -> PatchBuilder:
        """Add a sine wave oscillator.

        Args:
            frequency: Oscillator frequency in Hz
            amplitude: Oscillator amplitude (0.0 to 1.0)
            phase: Initial phase in degrees

        Returns:
            Self for method chaining

        Example:
            >>> patch = PatchBuilder().sine(440, amplitude=0.8).build()
        """
        self._source = SineOscillator(frequency, amplitude, phase, sample_rate=self._sample_rate)
        self._config["components"].append({
            "type": "sine_oscillator",
            "frequency": frequency,
            "amplitude": amplitude,
            "phase": phase,
        })
        logger.debug(f"Added sine oscillator: {frequency}Hz, amp={amplitude}")
        return self

    def square(self, frequency: float, amplitude: float = 1.0, phase: float = 0.0) -> PatchBuilder:
        """Add a square wave oscillator.

        Args:
            frequency: Oscillator frequency in Hz
            amplitude: Oscillator amplitude (0.0 to 1.0)
            phase: Initial phase in degrees

        Returns:
            Self for method chaining

        Example:
            >>> patch = PatchBuilder().square(220).volume(0.5).build()
        """
        self._source = SquareOscillator(frequency, amplitude, phase, sample_rate=self._sample_rate)
        self._config["components"].append({
            "type": "square_oscillator",
            "frequency": frequency,
            "amplitude": amplitude,
            "phase": phase,
        })
        logger.debug(f"Added square oscillator: {frequency}Hz, amp={amplitude}")
        return self

    def triangle(self, frequency: float, amplitude: float = 1.0, phase: float = 0.0) -> PatchBuilder:
        """Add a triangle wave oscillator.

        Args:
            frequency: Oscillator frequency in Hz
            amplitude: Oscillator amplitude (0.0 to 1.0)
            phase: Initial phase in degrees

        Returns:
            Self for method chaining

        Example:
            >>> patch = PatchBuilder().triangle(110).build()
        """
        self._source = TriangleOscillator(frequency, amplitude, phase, sample_rate=self._sample_rate)
        self._config["components"].append({
            "type": "triangle_oscillator",
            "frequency": frequency,
            "amplitude": amplitude,
            "phase": phase,
        })
        logger.debug(f"Added triangle oscillator: {frequency}Hz, amp={amplitude}")
        return self

    def sawtooth(self, frequency: float, amplitude: float = 1.0, phase: float = 0.0) -> PatchBuilder:
        """Add a sawtooth wave oscillator.

        Args:
            frequency: Oscillator frequency in Hz
            amplitude: Oscillator amplitude (0.0 to 1.0)
            phase: Initial phase in degrees

        Returns:
            Self for method chaining

        Example:
            >>> patch = PatchBuilder().sawtooth(440).build()
        """
        self._source = SawtoothOscillator(frequency, amplitude, phase, sample_rate=self._sample_rate)
        self._config["components"].append({
            "type": "sawtooth_oscillator",
            "frequency": frequency,
            "amplitude": amplitude,
            "phase": phase,
        })
        logger.debug(f"Added sawtooth oscillator: {frequency}Hz, amp={amplitude}")
        return self

    # ========================================================================
    # Modulation Methods
    # ========================================================================

    def adsr(
        self,
        attack: float,
        decay: float,
        sustain: float,
        release: float,
        target: str = "amplitude"
    ) -> PatchBuilder:
        """Add ADSR envelope modulation.

        Args:
            attack: Attack time in seconds
            decay: Decay time in seconds
            sustain: Sustain level (0.0 to 1.0)
            release: Release time in seconds
            target: Modulation target ("amplitude", "frequency", or "phase")

        Returns:
            Self for method chaining

        Example:
            >>> # Amplitude envelope (default)
            >>> patch = PatchBuilder().sine(440).adsr(0.1, 0.2, 0.7, 0.3).build()
        """
        envelope = ADSREnvelope(attack, decay, sustain, release, sample_rate=self._sample_rate)

        # Store modulator
        self._modulators[f"{target}_mod"] = envelope

        # If we have a source oscillator, wrap it in ModulatedOscillator
        if self._source and isinstance(self._source, Oscillator):
            # Pass envelope as positional argument and specify target via keyword
            if target == "amplitude":
                self._source = ModulatedOscillator(
                    self._source,
                    envelope,
                    amp_mod=lambda base, mod: base * mod
                )
            elif target == "frequency":
                self._source = ModulatedOscillator(
                    self._source,
                    envelope,
                    freq_mod=lambda base, mod: base * mod
                )
            elif target == "phase":
                self._source = ModulatedOscillator(
                    self._source,
                    envelope,
                    phase_mod=lambda base, mod: base + mod
                )

        self._config["components"].append({
            "type": "adsr_envelope",
            "attack": attack,
            "decay": decay,
            "sustain": sustain,
            "release": release,
            "target": target,
        })
        logger.debug(f"Added ADSR envelope: A={attack}, D={decay}, S={sustain}, R={release}, target={target}")
        return self

    def with_modulator(
        self,
        modulator: Modulator,
        target: str = "amplitude",
        depth: float = 1.0
    ) -> PatchBuilder:
        """Add custom modulator.

        Args:
            modulator: Modulator instance
            target: Modulation target ("amplitude", "frequency", or "phase")
            depth: Modulation depth (0.0 to 1.0)

        Returns:
            Self for method chaining

        Example:
            >>> from src.engine import SineOscillator
            >>> lfo = SineOscillator(5, amplitude=0.5)
            >>> patch = (PatchBuilder()
            ...     .sine(440)
            ...     .with_modulator(lfo, target="frequency", depth=0.1)
            ...     .build())
        """
        self._modulators[f"{target}_mod"] = modulator

        if self._source and isinstance(self._source, Oscillator):
            # Pass modulator as positional argument with appropriate function
            if target == "amplitude":
                self._source = ModulatedOscillator(
                    self._source,
                    modulator,
                    amp_mod=lambda base, mod: base * (1 + depth * mod)
                )
            elif target == "frequency":
                self._source = ModulatedOscillator(
                    self._source,
                    modulator,
                    freq_mod=lambda base, mod: base * (1 + depth * mod)
                )
            elif target == "phase":
                self._source = ModulatedOscillator(
                    self._source,
                    modulator,
                    phase_mod=lambda base, mod: base + depth * mod
                )

        self._config["components"].append({
            "type": "custom_modulator",
            "target": target,
            "depth": depth,
        })
        logger.debug(f"Added custom modulator: target={target}, depth={depth}")
        return self

    # ========================================================================
    # Modifier Methods (Effects Chain)
    # ========================================================================

    def volume(self, amplitude: float) -> PatchBuilder:
        """Add volume control.

        Args:
            amplitude: Volume level (0.0 to 1.0+)

        Returns:
            Self for method chaining

        Example:
            >>> patch = PatchBuilder().sine(440).volume(0.5).build()
        """
        self._modifiers.append(Volume(amplitude))
        self._config["components"].append({
            "type": "volume",
            "amplitude": amplitude,
        })
        logger.debug(f"Added volume: {amplitude}")
        return self

    def pan(self, position: float) -> PatchBuilder:
        """Add stereo panning.

        Args:
            position: Pan position (-1.0=left, 0.0=center, 1.0=right)

        Returns:
            Self for method chaining

        Example:
            >>> patch = PatchBuilder().sine(440).pan(-0.5).build()
        """
        self._modifiers.append(Panner(position))
        self._config["components"].append({
            "type": "panner",
            "position": position,
        })
        logger.debug(f"Added panner: {position}")
        return self

    def clip(self, min_val: float = -1.0, max_val: float = 1.0) -> PatchBuilder:
        """Add audio clipping/saturation.

        Args:
            min_val: Minimum clipping threshold
            max_val: Maximum clipping threshold

        Returns:
            Self for method chaining

        Example:
            >>> patch = PatchBuilder().sine(440).clip(-0.5, 0.5).build()
        """
        self._modifiers.append(Clipper((min_val, max_val)))
        self._config["components"].append({
            "type": "clipper",
            "min": min_val,
            "max": max_val,
        })
        logger.debug(f"Added clipper: [{min_val}, {max_val}]")
        return self

    # ========================================================================
    # Composition Methods
    # ========================================================================

    def add_oscillator(self, oscillator: Any) -> PatchBuilder:
        """Add an oscillator to create a multi-oscillator patch (mixing).

        Args:
            oscillator: Oscillator or patch to add

        Returns:
            Self for method chaining

        Example:
            >>> patch = (PatchBuilder()
            ...     .add_oscillator(PatchBuilder().sine(220).build())
            ...     .add_oscillator(PatchBuilder().sine(440).build())
            ...     .add_oscillator(PatchBuilder().sine(880).build())
            ...     .build())
        """
        if self._source is None:
            self._source = [oscillator]
        elif isinstance(self._source, list):
            self._source.append(oscillator)
        else:
            self._source = [self._source, oscillator]

        self._config["components"].append({
            "type": "add_oscillator",
        })
        logger.debug("Added oscillator to mixer")
        return self

    # ========================================================================
    # Name and Description Methods
    # ========================================================================

    def set_name(self, name: str) -> PatchBuilder:
        """Set the patch name.

        Args:
            name: Name for the patch

        Returns:
            Self for method chaining

        Example:
            >>> patch = PatchBuilder().set_name("My Lead Synth").sine(440)
        """
        self._name = name
        self._config["name"] = name
        logger.debug(f"Set patch name to '{name}'")
        return self

    def set_description(self, description: str) -> PatchBuilder:
        """Set the patch description.

        Args:
            description: Description of the patch

        Returns:
            Self for method chaining

        Example:
            >>> patch = (PatchBuilder()
            ...     .set_name("Warm Pad")
            ...     .set_description("Soft, warm pad sound with slow attack")
            ...     .sine(220)
            ...     .adsr(2.0, 1.0, 0.8, 3.0))
        """
        self._description = description
        self._config["description"] = description
        logger.debug(f"Set patch description")
        return self

    def get_name(self) -> str:
        """Get the patch name.

        Returns:
            Patch name
        """
        return self._name

    def get_description(self) -> str:
        """Get the patch description.

        Returns:
            Patch description
        """
        return self._description

    # ========================================================================
    # Component Access Methods
    # ========================================================================

    def get_source(self) -> Any | None:
        """Get the source oscillator/generator.

        Returns:
            Source oscillator or None if not set

        Example:
            >>> patch = PatchBuilder().sine(440)
            >>> source = patch.get_source()
            >>> print(type(source))
            <class 'src.engine.oscillator.SineOscillator'>
        """
        return self._source

    def get_modifiers(self) -> list[Modifier]:
        """Get list of modifiers (effects).

        Returns:
            List of modifier instances

        Example:
            >>> patch = PatchBuilder().sine(440).volume(0.5).pan(0.3)
            >>> modifiers = patch.get_modifiers()
            >>> print(len(modifiers))
            2
        """
        return self._modifiers.copy()

    def get_modulators(self) -> dict[str, Modulator]:
        """Get dictionary of modulators.

        Returns:
            Dictionary mapping modulator names to instances

        Example:
            >>> patch = PatchBuilder().sine(440).adsr(0.1, 0.2, 0.7, 0.3)
            >>> modulators = patch.get_modulators()
            >>> print('amplitude_mod' in modulators)
            True
        """
        return self._modulators.copy()

    def get_components(self) -> dict[str, Any]:
        """Get all components of the patch.

        Returns:
            Dictionary with 'source', 'modifiers', 'modulators', 'name', 'description', and 'sample_rate'

        Example:
            >>> patch = PatchBuilder().sine(440).adsr(0.1, 0.2, 0.7, 0.3).volume(0.5)
            >>> components = patch.get_components()
            >>> print(components.keys())
            dict_keys(['source', 'modifiers', 'modulators', 'name', 'description', 'sample_rate'])
        """
        return {
            'source': self._source,
            'modifiers': self._modifiers.copy(),
            'modulators': self._modulators.copy(),
            'name': self._name,
            'description': self._description,
            'sample_rate': self._sample_rate
        }

    # ========================================================================
    # Patch Inspection and Modification Methods
    # ========================================================================

    def describe(self) -> str:
        """Get a human-readable description of the patch.

        Returns:
            String description of the patch components

        Example:
            >>> patch = PatchBuilder("My Synth").sine(440).adsr(0.1, 0.2, 0.7, 0.3).volume(0.5)
            >>> print(patch.describe())
            Patch: My Synth

            - Sine Oscillator (440.0 Hz, amp=1.0)
            - ADSR Envelope (A=0.1s, D=0.2s, S=0.7, R=0.3s) -> amplitude
            - Volume (0.5)
        """
        lines = [f"Patch: {self._name}"]

        if self._description:
            lines.append(f"Description: {self._description}")

        lines.append("")  # Empty line

        for component in self._config["components"]:
            comp_type = component["type"]

            if comp_type == "sine_oscillator":
                lines.append(f"- Sine Oscillator ({component['frequency']} Hz, amp={component['amplitude']})")
            elif comp_type == "square_oscillator":
                lines.append(f"- Square Oscillator ({component['frequency']} Hz, amp={component['amplitude']})")
            elif comp_type == "triangle_oscillator":
                lines.append(f"- Triangle Oscillator ({component['frequency']} Hz, amp={component['amplitude']})")
            elif comp_type == "sawtooth_oscillator":
                lines.append(f"- Sawtooth Oscillator ({component['frequency']} Hz, amp={component['amplitude']})")
            elif comp_type == "adsr_envelope":
                lines.append(
                    f"- ADSR Envelope (A={component['attack']}s, D={component['decay']}s, "
                    f"S={component['sustain']}, R={component['release']}s) -> {component['target']}"
                )
            elif comp_type == "custom_modulator":
                lines.append(f"- Custom Modulator (target={component['target']}, depth={component['depth']})")
            elif comp_type == "volume":
                lines.append(f"- Volume ({component['amplitude']})")
            elif comp_type == "panner":
                lines.append(f"- Panner (position={component['position']})")
            elif comp_type == "clipper":
                lines.append(f"- Clipper (range=[{component['min']}, {component['max']}])")
            elif comp_type == "add_oscillator":
                lines.append("- Added Oscillator (mixed)")

        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        """Get a summary of patch characteristics.

        Returns:
            Dictionary with patch statistics

        Example:
            >>> patch = PatchBuilder("Test").sine(440).adsr(0.1, 0.2, 0.7, 0.3)
            >>> summary = patch.summary()
            >>> print(summary)
            {'name': 'Test', 'description': '', 'oscillators': 1, 'modulators': 1, 'effects': 0, 'components': 2}
        """
        oscillators = 0
        modulators = 0
        effects = 0

        for component in self._config["components"]:
            comp_type = component["type"]
            if "oscillator" in comp_type:
                oscillators += 1
            elif "envelope" in comp_type or "modulator" in comp_type:
                modulators += 1
            elif comp_type in ["volume", "panner", "clipper"]:
                effects += 1

        return {
            "name": self._name,
            "description": self._description,
            "oscillators": oscillators,
            "modulators": modulators,
            "effects": effects,
            "components": len(self._config["components"]),
            "sample_rate": self._sample_rate
        }

    def clone(self) -> PatchBuilder:
        """Create a copy of this patch builder.

        Returns:
            New PatchBuilder instance with same configuration

        Example:
            >>> original = PatchBuilder().sine(440).volume(0.5)
            >>> copy = original.clone()
            >>> copy.volume(0.7)  # Modify copy without affecting original
        """
        import copy as copy_module
        new_builder = PatchBuilder(name=self._name, description=self._description)
        new_builder._config = copy_module.deepcopy(self._config)
        new_builder._sample_rate = self._sample_rate

        # Rebuild components from config
        for component in self._config.get("components", []):
            comp_type = component["type"]

            if comp_type == "sine_oscillator":
                new_builder._source = SineOscillator(
                    component["frequency"],
                    component["amplitude"],
                    component["phase"],
                    sample_rate=self._sample_rate
                )
            elif comp_type == "square_oscillator":
                new_builder._source = SquareOscillator(
                    component["frequency"],
                    component["amplitude"],
                    component["phase"],
                    sample_rate=self._sample_rate
                )
            elif comp_type == "triangle_oscillator":
                new_builder._source = TriangleOscillator(
                    component["frequency"],
                    component["amplitude"],
                    component["phase"],
                    sample_rate=self._sample_rate
                )
            elif comp_type == "sawtooth_oscillator":
                new_builder._source = SawtoothOscillator(
                    component["frequency"],
                    component["amplitude"],
                    component["phase"],
                    sample_rate=self._sample_rate
                )
            elif comp_type == "volume":
                new_builder._modifiers.append(Volume(component["amplitude"]))
            elif comp_type == "panner":
                new_builder._modifiers.append(Panner(component["position"]))
            elif comp_type == "clipper":
                new_builder._modifiers.append(Clipper((component["min"], component["max"])))

        return new_builder

    def modify_frequency(self, new_frequency: float) -> PatchBuilder:
        """Modify the frequency of the oscillator in this patch.

        Args:
            new_frequency: New frequency in Hz

        Returns:
            Self for method chaining

        Example:
            >>> patch = PatchBuilder().sine(440).volume(0.5)
            >>> patch.modify_frequency(880)  # Change to A5
        """
        # Update config
        for component in self._config["components"]:
            if "oscillator" in component["type"]:
                component["frequency"] = new_frequency
                break

        # Update source if it exists
        if self._source and hasattr(self._source, 'freq'):
            self._source.freq = new_frequency
        elif self._source and hasattr(self._source, 'oscillator'):
            # ModulatedOscillator case
            self._source.oscillator.freq = new_frequency

        logger.debug(f"Modified frequency to {new_frequency}Hz")
        return self

    def modify_amplitude(self, new_amplitude: float) -> PatchBuilder:
        """Modify the amplitude of the oscillator in this patch.

        Args:
            new_amplitude: New amplitude (0.0 to 1.0)

        Returns:
            Self for method chaining

        Example:
            >>> patch = PatchBuilder().sine(440)
            >>> patch.modify_amplitude(0.8)
        """
        # Update config
        for component in self._config["components"]:
            if "oscillator" in component["type"]:
                component["amplitude"] = new_amplitude
                break

        # Update source if it exists
        if self._source and hasattr(self._source, 'amp'):
            self._source.amp = new_amplitude
        elif self._source and hasattr(self._source, 'oscillator'):
            # ModulatedOscillator case
            self._source.oscillator.amp = new_amplitude

        logger.debug(f"Modified amplitude to {new_amplitude}")
        return self

    def clear_effects(self) -> PatchBuilder:
        """Remove all effects (modifiers) from the patch.

        Returns:
            Self for method chaining

        Example:
            >>> patch = PatchBuilder().sine(440).volume(0.5).pan(0.3)
            >>> patch.clear_effects()  # Removes volume and pan
        """
        self._modifiers.clear()

        # Remove effects from config
        self._config["components"] = [
            c for c in self._config["components"]
            if c["type"] not in ["volume", "panner", "clipper"]
        ]

        logger.debug("Cleared all effects")
        return self

    # ========================================================================
    # Configuration Methods
    # ========================================================================

    def set_sample_rate(self, sample_rate: int) -> PatchBuilder:
        """Set the sample rate for the patch.

        Args:
            sample_rate: Sample rate in Hz

        Returns:
            Self for method chaining
        """
        self._sample_rate = sample_rate
        self._config["sample_rate"] = sample_rate
        logger.debug(f"Set sample rate: {sample_rate}Hz")
        return self

    # ========================================================================
    # Build Methods
    # ========================================================================

    def build(self) -> Any:
        """Build the final patch from the configuration.

        Returns:
            Built patch (oscillator, chain, or composer)

        Raises:
            ValueError: If no source oscillator has been added

        Example:
            >>> patch = PatchBuilder().sine(440).volume(0.5).build()
            >>> samples = patch.get_samples(44100)
        """
        if self._source is None:
            raise ValueError("No source oscillator added to patch. Use .sine(), .square(), etc.")

        # Handle multiple oscillators (mixing)
        if isinstance(self._source, list):
            source = WaveAdder(*self._source)
            logger.info(f"Built WaveAdder with {len(self._source)} oscillators")
        else:
            source = self._source

        # Apply modifiers in chain
        if self._modifiers:
            result = Chain(source, *self._modifiers)
            logger.info(f"Built Chain with {len(self._modifiers)} modifiers")
        else:
            result = source

        logger.info("Patch built successfully")
        return result

    # ========================================================================
    # Preset Methods
    # ========================================================================

    def save_preset(self, filepath: str | Path) -> None:
        """Save the current patch configuration as a preset.

        Args:
            filepath: Path to save the preset JSON file

        Example:
            >>> patch = PatchBuilder().sine(440).adsr(0.1, 0.2, 0.7, 0.3)
            >>> patch.save_preset("my_synth.json")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self._config, f, indent=2)

        logger.info(f"Saved preset to {filepath}")

    @classmethod
    def from_preset(cls, filepath: str | Path) -> PatchBuilder:
        """Load a patch configuration from a preset file.

        Args:
            filepath: Path to the preset JSON file

        Returns:
            PatchBuilder instance configured from preset

        Example:
            >>> patch = PatchBuilder.from_preset("my_synth.json").build()
            >>> samples = patch.get_samples(44100)
        """
        filepath = Path(filepath)

        with open(filepath, 'r') as f:
            config = json.load(f)

        # Create builder with name and description from config
        name = config.get("name", "Untitled Patch")
        description = config.get("description", "")
        builder = cls(name=name, description=description)

        # Set sample rate if specified
        if "sample_rate" in config:
            builder.set_sample_rate(config["sample_rate"])

        # Reconstruct patch from components
        for component in config.get("components", []):
            comp_type = component["type"]

            if comp_type == "sine_oscillator":
                builder.sine(
                    component["frequency"],
                    component.get("amplitude", 1.0),
                    component.get("phase", 0.0)
                )
            elif comp_type == "square_oscillator":
                builder.square(
                    component["frequency"],
                    component.get("amplitude", 1.0),
                    component.get("phase", 0.0)
                )
            elif comp_type == "triangle_oscillator":
                builder.triangle(
                    component["frequency"],
                    component.get("amplitude", 1.0),
                    component.get("phase", 0.0)
                )
            elif comp_type == "sawtooth_oscillator":
                builder.sawtooth(
                    component["frequency"],
                    component.get("amplitude", 1.0),
                    component.get("phase", 0.0)
                )
            elif comp_type == "adsr_envelope":
                builder.adsr(
                    component["attack"],
                    component["decay"],
                    component["sustain"],
                    component["release"],
                    component.get("target", "amplitude")
                )
            elif comp_type == "volume":
                builder.volume(component["amplitude"])
            elif comp_type == "panner":
                builder.pan(component["position"])
            elif comp_type == "clipper":
                builder.clip(component["min"], component["max"])

        logger.info(f"Loaded preset from {filepath}")
        return builder

    def get_config(self) -> dict[str, Any]:
        """Get the current patch configuration.

        Returns:
            Configuration dictionary
        """
        return self._config.copy()

