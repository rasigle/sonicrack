"""Registry-based PatchBuilder implementation.

This module provides a refactored PatchBuilder that uses the component registry
system for maximum maintainability and extensibility. Component-specific logic
is factored out into the registry, making the builder class generic and
automatically extensible when new components are registered.

The builder automatically generates methods for all registered components,
eliminating the need to manually add methods when new components are created.

Example:
    >>> from builder import PatchBuilder
    >>>
    >>> # Methods are automatically available for all registered components
    >>> patch = (PatchBuilder("My Synth")
    ...     .sine(440, amplitude=0.8)
    ...     .adsr(0.1, 0.2, 0.7, 0.3)
    ...     .volume(0.6)
    ...     .build())
"""

from __future__ import annotations
from typing import Any, Dict, List
import json
from pathlib import Path

from src.builder.component_registry import registry, ComponentCategory
from src.builder import default_components  # Auto-registers components
from src.engine.modulated_oscillator import ModulatedOscillator
from src.engine.composer import Chain, WaveAdder
from src.engine.modifier import Modifier
from src.engine.modulator import Modulator
from src.engine.oscillator import Oscillator
from src.constants import DEFAULT_SAMPLE_RATE
from src.utils.logging_config import get_logger

logger = get_logger("builder.registry_patch_builder")


class PatchBuilder:
    """Registry-based fluent API builder for audio synthesis patches.

    This builder uses the component registry system to automatically support
    all registered components without requiring manual method implementation
    for each component type.

    New components can be added by simply registering them - no changes to
    this class are required.

    Attributes:
        _source: The current signal source (oscillator or composer)
        _modifiers: List of modifiers to apply in chain
        _modulators: Dictionary of modulators for modulation
        _config: Configuration dictionary for preset saving
        _name: Name of the patch
        _description: Description of the patch
        _sample_rate: Sample rate for the patch
    """

    def __init__(self, name: str = "Untitled Patch", description: str = ""):
        """Initialize an empty patch builder.

        Args:
            name: Name for this patch
            description: Optional description of the patch
        """
        self._source: Any | None = None
        self._modifiers: List[Modifier] = []
        self._modulators: Dict[str, Modulator] = {}
        self._config: Dict[str, Any] = {
            "version": "1.0",
            "name": name,
            "description": description,
            "components": []
        }
        self._sample_rate = DEFAULT_SAMPLE_RATE
        self._name = name
        self._description = description

    def add_oscillator(self, comp_name: str, *args, **kwargs) -> PatchBuilder:
        """Generic method to add an oscillator component.

        Args:
            comp_name: Component name from registry
            *args: Positional arguments for component
            **kwargs: Keyword arguments for component

        Returns:
            Self for method chaining
        """
        descriptor = registry.get(comp_name)
        if not descriptor:
            raise ValueError(f"Unknown component: {comp_name}")

        # Inject sample_rate if not provided
        if "sample_rate" not in kwargs:
            kwargs["sample_rate"] = self._sample_rate

        # Create instance
        self._source = descriptor.create_instance(*args, **kwargs)

        # Add to config
        config = descriptor.to_config(*args, **kwargs)
        self._config["components"].append(config)

        logger.debug(f"Added {comp_name}: {args}, {kwargs}")
        return self

    def _add_modifier(self, comp_name: str, *args, **kwargs) -> PatchBuilder:
        """Generic method to add a modifier component.

        Args:
            comp_name: Component name from registry
            *args: Positional arguments for component
            **kwargs: Keyword arguments for component

        Returns:
            Self for method chaining
        """
        descriptor = registry.get(comp_name)
        if not descriptor:
            raise ValueError(f"Unknown component: {comp_name}")

        # Handle clipper special case (expects tuple)
        if comp_name == "clipper" and len(args) == 2:
            args = ((args[0], args[1]),)

        # Create instance
        instance = descriptor.create_instance(*args, **kwargs)
        self._modifiers.append(instance)

        # Add to config (with special handling for clipper)
        if comp_name == "clipper":
            if len(args) > 0 and isinstance(args[0], tuple):
                config = {"type": comp_name, "min": args[0][0], "max": args[0][1]}
            else:
                config = descriptor.to_config(*args, **kwargs)
        else:
            config = descriptor.to_config(*args, **kwargs)

        self._config["components"].append(config)

        logger.debug(f"Added {comp_name}: {args}, {kwargs}")
        return self

    def _add_modulator(
        self,
        comp_name: str,
        *args,
        target: str = "amplitude",
        **kwargs
    ) -> PatchBuilder:
        """Generic method to add a modulator component.

        Args:
            comp_name: Component name from registry
            *args: Positional arguments for component
            target: Modulation target ("amplitude", "frequency", or "phase")
            **kwargs: Keyword arguments for component

        Returns:
            Self for method chaining
        """
        descriptor = registry.get(comp_name)
        if not descriptor:
            raise ValueError(f"Unknown component: {comp_name}")

        # Inject sample_rate if not provided
        if "sample_rate" not in kwargs:
            kwargs["sample_rate"] = self._sample_rate

        # Create instance
        modulator = descriptor.create_instance(*args, **kwargs)

        # Store modulator
        self._modulators[f"{target}_mod"] = modulator

        # Wrap source if it's an oscillator
        if self._source and isinstance(self._source, Oscillator):
            if target == "amplitude":
                self._source = ModulatedOscillator(
                    self._source,
                    modulator,
                    amp_mod=lambda base, mod: base * mod
                )
            elif target == "frequency":
                self._source = ModulatedOscillator(
                    self._source,
                    modulator,
                    freq_mod=lambda base, mod: base * mod
                )
            elif target == "phase":
                self._source = ModulatedOscillator(
                    self._source,
                    modulator,
                    phase_mod=lambda base, mod: base + mod
                )

        # Add to config
        config = descriptor.to_config(*args, **kwargs)
        config["target"] = target
        self._config["components"].append(config)

        logger.debug(f"Added {comp_name} modulator: target={target}")
        return self

    # ========================================================================
    # Auto-generated Component Methods
    # ========================================================================

    def __getattr__(self, name: str):
        """Dynamically generate methods for registered components.

        This method intercepts attribute access and creates component methods
        on-the-fly based on the registry, eliminating the need for manually
        defining each component method.

        Args:
            name: Method name being accessed

        Returns:
            Method that creates the appropriate component

        Raises:
            AttributeError: If method doesn't correspond to a registered component
        """
        # Check if this is a registered component method
        for comp_name, descriptor in registry._components.items():
            if descriptor.method_name == name:
                # Determine component type and return appropriate handler
                if descriptor.category == ComponentCategory.OSCILLATOR:
                    return lambda *args, **kwargs: self.add_oscillator(comp_name, *args, **kwargs)
                elif descriptor.category == ComponentCategory.MODIFIER:
                    return lambda *args, **kwargs: self._add_modifier(comp_name, *args, **kwargs)
                elif descriptor.category == ComponentCategory.MODULATOR:
                    return lambda *args, **kwargs: self._add_modulator(comp_name, *args, **kwargs)

        # Special case for ADSR
        if name == "adsr":
            return self._adsr_helper

        # Not a component method
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def _adsr_helper(self, attack: float, decay: float, sustain: float,
                    release: float, target: str = "amplitude") -> PatchBuilder:
        """Helper for ADSR envelope with cleaner parameter names.

        Args:
            attack: Attack time in seconds
            decay: Decay time in seconds
            sustain: Sustain level (0.0 to 1.0)
            release: Release time in seconds
            target: Modulation target

        Returns:
            Self for method chaining
        """
        return self._add_modulator(
            "adsr_envelope",
            attack, decay, sustain, release,
            target=target
        )

    # ========================================================================
    # Metadata Methods
    # ========================================================================

    def set_name(self, name: str) -> PatchBuilder:
        """Set the patch name."""
        self._name = name
        self._config["name"] = name
        logger.debug(f"Set patch name to '{name}'")
        return self

    def set_description(self, description: str) -> PatchBuilder:
        """Set the patch description."""
        self._description = description
        self._config["description"] = description
        logger.debug(f"Set patch description")
        return self

    def get_name(self) -> str:
        """Get the patch name."""
        return self._name

    def get_description(self) -> str:
        """Get the patch description."""
        return self._description

    def set_sample_rate(self, sample_rate: int) -> PatchBuilder:
        """Set the sample rate."""
        self._sample_rate = sample_rate
        self._config["sample_rate"] = sample_rate
        logger.debug(f"Set sample rate: {sample_rate}Hz")
        return self

    # ========================================================================
    # Component Access Methods
    # ========================================================================

    def get_source(self) -> Any | None:
        """Get the source oscillator/generator."""
        return self._source

    def get_modifiers(self) -> List[Modifier]:
        """Get list of modifiers (effects)."""
        return self._modifiers.copy()

    def get_modulators(self) -> Dict[str, Modulator]:
        """Get dictionary of modulators."""
        return self._modulators.copy()

    def get_components(self) -> Dict[str, Any]:
        """Get all components of the patch."""
        return {
            'source': self._source,
            'modifiers': self._modifiers.copy(),
            'modulators': self._modulators.copy(),
            'name': self._name,
            'description': self._description,
            'sample_rate': self._sample_rate
        }

    def get_config(self) -> Dict[str, Any]:
        """Get the current patch configuration."""
        return self._config.copy()

    # ========================================================================
    # Inspection Methods
    # ========================================================================

    def describe(self) -> str:
        """Get a human-readable description of the patch."""
        lines = [f"Patch: {self._name}"]

        if self._description:
            lines.append(f"Description: {self._description}")

        lines.append("")

        for component in self._config["components"]:
            comp_type = component["type"]
            descriptor = registry.get(comp_type)

            if descriptor:
                # Build parameter string
                params = []
                for param in descriptor.config_params:
                    if param in component and param not in ["sample_rate", "target"]:
                        params.append(f"{param}={component[param]}")

                param_str = ", ".join(params)

                # Add target if present
                if "target" in component:
                    param_str += f" -> {component['target']}"

                lines.append(f"- {descriptor.description or comp_type} ({param_str})")
            else:
                lines.append(f"- {comp_type}")

        return "\n".join(lines)

    def summary(self) -> Dict[str, Any]:
        """Get a summary of patch characteristics."""
        oscillators = 0
        modulators = 0
        effects = 0

        for component in self._config["components"]:
            descriptor = registry.get(component["type"])
            if descriptor:
                if descriptor.category == ComponentCategory.OSCILLATOR:
                    oscillators += 1
                elif descriptor.category == ComponentCategory.MODULATOR:
                    modulators += 1
                elif descriptor.category == ComponentCategory.MODIFIER:
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

    # ========================================================================
    # Modification Methods
    # ========================================================================

    def modify_frequency(self, new_frequency: float) -> PatchBuilder:
        """Modify the frequency of the oscillator."""
        # Update config
        for component in self._config["components"]:
            if "frequency" in component:
                component["frequency"] = new_frequency
                break

        # Update source
        if self._source and hasattr(self._source, 'freq'):
            self._source.freq = new_frequency
        elif self._source and hasattr(self._source, 'oscillator'):
            self._source.oscillator.frequency = new_frequency

        logger.debug(f"Modified frequency to {new_frequency}Hz")
        return self

    def modify_amplitude(self, new_amplitude: float) -> PatchBuilder:
        """Modify the amplitude of the oscillator."""
        # Update config
        for component in self._config["components"]:
            if "amplitude" in component:
                component["amplitude"] = new_amplitude
                break

        # Update source
        if self._source and hasattr(self._source, 'amplitude'):
            self._source.amplitude = new_amplitude
        elif self._source and hasattr(self._source, 'oscillator'):
            self._source.oscillator.amplitude = new_amplitude

        logger.debug(f"Modified amplitude to {new_amplitude}")
        return self

    def clear_effects(self) -> PatchBuilder:
        """Remove all effects (modifiers)."""
        self._modifiers.clear()

        # Remove effects from config
        self._config["components"] = [
            c for c in self._config["components"]
            if registry.get(c["type"]) and
            registry.get(c["type"]).category != ComponentCategory.MODIFIER
        ]

        logger.debug("Cleared all effects")
        return self

    def clone(self) -> PatchBuilder:
        """Create a copy of this patch builder."""
        import copy as copy_module

        new_builder = PatchBuilder(name=self._name, description=self._description)
        new_builder._config = copy_module.deepcopy(self._config)
        new_builder._sample_rate = self._sample_rate

        # Rebuild from config
        for component in self._config.get("components", []):
            comp_type = component["type"]
            descriptor = registry.get(comp_type)

            if descriptor:
                if descriptor.category == ComponentCategory.OSCILLATOR:
                    new_builder._source = registry.create_from_config(
                        component,
                        sample_rate=self._sample_rate
                    )
                elif descriptor.category == ComponentCategory.MODIFIER:
                    instance = registry.create_from_config(component)
                    new_builder._modifiers.append(instance)

        return new_builder

    # ========================================================================
    # Build Methods
    # ========================================================================

    def build(self) -> Any:
        """Build the final patch from the configuration."""
        if self._source is None:
            raise ValueError("No source oscillator added to patch")

        # Handle multiple oscillators
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
        """Save the current patch configuration as a preset."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self._config, f, indent=2)

        logger.info(f"Saved preset to {filepath}")

    @classmethod
    def from_preset(cls, filepath: str | Path) -> PatchBuilder:
        """Load a patch configuration from a preset file."""
        filepath = Path(filepath)

        with open(filepath, 'r') as f:
            config = json.load(f)

        # Create builder
        name = config.get("name", "Untitled Patch")
        description = config.get("description", "")
        builder = cls(name=name, description=description)

        # Set sample rate
        if "sample_rate" in config:
            builder.set_sample_rate(config["sample_rate"])

        # Reconstruct from components using registry
        for component in config.get("components", []):
            comp_type = component["type"]
            descriptor = registry.get(comp_type)

            if not descriptor:
                logger.warning(f"Unknown component type: {comp_type}, skipping")
                continue

            # Extract parameters
            target = component.get("target", "amplitude")

            # Call appropriate method based on category
            if descriptor.category == ComponentCategory.OSCILLATOR:
                method = getattr(builder, descriptor.method_name)
                params = {k: v for k, v in component.items()
                         if k in descriptor.config_params and k != "sample_rate"}
                method(**params)

            elif descriptor.category == ComponentCategory.MODULATOR:
                if comp_type == "adsr_envelope":
                    builder.adsr(
                        component.get("attack_duration", 0.1),
                        component.get("decay_duration", 0.1),
                        component.get("sustain_level", 0.7),
                        component.get("release_duration", 0.3),
                        target=target
                    )

            elif descriptor.category == ComponentCategory.MODIFIER:
                method = getattr(builder, descriptor.method_name)
                if comp_type == "clipper":
                    method(component.get("min", -1.0), component.get("max", 1.0))
                else:
                    params = {k: v for k, v in component.items()
                             if k in descriptor.config_params}
                    method(**params)

        logger.info(f"Loaded preset from {filepath}")
        return builder

