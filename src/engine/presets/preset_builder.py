"""Simplified PresetBuilder with tree representation.

This module provides an optimized PresetBuilder that:
- Uses the component registry for auto-generated methods
- Generates config on-demand (no duplicate state)
- Provides tree visualization of preset structure
- Pre-caches component methods for performance

Example:
    >>> from src.engine import PresetBuilder
    >>>
    >>> # Build a preset with fluent API
    >>> preset_builder = (PresetBuilder("My Synth")
    ...     .sine(440, amplitude=0.8)
    ...     .adsr(0.1, 0.2, 0.7, 0.3)
    ...     .volume(0.6))
    >>>
    >>> # The saved config includes both a legacy flat component list and a
    >>> # portable graph with explicit audio/modulation routing.
    >>> graph = preset_builder.get_config()["graph"]
    >>> graph["output"]["kind"]
    'chain'
    >>>
    >>> # Build the actual preset
    >>> preset = preset_builder.build()
    >>>
    >>> # Visualize structure
    >>> print(preset_builder.to_tree())
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.constants import DEFAULT_SAMPLE_RATE
from src.engine.core.registry import ComponentCategory, audio_registry

logger = logging.getLogger(__name__)


_MODULATION_OPERATIONS = {
    "amplitude": "multiply",
    "frequency": "multiply",
    "phase": "add",
}


def _multiply_modulation(base, mod):
    return base * mod


def _add_modulation(base, mod):
    return base + mod


class PresetNode:
    """A node in the preset tree representing a component."""

    def __init__(self, component_type: str, component: Any, params: dict[str, Any]):
        """Initialize a preset node.

        Args:
            component_type: Type of component (oscillator, modifier, etc.)
            component: The actual component instance
            params: Parameters used to create this component
        """
        self.component_type = component_type
        self.component = component
        self.params = params
        self.children: list[PresetNode] = []

    def add_child(self, node: PresetNode) -> PresetNode:
        """Add a child node."""
        self.children.append(node)
        return node

    def to_tree_string(self, indent: int = 0, is_last: bool = True) -> list[str]:
        """Convert to tree string representation.

        Args:
            indent: Current indentation level
            is_last: Whether this is the last child

        Returns:
            List of formatted strings representing the tree
        """
        lines = []

        # Format parameters
        param_str = ", ".join(
            f"{k}={v}" for k, v in self.params.items() if k not in ["sample_rate"]
        )

        # Create node line with tree structure
        if indent == 0:
            prefix = ""
        else:
            prefix = "  " * (indent - 1) + ("└── " if is_last else "├── ")

        node_line = f"{prefix}{self.component_type}"
        if param_str:
            node_line += f"({param_str})"
        lines.append(node_line)

        # Add children
        for i, child in enumerate(self.children):
            child_is_last = i == len(self.children) - 1
            child_lines = child.to_tree_string(indent + 1, child_is_last)
            lines.extend(child_lines)

        return lines


class PresetBuilder:
    """Simplified fluent API builder for audio synthesis presets.

    This builder uses the component registry to automatically support
    all registered components without manual method implementation.

    Key features:
    - Auto-generated component methods (pre-cached for performance)
    - On-demand config generation (no duplicate state)
    - Tree visualization of preset structure
    - Simplified internal state

    Attributes:
        _name: Name of the preset
        _description: Description of the preset
        _sample_rate: Sample rate for the preset
        _source: The signal source (oscillator or multiple oscillators)
        _modifiers: List of modifiers to apply in chain
        _component_tree: Tree representation of the preset structure
        _component_methods: Cache of auto-generated component methods
    """

    def __init__(self, name: str = "Untitled Preset", description: str = ""):
        """Initialize an empty preset builder.

        Args:
            name: Name for this preset
            description: Optional description of the preset
        """
        self._name = name
        self._description = description
        self._sample_rate = DEFAULT_SAMPLE_RATE
        self._source: Any | None = None
        self._modifiers: list[Any] = []
        self._component_tree = PresetNode("Preset", None, {"name": name})

        # Pre-cache component methods for performance
        self._component_methods: dict[str, Callable] = {}
        self._generate_component_methods()

    def _generate_component_methods(self) -> None:
        """Pre-generate all component methods for performance.

        This replaces the slow __getattr__ approach with pre-cached methods,
        providing 10x faster method calls.
        """
        audio_registry._ensure_initialized()  # Make sure components are loaded

        for comp_name, component in audio_registry.components.items():
            descriptor = component.descriptor
            method_name = (
                descriptor.fluent_api_name if descriptor.fluent_api_name else comp_name
            )

            # Create appropriate method based on category
            if descriptor.category == ComponentCategory.OSCILLATOR:
                self._component_methods[method_name] = (
                    lambda *args, cn=comp_name, **kwargs: self._add_oscillator(
                        cn, *args, **kwargs
                    )
                )
            elif descriptor.category == ComponentCategory.MODIFIER:
                self._component_methods[method_name] = (
                    lambda *args, cn=comp_name, **kwargs: self._add_modifier(
                        cn, *args, **kwargs
                    )
                )
            elif descriptor.category == ComponentCategory.MODULATOR:
                self._component_methods[method_name] = (
                    lambda *args, cn=comp_name, **kwargs: self._add_modulator(
                        cn, *args, **kwargs
                    )
                )

        logger.debug(f"Pre-generated {len(self._component_methods)} component methods")

    def __getattr__(self, name: str) -> Callable[..., PresetBuilder]:
        """Get pre-cached component method.

        Args:
            name: Method name

        Returns:
            Component

        Raises:
            AttributeError: If component doesn't exist
        """
        if name in self._component_methods:
            return self._component_methods[name]

        raise AttributeError(
            f"Unknown component '{name}'. "
            f"Available components: {', '.join(list(self._component_methods.keys()))}"
        )

    # ========================================================================
    # Component Addition Methods
    # ========================================================================

    def _add_oscillator(self, comp_name: str, *args, **kwargs) -> PresetBuilder:
        """Generic method to add an oscillator component.

        Args:
            comp_name: Component name from registry
            *args: Positional arguments for component
            **kwargs: Keyword arguments for component

        Returns:
            Self for method chaining
        """
        component_class = audio_registry.get(comp_name, strict=True)
        assert component_class is not None

        # Inject sample_rate if not provided
        if "sample_rate" not in kwargs:
            kwargs["sample_rate"] = self._sample_rate

        # Create instance
        instance = component_class(*args, **kwargs)

        # Add to source
        if self._source is None:
            self._source = instance
        elif isinstance(self._source, list):
            self._source.append(instance)
        else:
            self._source = [self._source, instance]

        # Build params dict for tree
        params = component_class.descriptor.to_config(*args, **kwargs)

        # Add to tree
        node = PresetNode(comp_name, instance, params)
        self._component_tree.add_child(node)

        logger.debug(f"Added {comp_name}: {args}, {kwargs}")
        return self

    def _add_modifier(self, comp_name: str, *args, **kwargs) -> PresetBuilder:
        """Generic method to add a modifier component.

        Args:
            comp_name: Component name from registry
            *args: Positional arguments for component
            **kwargs: Keyword arguments for component

        Returns:
            Self for method chaining
        """
        component = audio_registry.get(comp_name, strict=True)
        assert component is not None

        # Create instance
        instance = component(*args, **kwargs)
        self._modifiers.append(instance)

        # Build params dict for tree
        params = component.descriptor.to_config(*args, **kwargs)

        # Add to tree
        node = PresetNode(comp_name, instance, params)
        self._component_tree.add_child(node)

        logger.debug(f"Added {comp_name}: {args}, {kwargs}")
        return self

    def _add_modulator(
        self, comp_name: str, *args, target: str = "amplitude", **kwargs
    ) -> PresetBuilder:
        """Generic method to add a modulator component.

        Args:
            comp_name: Component name from registry
            *args: Positional arguments for component
            target: Modulation target ("amplitude", "frequency", or "phase")
            **kwargs: Keyword arguments for component

        Returns:
            Self for method chaining
        """
        component_class = audio_registry.get(comp_name, strict=True)
        assert component_class is not None

        # Inject sample_rate if not provided
        if "sample_rate" not in kwargs:
            kwargs["sample_rate"] = self._sample_rate

        # Create instance
        modulator = component_class(*args, **kwargs)

        # Wrap source with ModulatedOscillator if we have an oscillator. For a
        # mixed source list, modulation applies to the most recently added source.
        source_to_modulate = None
        source_index = None
        if isinstance(self._source, list) and self._source:
            source_to_modulate = self._source[-1]
            source_index = len(self._source) - 1
        elif self._source is not None:
            source_to_modulate = self._source

        if source_to_modulate is not None and hasattr(source_to_modulate, "frequency"):
            mod_osc_class = audio_registry.get("ModulatedOscillator", strict=True)
            assert mod_osc_class is not None
            if target == "amplitude":
                modulated_source = mod_osc_class(
                    source_to_modulate,
                    modulator,
                    amp_mod=_multiply_modulation,
                )
            elif target == "frequency":
                modulated_source = mod_osc_class(
                    source_to_modulate,
                    modulator,
                    freq_mod=_multiply_modulation,
                )
            elif target == "phase":
                modulated_source = mod_osc_class(
                    source_to_modulate,
                    modulator,
                    phase_mod=_add_modulation,
                )
            else:
                modulated_source = source_to_modulate

            if source_index is None:
                self._source = modulated_source
            else:
                self._source[source_index] = modulated_source

        # Build params dict for tree
        params = component_class.descriptor.to_config(*args, **kwargs)
        params["target"] = target

        # Add to tree
        node = PresetNode(comp_name, modulator, params)
        self._component_tree.add_child(node)

        logger.debug(f"Added {comp_name} modulator: target={target}")
        return self

    # ========================================================================
    # Metadata Methods
    # ========================================================================

    def set_name(self, name: str) -> PresetBuilder:
        """Set the preset name."""
        self._name = name
        self._component_tree.params["name"] = name
        logger.debug(f"Set preset name to '{name}'")
        return self

    def set_description(self, description: str) -> PresetBuilder:
        """Set the preset description."""
        self._description = description
        logger.debug(f"Set preset description: {description}")
        return self

    def get_name(self) -> str:
        """Get the preset name."""
        return self._name

    def get_description(self) -> str:
        """Get the preset description."""
        return self._description

    def set_sample_rate(self, sample_rate: int) -> PresetBuilder:
        """Set the sample rate."""
        self._sample_rate = sample_rate
        logger.debug(f"Set sample rate: {sample_rate}Hz")
        return self

    # ========================================================================
    # Inspection/Access Methods (for backward compatibility)
    # ========================================================================

    def get_source(self) -> Any | None:
        """Get the current source oscillator(s)."""
        return self._source

    def get_modifiers(self) -> list[Any]:
        """Get list of modifiers."""
        return self._modifiers.copy()

    def get_modulators(self) -> dict[str, Any]:
        """Get dictionary of modulators (for backward compatibility).

        Note: In the simplified builder, modulators are wrapped into the source.
        This method tries to extract them for backward compatibility.
        """
        # Check if source is a ModulatedOscillator and extract modulator
        modulators = {}
        sources = self._source if isinstance(self._source, list) else [self._source]
        for source_index, source in enumerate(source for source in sources if source):
            if hasattr(source, "modulators"):
                for mod_index, mod in enumerate(source.modulators):
                    if source_index == 0 and mod_index == 0:
                        key = "amplitude_mod"
                    else:
                        key = f"source_{source_index}_modulator_{mod_index}"
                    modulators[key] = mod
        return modulators

    def get_components(self) -> dict[str, Any]:
        """Get components dictionary (for backward compatibility).

        Returns:
            Dictionary with source, modifiers, modulators, and metadata
        """
        return {
            "source": self._source,
            "modifiers": self._modifiers,
            "modulators": self.get_modulators(),
            "name": self._name,
            "description": self._description,
            "sample_rate": self._sample_rate,
        }

    def add_oscillator(self, comp_name_or_instance, *args, **kwargs) -> PresetBuilder:
        """Add an oscillator (backward compatibility method).

        Args:
            comp_name_or_instance: Component name string or instance
            *args: Arguments
            **kwargs: Keyword arguments

        Returns:
            Self for method chaining
        """
        if isinstance(comp_name_or_instance, str):
            return self._add_oscillator(comp_name_or_instance, *args, **kwargs)
        else:
            # Direct instance - add to source
            if self._source is None:
                self._source = comp_name_or_instance
            elif isinstance(self._source, list):
                self._source.append(comp_name_or_instance)
            else:
                self._source = [self._source, comp_name_or_instance]
            return self

    def modify_amplitude(self, value_or_modulator) -> PresetBuilder:
        """Modify amplitude for all oscillators in the preset.

        Args:
            value_or_modulator: Either a float value to set amplitude directly,
                              or a modulator instance to apply modulation

        Returns:
            Self for method chaining

        Note:
            When multiple oscillators are present, this modifies ALL of them.
        """
        if isinstance(value_or_modulator, (int, float)):
            # Direct amplitude modification - handle both single and multiple
            # oscillators
            if self._source:
                # Normalize to list for uniform handling
                sources = (
                    self._source if isinstance(self._source, list) else [self._source]
                )
                for source in sources:
                    if hasattr(source, "amplitude"):
                        source.amplitude = value_or_modulator

                # Update tree params for all oscillator nodes
                for node in self._component_tree.children:
                    if hasattr(node.component, "amplitude"):
                        node.params["amplitude"] = value_or_modulator
        else:
            # Modulator-based modification
            modulator = value_or_modulator
            if self._source:
                # Handle both single and multiple oscillators
                if isinstance(self._source, list):
                    # Apply modulation to first oscillator only (backward compatibility)
                    if hasattr(self._source[0], "frequency"):
                        mod_osc_desc = audio_registry.get(
                            "ModulatedOscillator", strict=True
                        )
                        assert mod_osc_desc is not None
                        self._source[0] = mod_osc_desc.create_instance(
                            self._source[0],
                            modulator,
                            amp_mod=_multiply_modulation,
                        )
                else:
                    # Single oscillator
                    if hasattr(self._source, "frequency"):
                        mod_osc_desc = audio_registry.get(
                            "ModulatedOscillator", strict=True
                        )
                        assert mod_osc_desc is not None
                        self._source = mod_osc_desc.create_instance(
                            self._source,
                            modulator,
                            amp_mod=_multiply_modulation,
                        )
        return self

    def modify_frequency(self, value_or_modulator) -> PresetBuilder:
        """Modify frequency for all oscillators in the preset.

        Args:
            value_or_modulator: Either a numeric value to set frequency directly,
                              or a modulator instance to apply modulation

        Returns:
            Self for method chaining

        Note:
            When multiple oscillators are present, this modifies ALL of them.
        """
        if isinstance(value_or_modulator, (int, float)):
            # Direct frequency modification - handle both single and multiple o
            # scillators
            if self._source:
                # Normalize to list for uniform handling
                sources = (
                    self._source if isinstance(self._source, list) else [self._source]
                )
                for source in sources:
                    if hasattr(source, "frequency"):
                        source.frequency = value_or_modulator

                # Update tree params for all oscillator nodes
                for node in self._component_tree.children:
                    if hasattr(node.component, "frequency"):
                        node.params["frequency"] = value_or_modulator
        else:
            # Modulator-based modification
            modulator = value_or_modulator
            if self._source:
                # Handle both single and multiple oscillators
                if isinstance(self._source, list):
                    # Apply modulation to first oscillator only (backward compatibility)
                    if hasattr(self._source[0], "frequency"):
                        mod_osc_desc = audio_registry.get(
                            "ModulatedOscillator", strict=True
                        )
                        assert mod_osc_desc is not None
                        self._source[0] = mod_osc_desc.create_instance(
                            self._source[0],
                            modulator,
                            freq_mod=_multiply_modulation,
                        )
                else:
                    # Single oscillator
                    if hasattr(self._source, "frequency"):
                        mod_osc_desc = audio_registry.get(
                            "ModulatedOscillator", strict=True
                        )
                        assert mod_osc_desc is not None
                        self._source = mod_osc_desc.create_instance(
                            self._source,
                            modulator,
                            freq_mod=_multiply_modulation,
                        )
        return self

    def modify_oscillator_at(
        self, index: int, frequency: float | None = None, amplitude: float | None = None
    ) -> PresetBuilder:
        """Modify a specific oscillator by index.

        Args:
            index: Index of the oscillator to modify (0-based)
            frequency: New frequency value (optional)
            amplitude: New amplitude value (optional)

        Returns:
            Self for method chaining

        Raises:
            IndexError: If index is out of range
            ValueError: If no oscillators exist

        Example:
            >>> preset = (PresetBuilder()
            ...     .sine(440)
            ...     .square(550)
            ...     .triangle(660))
            >>> # Change only the second oscillator (square) to 880 Hz
            >>> preset.modify_oscillator_at(1, frequency=880)
        """
        if self._source is None:
            raise ValueError("No oscillators in preset")

        # Normalize to list
        sources = self._source if isinstance(self._source, list) else [self._source]

        if index < 0 or index >= len(sources):
            raise IndexError(
                f"Oscillator index {index} out of range (0-{len(sources) - 1})"
            )

        # Modify the specific oscillator
        osc = sources[index]
        if frequency is not None and hasattr(osc, "frequency"):
            osc.frequency = frequency
        if amplitude is not None and hasattr(osc, "amplitude"):
            osc.amplitude = amplitude

        # Update tree params for this specific oscillator
        osc_nodes = [
            node
            for node in self._component_tree.children
            if self._get_node_category(node.component_type)
            == ComponentCategory.OSCILLATOR
        ]
        if index < len(osc_nodes):
            node = osc_nodes[index]
            if frequency is not None:
                node.params["frequency"] = frequency
            if amplitude is not None:
                node.params["amplitude"] = amplitude

        logger.debug(
            f"Modified oscillator {index}: frequency={frequency}, amplitude={amplitude}"
        )
        return self

    def modify_oscillators_with_offsets(
        self,
        frequency_offsets: list[float] | None = None,
        amplitude_multipliers: list[float] | None = None,
    ) -> PresetBuilder:
        """Modify multiple oscillators with relative offsets.

        Args:
            frequency_offsets: List of semitone offsets to apply to each oscillator.
                              E.g., [0, 5, 7] creates a major chord (root, 5th, 7th).
                              Uses 12-TET: frequency * 2^(offset/12)
            amplitude_multipliers: List of amplitude multipliers for each oscillator.
                                  E.g., [1.0, 0.8, 0.6] for descending volumes

        Returns:
            Self for method chaining

        Example:
            >>> # Create a major chord (C, E, G) at 440 Hz base
            >>> preset = (PresetBuilder()
            ...     .sine(440)
            ...     .sine(440)
            ...     .sine(440))
            >>> preset.modify_oscillators_with_offsets([0, 4, 7])  # Major chord
        """
        if self._source is None:
            return self

        # Normalize to list
        sources = self._source if isinstance(self._source, list) else [self._source]

        # Apply frequency offsets (semitones)
        if frequency_offsets:
            for i, offset in enumerate(frequency_offsets):
                if i >= len(sources):
                    break
                osc = sources[i]
                if hasattr(osc, "frequency"):
                    # Convert semitones to frequency ratio: 2^(offset/12)
                    base_freq = osc.frequency
                    new_freq = base_freq * (2 ** (offset / 12))
                    osc.frequency = new_freq

                    # Update tree
                    osc_nodes = [
                        node
                        for node in self._component_tree.children
                        if self._get_node_category(node.component_type)
                        == ComponentCategory.OSCILLATOR
                    ]
                    if i < len(osc_nodes):
                        osc_nodes[i].params["frequency"] = new_freq

        # Apply amplitude multipliers
        if amplitude_multipliers:
            for i, multiplier in enumerate(amplitude_multipliers):
                if i >= len(sources):
                    break
                osc = sources[i]
                if hasattr(osc, "amplitude"):
                    base_amp = osc.amplitude
                    new_amp = base_amp * multiplier
                    osc.amplitude = new_amp

                    # Update tree
                    osc_nodes = [
                        node
                        for node in self._component_tree.children
                        if self._get_node_category(node.component_type)
                        == ComponentCategory.OSCILLATOR
                    ]
                    if i < len(osc_nodes):
                        osc_nodes[i].params["amplitude"] = new_amp

        logger.debug(
            f"Applied offsets - frequency: {frequency_offsets}, amplitude: "
            f"{amplitude_multipliers}"
        )
        return self

    def get_oscillator_count(self) -> int:
        """Get the number of oscillators in the preset.

        Returns:
            Number of oscillators
        """
        if self._source is None:
            return 0
        if isinstance(self._source, list):
            return len(self._source)
        return 1

    # ========================================================================
    # Tree Visualization
    # ========================================================================

    def to_tree(self) -> str:
        """Get tree representation of the preset structure.

        Returns:
            String representation of the preset as a tree
        """
        lines = self._component_tree.to_tree_string()
        return "\n".join(lines)

    def print_tree(self) -> None:
        """Print the preset tree to console."""
        print(self.to_tree())

    def _get_node_category(self, component_type: str) -> ComponentCategory:
        """Resolve a component category for a tree node via the registry."""
        component_class = audio_registry.get(component_type, strict=True)
        assert component_class is not None
        return component_class.descriptor.category

    def _component_method_name(self, component_type: str) -> str:
        component_class = audio_registry.get(component_type, strict=True)
        assert component_class is not None
        descriptor = component_class.descriptor
        return descriptor.fluent_api_name or component_type

    def _component_params_from_config(
        self, component_type: str, component_config: dict[str, Any]
    ) -> dict[str, Any]:
        component_class = audio_registry.get(component_type, strict=True)
        assert component_class is not None
        parameter_names = component_class.descriptor.parameter_names
        return {
            key: value
            for key, value in component_config.items()
            if key not in {"name", "category", "description", "id"}
            and key in parameter_names
        }

    def _call_component_method(
        self, component_type: str, params: dict[str, Any], **extra_kwargs: Any
    ) -> None:
        method = self._component_methods.get(
            self._component_method_name(component_type)
        )
        if method:
            method(**params, **extra_kwargs)
        else:
            logger.warning(
                f"Method '{self._component_method_name(component_type)}' not found "
                f"for component '{component_type}'"
            )

    def _build_graph_config(self) -> dict[str, Any]:
        """Build a portable graph view of the current builder topology."""
        nodes = []
        source_ids = []
        modifier_ids = []
        modulation_edges = []

        for index, node in enumerate(self._component_tree.children, start=1):
            node_id = f"n{index}"
            component_config = {**node.params, "id": node_id}
            nodes.append(component_config)

            category = self._get_node_category(node.component_type)
            if category == ComponentCategory.OSCILLATOR:
                source_ids.append(node_id)
            elif category == ComponentCategory.MODIFIER:
                modifier_ids.append(node_id)
            elif category == ComponentCategory.MODULATOR:
                target = node.params.get("target", "amplitude")
                if source_ids:
                    modulation_edges.append(
                        {
                            "from": node_id,
                            "to": source_ids[-1],
                            "type": "modulation",
                            "target": target,
                            "operation": _MODULATION_OPERATIONS.get(target, "custom"),
                        }
                    )

        audio_edges = []
        output_source: str | dict[str, Any] | None
        if len(source_ids) > 1:
            output_source = {
                "kind": "mixer",
                "id": "mix1",
                "sources": source_ids,
                "mix_mode": "average",
            }
            for source_id in source_ids:
                audio_edges.append({"from": source_id, "to": "mix1", "type": "audio"})
            previous = "mix1"
        elif source_ids:
            output_source = source_ids[0]
            previous = source_ids[0]
        else:
            output_source = None
            previous = None

        for modifier_id in modifier_ids:
            if previous is not None:
                audio_edges.append(
                    {"from": previous, "to": modifier_id, "type": "audio"}
                )
            previous = modifier_id

        output_kind = "chain" if modifier_ids else "source"
        return {
            "schema_version": 1,
            "nodes": nodes,
            "edges": audio_edges + modulation_edges,
            "output": {
                "kind": output_kind,
                "source": output_source,
                "modifiers": modifier_ids,
            },
        }

    # ========================================================================
    # Config Generation (On-Demand)
    # ========================================================================

    def get_config(self) -> dict[str, Any]:
        """Generate configuration dictionary from current preset state.

        Config is generated on-demand instead of being stored as duplicate state.

        Returns:
            Configuration dictionary
        """
        components = []

        # Traverse tree to build config
        for node in self._component_tree.children:
            comp_config = {**node.params}
            components.append(comp_config)

        return {
            "version": "1.0",
            "name": self._name,
            "description": self._description,
            "sample_rate": self._sample_rate,
            "components": components,
            "graph": self._build_graph_config(),
        }

    # ========================================================================
    # Build Methods
    # ========================================================================

    def build(self) -> Any:
        """Build the final preset from the configuration.

        Returns:
            Built audio preset (Chain or source)

        Raises:
            ValueError: If no source oscillator added
        """
        if self._source is None:
            raise ValueError("No source oscillator added to preset.")

        # Handle multiple oscillators - use WaveAdder
        if isinstance(self._source, list):
            wave_adder_class = audio_registry.get("WaveAdder")
            if wave_adder_class:
                source = wave_adder_class(*self._source)
                logger.debug(f"Built WaveAdder with {len(self._source)} oscillators")
            else:
                logger.warning("WaveAdder not registered, using first oscillator only")
                source = self._source[0]
        else:
            source = self._source

        # Apply modifiers in chain
        if self._modifiers:
            chain_class = audio_registry.get("Chain")
            if chain_class:
                result = chain_class(source, *self._modifiers)
                logger.debug(f"Built Chain with {len(self._modifiers)} modifiers")
            else:
                logger.warning(
                    "Chain not registered, returning source without modifiers"
                )
                result = source
        else:
            result = source

        logger.debug(f"Preset '{self._name}' built successfully")
        return result

    # ========================================================================
    # Inspection Methods
    # ========================================================================

    def describe(self) -> str:
        """Get a human-readable description of the preset.

        Returns:
            Multi-line string describing the preset
        """
        lines = [f"Preset name: {self._name}"]

        if self._description:
            lines.append(f"Description: {self._description}")

        lines.append("")

        for node in self._component_tree.children:
            component = audio_registry.get(node.component_type, strict=True)
            assert component is not None

            # Build parameter string
            params = []
            for key, value in node.params.items():
                if key not in ["sample_rate"]:
                    params.append(f"{key}={value}")

            param_str = ", ".join(params)

            descriptor = component.descriptor
            desc = descriptor.description if descriptor else node.component_type
            lines.append(f"- {desc} ({param_str})")

        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        """Get a summary of preset characteristics.

        Returns:
            Dictionary with preset statistics
        """
        oscillators = 0
        modulators = 0
        effects = 0

        for node in self._component_tree.children:
            component_class = audio_registry.get(node.component_type, strict=True)
            assert component_class is not None
            category = component_class.descriptor.category
            if category == ComponentCategory.OSCILLATOR:
                oscillators += 1
            elif category == ComponentCategory.MODULATOR:
                modulators += 1
            elif category == ComponentCategory.MODIFIER:
                effects += 1

        return {
            "name": self._name,
            "description": self._description,
            "oscillators": oscillators,
            "modulators": modulators,
            "effects": effects,
            "components": len(self._component_tree.children),
            "sample_rate": self._sample_rate,
        }

    # ========================================================================
    # Modification Methods
    # ========================================================================

    def clear_effects(self) -> PresetBuilder:
        """Remove all effects (modifiers).

        Returns:
            Self for method chaining
        """
        self._modifiers.clear()

        # Remove modifier nodes from tree
        self._component_tree.children = [
            node
            for node in self._component_tree.children
            if self._get_node_category(node.component_type)
            != ComponentCategory.MODIFIER
        ]

        logger.debug("Cleared all effects")
        return self

    def clone(self) -> PresetBuilder:
        """Create a copy of this preset builder.

        Returns:
            New PresetBuilder instance with same configuration
        """
        return self.from_config(self.get_config())

    # ========================================================================
    # Preset Methods
    # ========================================================================

    def save_preset(self, filepath: str | Path) -> None:
        """Save the current preset configuration as a preset.

        Args:
            filepath: Path to save the preset file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        config = self.get_config()
        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved preset to {filepath}")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> PresetBuilder:
        """Create a builder from a preset configuration dictionary."""
        name = config.get("name", "Untitled Preset")
        description = config.get("description", "")
        builder = cls(name=name, description=description)
        if "sample_rate" in config:
            builder.set_sample_rate(config["sample_rate"])

        graph = config.get("graph")
        if isinstance(graph, dict):
            builder._load_graph_config(graph)
        else:
            builder._load_legacy_components(config.get("components", []))

        return builder

    def _load_graph_config(self, graph: dict[str, Any]) -> None:
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        nodes_by_id = {node["id"]: node for node in nodes if "id" in node}
        loaded_nodes: set[str] = set()
        modulation_edges_by_target: dict[str, list[dict[str, Any]]] = {}
        for edge in edges:
            if edge.get("type") == "modulation":
                modulation_edges_by_target.setdefault(edge.get("to"), []).append(edge)

        output = graph.get("output", {})
        source_ref = output.get("source")
        if isinstance(source_ref, dict):
            source_ids = list(source_ref.get("sources", []))
        elif isinstance(source_ref, str):
            source_ids = [source_ref]
        else:
            source_ids = []

        for source_id in source_ids:
            self._load_graph_node(source_id, nodes_by_id, loaded_nodes)
            for edge in modulation_edges_by_target.get(source_id, []):
                self._load_graph_node(
                    edge.get("from"),
                    nodes_by_id,
                    loaded_nodes,
                    target=edge.get("target", "amplitude"),
                )

        for modifier_id in output.get("modifiers", []):
            self._load_graph_node(modifier_id, nodes_by_id, loaded_nodes)

        # Load disconnected nodes last so partially edited preset files remain usable.
        for component in nodes:
            node_id = component.get("id")
            if node_id not in loaded_nodes:
                self._load_graph_node(node_id, nodes_by_id, loaded_nodes)

    def _load_graph_node(
        self,
        node_id: str | None,
        nodes_by_id: dict[str, dict[str, Any]],
        loaded_nodes: set[str],
        *,
        target: str | None = None,
    ) -> None:
        if node_id is None or node_id in loaded_nodes or node_id not in nodes_by_id:
            return

        component = nodes_by_id[node_id]
        comp_type = component["name"]
        params = self._component_params_from_config(comp_type, component)
        category = self._get_node_category(comp_type)

        if category == ComponentCategory.MODULATOR:
            target = target or component.get("target", "amplitude")
            self._call_component_method(comp_type, params, target=target)
        else:
            self._call_component_method(comp_type, params)
        loaded_nodes.add(node_id)

    def _load_legacy_components(self, components: list[dict[str, Any]]) -> None:
        for component in components:
            comp_type = component["name"]
            params = self._component_params_from_config(comp_type, component)
            extra_kwargs = {}
            if self._get_node_category(comp_type) == ComponentCategory.MODULATOR:
                extra_kwargs["target"] = component.get("target", "amplitude")
            self._call_component_method(comp_type, params, **extra_kwargs)

    @classmethod
    def from_preset(cls, filepath: str | Path) -> PresetBuilder:
        """Load a preset configuration from a preset file.

        Args:
            filepath: Path to the preset file

        Returns:
            PresetBuilder instance loaded from preset
        """
        filepath = Path(filepath)

        with open(filepath) as f:
            config = json.load(f)

        builder = cls.from_config(config)
        logger.info(f"Loaded preset from {filepath}")
        return builder
