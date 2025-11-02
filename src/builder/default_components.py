"""Default component registrations for the audio synthesis engine.

This module registers all built-in engine components with the component registry,
making them available to the PatchBuilder system. New components can be added
here without modifying the PatchBuilder class.

The registration system separates component definitions from the builder logic,
enabling better maintainability and extensibility.
"""

from src.builder.component_registry import (
    registry,
    register_component,
    ComponentCategory,
)
from src.engine.oscillator import (
    SineOscillator,
    SquareOscillator,
    SawtoothOscillator,
    TriangleOscillator,
)
from src.engine.modulator import ADSREnvelope
from src.engine.modifier import Volume, Panner, Clipper
from src.utils.logging_config import get_logger

logger = get_logger("builder.default_components")


def register_oscillators():
    """Register all oscillator components."""

    register_component(
        name="sine_oscillator",
        category=ComponentCategory.OSCILLATOR,
        factory=SineOscillator,
        config_params=["frequency", "amplitude", "phase", "sample_rate"],
        description="Pure sine wave oscillator",
    )

    register_component(
        name="square_oscillator",
        category=ComponentCategory.OSCILLATOR,
        factory=SquareOscillator,
        config_params=["frequency", "amplitude", "phase", "sample_rate"],
        description="Square wave oscillator",
    )

    register_component(
        name="triangle_oscillator",
        category=ComponentCategory.OSCILLATOR,
        factory=TriangleOscillator,
        config_params=["frequency", "amplitude", "phase", "sample_rate"],
        description="Triangle wave oscillator",
    )

    register_component(
        name="sawtooth_oscillator",
        category=ComponentCategory.OSCILLATOR,
        factory=SawtoothOscillator,
        config_params=["frequency", "amplitude", "phase", "sample_rate"],
        description="Sawtooth wave oscillator",
    )


def register_modulators():
    """Register all modulator components."""

    register_component(
        name="adsr_envelope",
        category=ComponentCategory.MODULATOR,
        factory=ADSREnvelope,
        config_params=[
            "attack_duration",
            "decay_duration",
            "sustain_level",
            "release_duration",
            "sample_rate",
            "target",
        ],
        description="ADSR envelope generator",
    )


def register_modifiers():
    """Register all modifier components."""

    register_component(
        name="volume",
        category=ComponentCategory.MODIFIER,
        factory=Volume,
        config_params=["amplitude"],
        description="Volume/amplitude control",
    )

    register_component(
        name="panner",
        category=ComponentCategory.MODIFIER,
        factory=Panner,
        config_params=["position"],
        description="Stereo panning (-1.0 left, 0.0 center, 1.0 right)",
    )

    register_component(
        name="clipper",
        category=ComponentCategory.MODIFIER,
        factory=Clipper,
        config_params=["clip_range"],
        description="Audio clipping/saturation effect",
    )


def register_all_components():
    """Register all default components.

    This function should be called once at module initialization to populate
    the component registry with all built-in components.
    """
    logger.info("Registering default components...")

    register_oscillators()
    register_modulators()
    register_modifiers()

    component_count = len(registry.list_components())
    logger.info(f"Registered {component_count} default components")


# Auto-register on import
register_all_components()
