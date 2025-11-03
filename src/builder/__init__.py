"""High-level patch building and preset management.

This package provides fluent API builders and preset management for creating
and organizing audio synthesis patches. These are application-level utilities
that orchestrate the core engine components.

Components:
    PatchBuilder: Fluent API for creating synthesis patches
    PresetLibrary: Manage collections of saved builder

Example:
    >>> from src.builder import PatchBuilder, PatchLibrary
    >>>
    >>> # Create patch with fluent API
    >>> patch = (PatchBuilder()
    ...     .sine(440)
    ...     .adsr(0.1, 0.2, 0.7, 0.3)
    ...     .volume(0.5)
    ...     .build())
    >>>
    >>> # Save and load builder
    >>> library = PatchLibrary("my_presets/")
    >>> library.save(PatchBuilder().sine(440), "my_sound", category="leads")
    >>> loaded = library.load("leads/my_sound").build()
"""

from src.builder.patch_library import PatchLibrary
from src.builder.patch_builder import PatchBuilder
from src.engine.engine_component_registry import (
    registry,
    ComponentCategory,
    ComponentDescriptor,
    register_component,
)

__all__ = [
    "PatchBuilder",
    "PatchLibrary",
    "registry",
    "ComponentDescriptor",
    "ComponentCategory",
    "register_component",
]
