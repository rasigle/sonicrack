"""High-level audio engine preset building and management.

This package provides API builders and preset management for creating
and organizing audio synthesis presets. These are application-level utilities
that orchestrate the core engine components.

Components:
    PresetBuilder: Fluent API for creating synthesis presets
    PresetLibrary: Manage collections of saved builder

Example:
    >>> from src.engine import PresetBuilder, PresetLibrary
    >>>
    >>> # Create preset with fluent API
    >>> preset = (PresetBuilder()
    ...     .sine(440)
    ...     .adsr(0.1, 0.2, 0.7, 0.3)
    ...     .volume(0.5)
    ...     .build())
    >>>
    >>> # Save and load builder
    >>> library = PresetLibrary("my_presets/")
    >>> library.save(PresetBuilder().sine(440), "my_sound", category="leads")
    >>> loaded = library.load("leads/my_sound").build()
"""

from src.engine.presets.preset_builder import PresetBuilder
from src.engine.presets.preset_library import PresetLibrary

__all__ = [
    "PresetBuilder",
    "PresetLibrary",
]
