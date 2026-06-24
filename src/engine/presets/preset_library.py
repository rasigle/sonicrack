"""Preset library management for audio synthesis presets.

This module provides the PresetLibrary class for organizing, saving, and loading
audio synthesis builder in a structured manner with category support.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.engine.presets.preset_builder import PresetBuilder

logger = logging.getLogger(__name__)


class PresetLibrary:
    """Manage a library of audio synthesis presets.

    This class provides utilities for managing collections of builder,
    including listing, loading, saving, and organizing builder with
    category support.

    Attributes:
        preset_dir: Directory containing preset files

    Example:
        >>> from engine import PresetLibrary, PresetBuilder
        >>>
        >>> #
        >>> library = PresetLibrary("builder/")
        >>>
        >>> # List available builder
        >>> builder = library.list_presets()
        >>>
        >>> # Load a preset via name
        >>> preset = library.load("bass_synth").build()
        >>>
        >>> # Save current preset
        >>> sine_osc = PresetBuilder().sine(440).adsr(0.1, 0.2, 0.7, 0.3)
        >>> library.save(sine_osc, "my_preset", category="leads")
    """

    def __init__(self, preset_dir: str | Path = "builder"):
        """Initialize preset library.

        Args:
            preset_dir: Directory to store/load builder from
        """
        self.preset_dir = Path(preset_dir)
        self.preset_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized preset library at {self.preset_dir}")

    def list_presets(self, category: str | None = None) -> list[str]:
        """List available builder.

        Args:
            category: Optional category to filter by

        Returns:
            List of preset names (without .json extension)

        Example:
            >>> library = PresetLibrary()
            >>> all_presets = library.list_presets()
            >>> lead_presets = library.list_presets(category="leads")
        """
        if category:
            search_dir = self.preset_dir / category
            if not search_dir.exists():
                return []
        else:
            search_dir = self.preset_dir

        presets = []
        for preset_file in search_dir.rglob("*.json"):
            # Get relative path from preset_dir
            rel_path = preset_file.relative_to(self.preset_dir)
            # Remove .json extension and normalize path separators
            preset_name = str(rel_path.with_suffix("")).replace("\\", "/")
            presets.append(preset_name)

        return sorted(presets)

    def load(self, name: str) -> PresetBuilder:
        """Load a preset by name.

        Args:
            name: Preset name (with or without .json extension)

        Returns:
            PresetBuilder configured from preset

        Example:
            >>> library = PresetLibrary()
            >>> preset = library.load("bass_synth").build()
        """
        from src.engine.presets.preset_builder import PresetBuilder

        filepath = self.preset_dir / name
        if not filepath.suffix:
            filepath = filepath.with_suffix(".json")

        return PresetBuilder.from_preset(filepath)

    def save(
        self,
        builder: PresetBuilder,
        name: str | None = None,
        category: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save a preset as a preset.

        Args:
            builder: PresetBuilder instance to save
            name: Preset filename (if None, uses preset name)
            category: Optional category/subdirectory
            metadata: Optional metadata (author, tags, etc.)
                     Note: preset description is automatically included

        Example:
            >>> from engine import PresetBuilder, PresetLibrary
            >>>library = PresetLibrary("builder/")
            >>> builder = (PresetBuilder("Warm Lead")
            ...     .set_description("Smooth lead sound")
            ...     .sine(440)
            ...     .adsr(0.1, 0.2, 0.7, 0.3))
            >>> library.save(builder, category="leads")  # Uses preset name
            >>> # Or with custom filename:
            >>> library.save(builder, "my_lead", category="leads")
        """
        # Use preset name if no filename provided
        if name is None:
            name = builder.get_name()
            # Sanitize filename
            name = name.replace(" ", "_").replace("/", "_").replace("\\", "_")

        if category:
            filepath = self.preset_dir / category / f"{name}.json"
        else:
            filepath = self.preset_dir / f"{name}.json"

        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Get config (already includes name and description)
        config = builder.get_config()

        # Add additional metadata if provided
        if metadata:
            if "metadata" not in config:
                config["metadata"] = {}
            config["metadata"].update(metadata)

        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved preset '{name}' to {filepath}")

    def delete(self, name: str) -> None:
        """Delete a preset.

        Args:
            name: Preset name to delete

        Example:
            >>> library = PresetLibrary()
            >>> library.delete("old_preset")
        """
        filepath = self.preset_dir / name
        if not filepath.suffix:
            filepath = filepath.with_suffix(".json")

        if filepath.exists():
            filepath.unlink()
            logger.info(f"Deleted preset '{name}'")
        else:
            logger.warning(f"Preset '{name}' not found")

    def get_categories(self) -> list[str]:
        """Get list of preset categories.

        Returns:
            List of category names

        Example:
            >>> library = PresetLibrary()
            >>> lib_categories = library.get_categories()
            >>> print(lib_categories)
            ['bass', 'leads', 'pads', 'fx']
        """
        categories = set()
        for preset_file in self.preset_dir.rglob("*.json"):
            rel_path = preset_file.relative_to(self.preset_dir)
            if len(rel_path.parts) > 1:
                categories.add(rel_path.parts[0])

        return sorted(categories)
