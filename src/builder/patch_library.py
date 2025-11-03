"""Preset library management for audio synthesis patches.

This module provides the PatchLibrary class for organizing, saving, and loading
audio synthesis builder in a structured manner with category support.
"""

from __future__ import annotations
from typing import Any, TYPE_CHECKING
from pathlib import Path
import json

from src.utils.logging_config import get_logger

if TYPE_CHECKING:
    from src.builder.patch_builder import PatchBuilder

logger = get_logger("builder.patch_library")


class PatchLibrary:
    """Manage a library of audio synthesis patches.

    This class provides utilities for managing collections of builder,
    including listing, loading, saving, and organizing builder with
    category support.

    Attributes:
        preset_dir: Directory containing preset files

    Example:
        >>> from src.builder import PatchLibrary, PatchBuilder
        >>>
        >>> #
        >>> library = PatchLibrary("builder/")
        >>>
        >>> # List available builder
        >>> builder = library.list_presets()
        >>>
        >>> # Load a preset via name
        >>> patch = library.load("bass_synth").build()
        >>>
        >>> # Save current patch
        >>> sine_osc = PatchBuilder().sine(440).adsr(0.1, 0.2, 0.7, 0.3)
        >>> library.save(sine_osc, "my_patch", category="leads")
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
            >>> library = PatchLibrary()
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

    def load(self, name: str) -> PatchBuilder:
        """Load a preset by name.

        Args:
            name: Preset name (with or without .json extension)

        Returns:
            PatchBuilder configured from preset

        Example:
            >>> library = PatchLibrary()
            >>> patch = library.load("bass_synth").build()
        """
        from src.builder.patch_builder import PatchBuilder

        filepath = self.preset_dir / name
        if not filepath.suffix:
            filepath = filepath.with_suffix(".json")

        return PatchBuilder.from_preset(filepath)

    def save(
        self,
        builder: PatchBuilder,
        name: str | None = None,
        category: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save a patch as a preset.

        Args:
            builder: PatchBuilder instance to save
            name: Preset filename (if None, uses patch name)
            category: Optional category/subdirectory
            metadata: Optional metadata (author, tags, etc.)
                     Note: patch description is automatically included

        Example:
            >>> from src.builder import PatchBuilder, PatchLibrary
            >>>library = PatchLibrary("builder/")
            >>> builder = (PatchBuilder("Warm Lead")
            ...     .set_description("Smooth lead sound")
            ...     .sine(440)
            ...     .adsr(0.1, 0.2, 0.7, 0.3))
            >>> library.save(builder, category="leads")  # Uses patch name
            >>> # Or with custom filename:
            >>> library.save(builder, "my_lead", category="leads")
        """
        # Use patch name if no filename provided
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
            >>> library = PatchLibrary()
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
            >>> library = PatchLibrary()
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
