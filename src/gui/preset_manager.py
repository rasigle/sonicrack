"""Preset management system for saving and loading patches.

This module provides functionality to serialize patches to JSON format
and restore them, including all modules, connections, and parameters.
"""

import json
import logging
from typing import Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class PresetManager:
    """Manages saving and loading of synth patches as presets.

    Presets are stored as JSON files containing:
    - Module types and positions
    - Module parameters
    - Cable connections
    - Metadata (name, author, tags, etc.)
    """

    def __init__(self, preset_directory: Path | None = None):
        """Initialize the preset manager.

        Args:
            preset_directory: Directory to store presets. If None, uses default.
        """
        if preset_directory is None:
            preset_directory = Path.home() / ".audioplayground" / "presets"

        self.preset_directory = Path(preset_directory)
        self.preset_directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"Preset directory: {self.preset_directory}")

    def save_preset(
        self,
        modules: list[Any],
        connections: list[tuple[Any, Any]],
        name: str,
        author: str = "",
        description: str = "",
        tags: list[str] = None,
        category: str = "User",
    ) -> Path | None:
        """Save a patch as a preset.

        Args:
            modules: List of module widgets
            connections: List of (output_port, input_port) tuples
            name: Preset name
            author: Author name
            description: Preset description
            tags: List of tags for categorization
            category: Category (e.g., "Bass", "Lead", "Pad", "User")

        Returns:
            Path to saved preset file, or None if save failed
        """
        try:
            # Build preset data structure
            preset_data = {
                "metadata": {
                    "name": name,
                    "author": author,
                    "description": description,
                    "tags": tags or [],
                    "category": category,
                    "created": datetime.now().isoformat(),
                    "version": "1.0",
                },
                "modules": [],
                "connections": [],
            }

            # Create module ID mapping
            module_ids = {id(module): idx for idx, module in enumerate(modules)}

            # Serialize modules
            for module_id, module in zip(module_ids.values(), modules):
                module_data = {
                    "id": module_id,
                    "type": module.module_title,
                    "custom_name": (
                        module.custom_name if hasattr(module, "custom_name") else ""
                    ),
                    "component_category": module.module_category,
                    "position": {"x": module.pos().x(), "y": module.pos().y()},
                    "parameters": module.get_parameters(),
                }
                preset_data["modules"].append(module_data)

            # Serialize connections
            for start_port, end_port in connections:
                connection_data = {
                    "from_module": module_ids[id(start_port.parent_module)],
                    "from_port": start_port.index,
                    "to_module": module_ids[id(end_port.parent_module)],
                    "to_port": end_port.index,
                }
                preset_data["connections"].append(connection_data)

            # Save to file
            filename = self._sanitize_filename(name) + ".json"
            filepath = self.preset_directory / filename

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(preset_data, f, indent=2)

            logger.info(f"Preset saved: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to save preset: {e}", exc_info=True)
            return None

    def load_preset(self, filepath: Path) -> dict[str, Any] | None:
        """Load a preset from file.

        Args:
            filepath: Path to preset file

        Returns:
            Preset data dictionary, or None if load failed
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                preset_data = json.load(f)

            logger.info(f"Preset loaded: {filepath}")
            return preset_data

        except Exception as e:
            logger.error(f"Failed to load preset: {e}", exc_info=True)
            return None

    def list_presets(self, category: str | None = None) -> list[dict[str, Any]]:
        """List available presets.

        Args:
            category: Optional category filter

        Returns:
            List of preset metadata dictionaries
        """
        presets = []

        try:
            for filepath in self.preset_directory.glob("*.json"):
                preset_data = self.load_preset(filepath)
                if preset_data:
                    metadata = preset_data.get("metadata", {})

                    # Filter by category if specified
                    if category and metadata.get("category") != category:
                        continue

                    metadata["filepath"] = str(filepath)
                    presets.append(metadata)

            # Sort by name
            presets.sort(key=lambda p: p.get("name", ""))

        except Exception as e:
            logger.error(f"Failed to list presets: {e}", exc_info=True)

        return presets

    def delete_preset(self, filepath: Path) -> bool:
        """Delete a preset file.

        Args:
            filepath: Path to preset file

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            Path(filepath).unlink()
            logger.info(f"Preset deleted: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete preset: {e}", exc_info=True)
            return False

    def get_categories(self) -> list[str]:
        """Get list of all preset categories.

        Returns:
            List of unique category names
        """
        categories = set()

        for preset in self.list_presets():
            category = preset.get("category", "User")
            categories.add(category)

        return sorted(categories)

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Sanitize a preset name for use as filename.

        Args:
            name: Preset name

        Returns:
            Sanitized filename (without extension)
        """
        # Replace invalid characters with underscore
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            name = name.replace(char, "_")

        # Remove leading/trailing whitespace and dots
        name = name.strip(". ")

        # Limit length
        if len(name) > 100:
            name = name[:100]

        return name if name else "unnamed_preset"

    def export_preset(self, filepath: Path, export_path: Path) -> bool:
        """Export a preset to a different location.

        Args:
            filepath: Source preset file
            export_path: Destination path

        Returns:
            True if exported successfully, False otherwise
        """
        try:
            import shutil

            shutil.copy2(filepath, export_path)
            logger.info(f"Preset exported: {export_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export preset: {e}", exc_info=True)
            return False

    def import_preset(self, import_path: Path) -> Path | None:
        """Import a preset from an external location.

        Args:
            import_path: Path to preset file to import

        Returns:
            Path to imported preset in preset directory, or None if failed
        """
        try:
            import shutil

            destination = self.preset_directory / import_path.name
            shutil.copy2(import_path, destination)
            logger.info(f"Preset imported: {destination}")
            return destination
        except Exception as e:
            logger.error(f"Failed to import preset: {e}", exc_info=True)
            return None
