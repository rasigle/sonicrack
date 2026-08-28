"""Preset management system for saving and loading patches.

This module provides functionality to serialize patches to JSON format
and restore them, including all modules, connections, and parameters.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from sonicrack.constants import DEFAULT_PRESET_DIRECTORY, PRESET_FILE_EXTENSION

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
            preset_directory = DEFAULT_PRESET_DIRECTORY

        self.preset_directory = Path(preset_directory)
        self.preset_directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"Preset directory: {self.preset_directory}")

    def save_preset(
        self,
        modules: list[Any],
        connections: list[tuple[Any, Any]],
        metadata: dict[str, Any],
        file_path: str | Path | None = None,
        save_as_library_preset: bool = False,
    ) -> Path | None:
        """Actually save a patch as a file.

        Args:
            modules: List of module widgets
            connections: List of (output_port, input_port) tuples
            metadata: Preset metadata dictionary, including:
                # name: Preset name
                # author: Author name
                # description: Preset description
                # tags: List of tags for categorization
                # category: Category (e.g., "Bass", "Lead", "Pad", "User")
            file_path: Optional path to save preset file. If None, uses preset name.
            save_as_library_preset: If True, saves to library presets directory

        Returns:
            Path to saved preset file, or None if save failed
        """
        # Determine the actual file path
        if save_as_library_preset:
            # Saving to library - use preset name
            preset_name = metadata.get("name", "unnamed_preset")
            sanitized_name = _sanitize_filename(preset_name)
            file_path = self.preset_directory / sanitized_name
        else:
            # Saving to specific path (e.g., Save Patch As)
            if not file_path:
                logger.error("No file path provided for save_preset")
                return None
            file_path = Path(file_path)
            # Ensure extension
            if not str(file_path).endswith(PRESET_FILE_EXTENSION):
                file_path = Path(str(file_path) + PRESET_FILE_EXTENSION)

        # Serialize the patch
        preset_data = self._serialize_patch(modules, connections, metadata)

        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Save to file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(preset_data, f, indent=2)

            logger.info(f"Preset saved successfully to: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Failed to save preset to {file_path}: {e}", exc_info=True)
            return None

    @staticmethod
    def load_preset(filepath: Path) -> dict[str, Any] | None:
        """Load a preset from file.

        Args:
            filepath: Path to preset file

        Returns:
            Preset data dictionary, or None if load failed
        """
        try:
            with open(filepath, encoding="utf-8") as f:
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
        presets: list[dict[str, Any]] = []

        try:
            seen: set[Path] = set()
            candidates: list[Path] = list(
                self.preset_directory.glob(f"*{PRESET_FILE_EXTENSION}")
            )
            if PRESET_FILE_EXTENSION.lower() != ".json":
                candidates.extend(self.preset_directory.glob("*.json"))
            for filepath in candidates:
                resolved = filepath.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
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

    @staticmethod
    def delete_preset(filepath: Path) -> bool:
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
        categories: set[str] = set()

        for preset in self.list_presets():
            category = preset.get("category", "User")
            categories.add(category)

        return sorted(categories)

    @staticmethod
    def _serialize_patch(
        modules: list[Any], connections: list[tuple[Any, Any]], metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Serialize a patch to a dictionary structure.

        This is a helper method used by both save_preset and direct patch saving.
        It serializes modules and connections without metadata.

        Args:
            modules: List of module widgets
            connections: List of (output_port, input_port) tuples

        Returns:
            Dictionary containing serialized patch data
        """
        # Build preset data structure
        preset_data: dict[str, Any] = {
            "metadata": {
                "name": metadata.get("name", "Untitled Preset"),
                "author": metadata.get("author", ""),
                "description": metadata.get("description", ""),
                "tags": metadata.get("tags", []),
                "category": metadata.get("category", "User"),
                "created": datetime.now().isoformat(),
                "version": "1.0",
            },
            "modules": [],
            "connections": [],
        }

        # Create module ID mapping
        module_ids = {id(module): idx for idx, module in enumerate(modules)}

        # Serialize modules
        for module_id, module in zip(module_ids.values(), modules, strict=False):
            module_metadata = module.metadata
            module_data = {
                "id": module_id,
                "type": module_metadata.title,
                "custom_name": module.custom_name,
                "module_category": module_metadata.category.value,
                "position": {"x": module.pos().x(), "y": module.pos().y()},
                "parameters": module.get_parameters(),
            }
            preset_data["modules"].append(module_data)

        # Serialize connections
        for start_port, end_port in connections:
            connection_data = {
                "source_module": module_ids[id(start_port.parent_module)],
                "source_port": start_port.port_name,
                "target_module": module_ids[id(end_port.parent_module)],
                "target_port": end_port.port_name,
            }
            preset_data["connections"].append(connection_data)

        return preset_data

    @staticmethod
    def export_preset(filepath: Path, export_path: Path) -> bool:
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


def _sanitize_filename(name: str) -> str:
    """Sanitize a preset name for use as filename.

    Args:
        name: Preset name (not a full path!)

    Returns:
        Sanitized filename with extension
    """
    if not name:
        name = "unnamed_preset"

    # Replace invalid filename characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, "_")

    # Remove leading/trailing whitespace and dots
    name = name.strip(". ")

    # Ensure we have a valid name
    if not name:
        name = "unnamed_preset"

    # Add extension if not present
    if not name.lower().endswith(PRESET_FILE_EXTENSION):
        name += PRESET_FILE_EXTENSION

    return name
