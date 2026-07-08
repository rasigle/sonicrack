"""Core GUI contracts, data models, and runtime helpers."""

from sonicrack.gui.core.module import AudioModule, ModuleCategory, ModuleMetadata
from sonicrack.gui.core.port import Port
from sonicrack.gui.core.preset_manager import PresetManager
from sonicrack.gui.core.runtime import (
    RuntimeModuleSpec,
    RuntimeParameters,
    get_runtime_spec,
    process_runtime_module,
)

__all__ = [
    "AudioModule",
    "ModuleCategory",
    "ModuleMetadata",
    "Port",
    "PresetManager",
    "RuntimeModuleSpec",
    "RuntimeParameters",
    "get_runtime_spec",
    "process_runtime_module",
]
