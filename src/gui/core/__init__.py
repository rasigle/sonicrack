"""Core GUI contracts, data models, and runtime helpers."""

from src.gui.core.module import AudioModule, ModuleCategory, ModuleMetadata
from src.gui.core.port import Port
from src.gui.core.preset_manager import PresetManager
from src.gui.core.runtime import (
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
