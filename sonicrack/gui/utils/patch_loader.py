"""Headless patch loader for benchmarking and testing.

This module provides functionality to load and render patches without
the full GUI infrastructure, enabling performance testing and validation.

Note: This module requires PyQt6 and will create a QApplication instance
automatically for headless rendering of GUI modules.
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _target_port_aliases(port_name: str | None) -> tuple[str, ...]:
    """Return current and legacy target input names for patch loading."""
    if not isinstance(port_name, str):
        return ()

    aliases = {
        "In": ("In", "Left/Mono", "L/Mono", "Input", "in"),
        "in": ("in", "In", "Left/Mono", "L/Mono", "Input"),
        "Input": ("Input", "In", "Left/Mono", "L/Mono", "in"),
        "L/Mono": ("L/Mono", "Left/Mono", "In"),
        "Left/Mono": ("Left/Mono", "L/Mono", "In"),
        "R": ("R", "Right", "In"),
        "Right": ("Right", "R", "In"),
        "Mod": ("Mod", "mod", "Modulation", "modulation"),
        "mod": ("mod", "Mod", "Modulation", "modulation"),
    }
    return aliases.get(port_name, (port_name,))


# Initialize Qt application for headless rendering
# This MUST happen before any GUI modules are imported
_qt_app = None


def _ensure_qt_application():
    """Ensure QApplication exists for GUI module instantiation."""
    global _qt_app

    if _qt_app is not None:
        return _qt_app

    try:
        import sys

        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QApplication

        # Check if application already exists
        existing_app = QApplication.instance()
        if existing_app is not None:
            _qt_app = existing_app
            logger.debug("Using existing QApplication")
            return _qt_app

        # Create new application for headless mode
        with contextlib.suppress(BaseException):
            QApplication.setAttribute(
                Qt.ApplicationAttribute.AA_ShareOpenGLContexts, False
            )

        _qt_app = QApplication(sys.argv)
        _qt_app.setQuitOnLastWindowClosed(False)
        logger.info("Created QApplication for headless rendering")

        return _qt_app

    except ImportError as e:
        logger.error(f"Failed to create QApplication: {e}")
        raise RuntimeError("PyQt6 is required for patch loading") from e


# Create Qt application immediately on import
try:
    _ensure_qt_application()
except Exception as e:
    logger.warning(f"Failed to initialize Qt application: {e}")
    logger.warning("Patch loading may not work properly")


def _get_module_registry():
    """Lazy load module registry to avoid Qt dependencies at import time."""
    try:
        # Need QApplication for QWidget-based modules
        import sys

        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QApplication

        # Create QApplication if not exists
        app = QApplication.instance()
        if app is None:
            # Set attributes before creating application
            with contextlib.suppress(BaseException):
                QApplication.setAttribute(
                    Qt.ApplicationAttribute.AA_ShareOpenGLContexts, False
                )

            # Create application for headless mode
            app = QApplication(sys.argv)
            app.setQuitOnLastWindowClosed(False)
            logger.info("Created QApplication for headless rendering")

        from sonicrack.gui.module_registry import (
            get_registry,
            initialize_module_registry,
        )

        # Get the registry and ensure it's initialized
        registry = get_registry()

        # If registry is empty, initialize it
        if registry.count() == 0:
            logger.info("Initializing module registry...")
            initialize_module_registry()
            logger.info(f"Initialized {registry.count()} modules")

        return registry
    except ImportError as e:
        logger.error(f"Failed to import module registry: {e}")
        logger.error("GUI modules may not be available in this environment")
        raise RuntimeError(
            "Module registry not available. This tool requires the GUI package."
        ) from e


class HeadlessPatchRenderer:
    """Renders patches without GUI for benchmarking.

    This class loads .apr patch files and renders audio by instantiating
    modules and using a proper AudioEngine for rendering.

    Usage:
        renderer = HeadlessPatchRenderer()
        renderer.load_patch("patch.apr")
        audio = renderer.render(num_samples=44100)
    """

    def __init__(self, sample_rate: int = 44100):
        """Initialize patch renderer.

        Args:
            sample_rate: Sample rate for audio rendering (default: 44100)
        """
        self.modules: dict[int, Any] = {}
        self.connections: list[dict[str, Any]] = []
        self.patch_data: dict[str, Any] | None = None
        self._output_module: Any = None
        self._registry: Any = None
        self._audio_engine: Any = None
        self._sample_rate = sample_rate

    def _ensure_registry(self):
        """Ensure module registry is loaded."""
        if self._registry is None:
            self._registry = _get_module_registry()

    def load_patch(self, filepath: str | Path) -> bool:
        """Load a patch file.

        Args:
            filepath: Path to .apr patch file

        Returns:
            True if successful, False otherwise
        """
        filepath = Path(filepath)

        try:
            # Ensure registry is loaded
            self._ensure_registry()

            with open(filepath, encoding="utf-8") as f:
                self.patch_data = json.load(f)

            # Clear previous state
            self.modules.clear()
            self.connections.clear()
            self._output_module = None

            # Create AudioEngine for proper rendering
            from sonicrack.gui.audio_engine import AudioEngine

            self._audio_engine = AudioEngine(sample_rate=self._sample_rate)

            # Create modules
            for module_data in self.patch_data.get("modules", []):
                if not self._create_module(module_data):
                    logger.warning(
                        f"Failed to create module: {module_data.get('type')}"
                    )

            # Store connections for later
            self.connections = self.patch_data.get("connections", [])

            # Establish connections
            self._connect_modules()

            logger.info(f"Patch loaded: {filepath.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load patch: {e}", exc_info=True)
            return False

    def _create_module(self, module_data: dict[str, Any]) -> bool:
        """Create a module instance.

        Args:
            module_data: Module configuration dict

        Returns:
            True if successful
        """
        try:
            module_type = module_data.get("type")
            module_id = module_data.get("id")
            parameters = module_data.get("parameters", {})

            # Get module class
            module_class = self._registry.get(module_type)
            if not module_class:
                logger.warning(f"Unknown module type: {module_type}")
                return False

            # Create instance
            module = module_class()

            # Set parameters
            if hasattr(module, "set_parameters"):
                try:
                    module.set_parameters(parameters)
                except Exception as e:
                    logger.warning(f"Failed to set parameters for {module_type}: {e}")

            # Store module
            self.modules[module_id] = module

            # Track output module
            if module_type == "Output":
                self._output_module = module

            return True

        except Exception as e:
            logger.error(f"Error creating module: {e}", exc_info=True)
            return False

    def _connect_modules(self) -> None:
        """Establish connections between modules."""
        for conn in self.connections:
            try:
                source_id = conn.get("source_module")
                source_port = conn.get("source_port")
                target_id = conn.get("target_module")
                target_port = conn.get("target_port")

                source_module = self.modules.get(source_id)
                target_module = self.modules.get(target_id)

                if not source_module or not target_module:
                    logger.warning(f"Missing module for connection: {conn}")
                    continue

                # Debug: Log available ports
                if hasattr(source_module, "outputs"):
                    logger.debug(
                        f"Source module outputs: {list(source_module.outputs.keys())}"
                    )
                if hasattr(target_module, "inputs"):
                    logger.debug(
                        f"Target module inputs: {list(target_module.inputs.keys())}"
                    )

                # Get ports with fallback for legacy names
                source = None
                target = None

                if hasattr(source_module, "outputs"):
                    source = source_module.outputs.get(source_port)

                    # If source port not found, try common aliases
                    if not source:
                        source_aliases = {
                            "Out": ["out", "output", "Output"],
                            "out": ["Out", "output", "Output"],
                        }

                        for alias in source_aliases.get(source_port, []):
                            source = source_module.outputs.get(alias)
                            if source:
                                logger.debug(
                                    f"Mapped source port {source_port} -> {alias}"
                                )
                                break

                if hasattr(target_module, "inputs"):
                    target = target_module.inputs.get(target_port)

                    # If target port not found, try common aliases
                    if not target:
                        for alias in _target_port_aliases(target_port):
                            if alias == target_port:
                                continue
                            target = target_module.inputs.get(alias)
                            if target:
                                logger.debug(
                                    f"Mapped target port {target_port} -> {alias}"
                                )
                                break

                if source and target:
                    target.connect(source)
                    logger.debug(f"Connected {source_port} -> {target_port}")
                else:
                    logger.warning(
                        f"Missing ports for connection: {source_port} -> {target_port}"
                    )

            except Exception as e:
                logger.error(f"Error connecting modules: {e}", exc_info=True)

    def render(
        self,
        num_samples: int = 512,
        sample_rate: int = 44100,
    ) -> np.ndarray:
        """Render audio from the loaded patch.

        Args:
            num_samples: Number of samples to render
            sample_rate: Sample rate in Hz

        Returns:
            Audio samples as numpy array

        Raises:
            RuntimeError: If no patch is loaded or rendering fails
        """
        if not self.modules:
            raise RuntimeError("No patch loaded")

        if self._output_module is None:
            raise RuntimeError("No output module in patch")

        if self._audio_engine is None:
            raise RuntimeError("Audio engine not initialized")

        try:
            # Use the audio engine to properly render the patch
            # Get the output module's input port(s)
            output_port = None

            # Try various port names
            if hasattr(self._output_module, "inp_port_l"):
                output_port = self._output_module.inp_port_l
            elif hasattr(self._output_module, "in_port"):
                output_port = self._output_module.in_port
            elif hasattr(self._output_module, "inputs"):
                # Try to find the first connected input port
                for port in self._output_module.inputs.values():
                    from sonicrack.gui.core.port import Port

                    if isinstance(port, Port) and port.is_connected:
                        output_port = port
                        break

            if output_port:
                from sonicrack.gui.core.port import Port

                if isinstance(output_port, Port):
                    # Render using the audio engine's proper pipeline
                    rendered_values = self._audio_engine.render_ports(
                        [output_port], num_samples
                    )

                    if rendered_values and len(rendered_values) > 0:
                        audio = rendered_values[0]

                        if isinstance(audio, np.ndarray):
                            # Ensure correct length
                            if len(audio) >= num_samples:
                                return audio[:num_samples].astype(np.float32)
                            else:
                                # Pad with zeros if needed
                                result = np.zeros(num_samples, dtype=np.float32)
                                result[: len(audio)] = audio
                                return result
                        elif isinstance(audio, (int, float)):
                            return np.full(num_samples, audio, dtype=np.float32)

            # Fallback: return silence
            logger.warning("No audio output available, returning silence")
            return np.zeros(num_samples, dtype=np.float32)

        except Exception as e:
            logger.error(f"Rendering failed: {e}", exc_info=True)
            raise RuntimeError(f"Rendering failed: {e}") from e

    def get_modules(self) -> list[Any]:
        """Get list of module instances.

        Returns:
            List of module instances
        """
        return list(self.modules.values())

    def get_module_count(self) -> int:
        """Get number of modules in patch.

        Returns:
            Module count
        """
        return len(self.modules)

    def get_connection_count(self) -> int:
        """Get number of connections in patch.

        Returns:
            Connection count
        """
        return len(self.connections)

    def describe(self) -> str:
        """Get patch description.

        Returns:
            Human-readable patch description
        """
        if not self.patch_data:
            return "No patch loaded"

        metadata = self.patch_data.get("metadata", {})
        name = metadata.get("name", "Untitled")
        modules = len(self.modules)
        connections = len(self.connections)

        lines = [
            f"Patch: {name}",
            f"Modules: {modules}",
            f"Connections: {connections}",
        ]

        # List modules
        lines.append("\nModules:")
        for module_id, module in self.modules.items():
            module_type = type(module).__name__
            lines.append(f"  [{module_id}] {module_type}")

        return "\n".join(lines)


def load_and_render_patch(
    filepath: str | Path,
    num_samples: int = 512,
    sample_rate: int = 44100,
) -> np.ndarray:
    """Convenience function to load and render a patch.

    Args:
        filepath: Path to patch file
        num_samples: Number of samples to render
        sample_rate: Sample rate in Hz

    Returns:
        Rendered audio samples

    Example:
        >>> audio = load_and_render_patch("example.apr", num_samples=44100)
        >>> print(f"Rendered {len(audio)} samples")
    """
    renderer = HeadlessPatchRenderer()
    if not renderer.load_patch(filepath):
        raise RuntimeError(f"Failed to load patch: {filepath}")

    return renderer.render(num_samples, sample_rate)


def validate_patch_file(filepath: str | Path) -> dict[str, Any]:
    """Validate a patch file and return information about it.

    Args:
        filepath: Path to patch file

    Returns:
        Dictionary with patch information:
            - valid: True if patch is valid
            - error: Error message if invalid
            - module_count: Number of modules
            - connection_count: Number of connections
            - metadata: Patch metadata

    Example:
        >>> info = validate_patch_file("example.apr")
        >>> if info["valid"]:
        ...     print(f"Valid patch with {info['module_count']} modules")
    """
    filepath = Path(filepath)
    result = {
        "valid": False,
        "error": None,
        "module_count": 0,
        "connection_count": 0,
        "metadata": {},
    }

    try:
        with open(filepath, encoding="utf-8") as f:
            patch_data = json.load(f)

        modules = patch_data.get("modules", [])
        connections = patch_data.get("connections", [])
        metadata = patch_data.get("metadata", {})

        result["valid"] = True
        result["module_count"] = len(modules)
        result["connection_count"] = len(connections)
        result["metadata"] = metadata

    except Exception as e:
        result["error"] = str(e)

    return result
