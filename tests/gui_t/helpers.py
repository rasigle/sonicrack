"""Shared helpers for GUI runtime tests."""

from __future__ import annotations

import numpy as np

from sonicrack.patching.port import Port


def connect_signal(input_port: Port, values: np.ndarray) -> Port:
    """Write ``values`` on a throwaway output and connect it to ``input_port``."""
    output_port = Port("output", "Test Out")
    output_port.write(values)
    output_port.connect(input_port)
    return output_port
