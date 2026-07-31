"""Small shared helpers for module-owned runtime processors.

Engine-domain buffer/CV helpers come from ``soniclab``; port/parameter helpers
that depend on SonicRack's graph model stay here.

Timing signals (Gate / Trigger / Clock)
---------------------------------------
``PortSignal.GATE`` and ``PortSignal.TRIGGER`` are distinct cable identities
but share the same voltage convention (``0`` / ``1`` with Schmitt edges).
Clock outputs are triggers: short high pulses on each step.

Edge-driven consumers (decay envelopes, S&H, sequencer clock inputs) only need
the rising edge. Sustain-driven consumers (ADSR, voices) need a usable high
duration. Single-sample triggers are too short for the latter — use
:func:`ensure_min_pulse_width` (or emit wider pulses at the source) so
Trigger→Gate patches produce audible notes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from soniclab.utils.buffers import as_samples, silence
from soniclab.utils.cv import apply_cv_influence, gate_transition_indices
from soniclab.utils.ramping import ramp_if_changed

if TYPE_CHECKING:
    from sonicrack.patching.port import Port

RuntimeParameters = Mapping[str, object]

# Shared immutable empty mapping for modules with no runtime parameters.
EMPTY_PARAMETERS: RuntimeParameters = MappingProxyType({})

# Eurorack-style minimum trigger high time. Single-sample "step markers" from
# StepClock are ~0.02 ms at 48 kHz — far below what sustain-style gate
# consumers need to leave idle. 2 ms is a common hardware trigger width.
DEFAULT_MIN_TRIGGER_SECONDS = 0.002
_GATE_LOW = 0.3
_GATE_HIGH = 0.7


def min_trigger_samples(sample_rate: float, seconds: float = DEFAULT_MIN_TRIGGER_SECONDS) -> int:
    """Return sample count for a minimum trigger/gate pulse width."""
    rate = max(1.0, float(sample_rate))
    return max(1, int(round(max(0.0, float(seconds)) * rate)))


def ensure_min_pulse_width(
    signal: np.ndarray,
    previous_level: float,
    width_samples: int,
    hold_remaining: int = 0,
    *,
    low: float = _GATE_LOW,
    high: float = _GATE_HIGH,
) -> tuple[np.ndarray, int]:
    """Ensure every high pulse lasts at least ``width_samples``.

    Sustained gates longer than the minimum pass through unchanged. Short
    trigger-like pulses (including single-sample clock ticks) are extended so
    Gate inputs can open ADSR/voice envelopes.

    Args:
        signal: Gate/trigger buffer in roughly ``[0, 1]``.
        previous_level: Last sample of the previous buffer (for rising edges).
        width_samples: Minimum high samples after each rising edge.
        hold_remaining: Forced-high samples left from the previous buffer.
        low / high: Schmitt thresholds matching :func:`gate_transition_indices`.

    Returns:
        ``(output, next_hold_remaining)``
    """
    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    width = max(1, int(width_samples))
    hold = max(0, int(hold_remaining))
    if values.size == 0:
        return values.copy(), hold

    out = np.empty_like(values)
    prev = float(previous_level)
    for index, value in enumerate(values):
        level = float(value)
        is_high = level > high
        if prev < low and is_high:
            hold = max(hold, width)

        if is_high or hold > 0:
            out[index] = 1.0
            if hold > 0:
                hold -= 1
        else:
            out[index] = 0.0
        prev = level

    return out, hold


def read_samples(port: Port, num_samples: int) -> np.ndarray:
    return as_samples(port.read(num_samples), num_samples)


def read_optional_port(port: Any | None, num_samples: int) -> np.ndarray | None:
    """Read a port when connected; otherwise return None.

    Safe for missing ports and for objects that expose ``is_connected`` either
    as a property on the port model or via a thin widget wrapper.
    """
    if port is None or not getattr(port, "is_connected", False):
        return None
    return read_samples(port, num_samples)


# Alias used by effect CV helpers / older call sites.
read_optional_cv = read_optional_port


def write_silence_if_disconnected(
    input_port: Any,
    output_port: Any,
    num_samples: int,
) -> bool:
    """Write silence and return True when the input port is not connected."""
    if getattr(input_port, "is_connected", False):
        return False
    output_port.write(silence(num_samples))
    return True


def parameter(
    parameters: RuntimeParameters,
    name: str,
    fallback_getter: Callable[[], object],
) -> object:
    if name in parameters:
        return parameters[name]
    return fallback_getter()


def float_parameter(
    parameters: RuntimeParameters,
    name: str,
    fallback_getter: Callable[[], object],
) -> float:
    value = parameter(parameters, name, fallback_getter)
    return float(cast(float | int | str, value))


def str_parameter(
    parameters: RuntimeParameters,
    name: str,
    fallback_getter: Callable[[], object],
) -> str:
    return str(parameter(parameters, name, fallback_getter))


def write_output(
    output_port: Port, value: float | np.ndarray, num_samples: int
) -> None:
    if value is None:
        output_port.write(silence(num_samples))
    elif isinstance(value, tuple):
        output_port.write(np.asarray(value, dtype=np.float32))
    else:
        output_port.write(value)


__all__ = [
    "DEFAULT_MIN_TRIGGER_SECONDS",
    "EMPTY_PARAMETERS",
    "RuntimeParameters",
    "apply_cv_influence",
    "as_samples",
    "ensure_min_pulse_width",
    "float_parameter",
    "gate_transition_indices",
    "min_trigger_samples",
    "parameter",
    "ramp_if_changed",
    "read_optional_cv",
    "read_optional_port",
    "read_samples",
    "silence",
    "str_parameter",
    "write_output",
    "write_silence_if_disconnected",
]
