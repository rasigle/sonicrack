"""Small shared helpers for module-owned runtime processors."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from sonicrack.patching.port import Port

RuntimeParameters = Mapping[str, object]

# Shared immutable empty mapping for modules with no runtime parameters.
EMPTY_PARAMETERS: RuntimeParameters = MappingProxyType({})

# Reusable silence buffers keyed by length. Treat returned arrays as read-only.
_SILENCE_CACHE: dict[int, np.ndarray] = {}
_SILENCE_CACHE_MAX_SIZES = 16


def silence(num_samples: int, *, writable: bool = False) -> np.ndarray:
    """Return a zero float32 buffer of ``num_samples``.

    Cached read-only buffers are returned for common sizes so silent paths avoid
    allocating every callback. Pass ``writable=True`` when the caller will mutate
    the array (e.g. padding into a temporary).
    """
    if num_samples <= 0:
        return np.zeros(0, dtype=np.float32)

    if writable:
        return np.zeros(num_samples, dtype=np.float32)

    cached = _SILENCE_CACHE.get(num_samples)
    if cached is None:
        cached = np.zeros(num_samples, dtype=np.float32)
        cached.setflags(write=False)
        if len(_SILENCE_CACHE) >= _SILENCE_CACHE_MAX_SIZES:
            _SILENCE_CACHE.pop(next(iter(_SILENCE_CACHE)))
        _SILENCE_CACHE[num_samples] = cached
    return cached


def as_samples(value: object, num_samples: int) -> np.ndarray:
    if value is None:
        return silence(num_samples)

    if isinstance(value, np.ndarray):
        samples = (
            value
            if value.dtype == np.float32
            else np.asarray(value, dtype=np.float32)
        )
    else:
        samples = np.asarray(value, dtype=np.float32)

    if samples.ndim == 0:
        return np.full(num_samples, float(samples), dtype=np.float32)

    if len(samples) == num_samples:
        return samples

    if len(samples) < num_samples:
        if samples.ndim == 1:
            padded = np.zeros(num_samples, dtype=np.float32)
        else:
            padded = np.zeros((num_samples, *samples.shape[1:]), dtype=np.float32)
        padded[: len(samples)] = samples
        return padded

    return samples[:num_samples].copy()


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


def ramp_if_changed(
    previous: float, current: float, num_samples: int
) -> np.ndarray | None:
    """Return a linear ramp across the buffer when a scalar param changes."""
    if previous == current:
        return None
    return np.linspace(previous, current, num_samples, dtype=np.float32)


def apply_cv_influence(
    cv_amplitude: float | np.ndarray,
    influence: float,
    *,
    output_gain: float | np.ndarray,
) -> float | np.ndarray:
    """Blend unipolar CV with a fixed gain, then apply final output gain.

    Used by VCA-style amplitude control: influence 0 keeps only ``output_gain``,
    influence 1 is fully CV-driven, then both are scaled by ``output_gain``.

    ``output_gain`` may be a scalar or a per-sample curve (e.g. dezippered gain).
    """
    amount = max(0.0, min(1.0, float(influence)))
    cv = np.asarray(cv_amplitude, dtype=np.float32)
    gain = np.asarray(output_gain, dtype=np.float32)
    modulation = (1.0 - amount) + (cv * amount)
    result = np.clip(modulation * gain, 0.0, 1.0)
    if result.ndim == 0:
        return float(result)
    return result.astype(np.float32, copy=False)


def gate_transition_indices(
    gate_signal: np.ndarray,
    previous_gate: float,
    *,
    low: float = 0.3,
    high: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Find sample indices of Schmitt gate note-on / note-off transitions.

    Returns:
        (note_on_indices, note_off_indices, final_gate_level)
    """
    values = np.asarray(gate_signal, dtype=np.float32).reshape(-1)
    if values.size == 0:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty, float(previous_gate)

    prev = np.empty_like(values)
    prev[0] = previous_gate
    prev[1:] = values[:-1]

    note_ons = np.flatnonzero((prev < low) & (values > high))
    note_offs = np.flatnonzero((prev > high) & (values < low))
    return note_ons, note_offs, float(values[-1])


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
