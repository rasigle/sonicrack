"""Small shared helpers for module-owned runtime processors."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

import numpy as np

RuntimeParameters = Mapping[str, object]


def silence(num_samples: int) -> np.ndarray:
    return np.zeros(num_samples, dtype=np.float32)


def as_samples(value: object, num_samples: int) -> np.ndarray:
    if value is None:
        return silence(num_samples)
    samples = np.asarray(value, dtype=np.float32)
    if samples.ndim == 0:
        return np.full(num_samples, float(samples), dtype=np.float32)
    if len(samples) == num_samples:
        return samples
    if len(samples) < num_samples:
        padded = silence(num_samples)
        padded[: len(samples)] = samples
        return padded
    return samples[:num_samples].copy()


def read_samples(port: Any, num_samples: int) -> np.ndarray:
    return as_samples(port.read(num_samples), num_samples)


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


def write_output(output_port: Any, value: object, num_samples: int) -> None:
    if value is None:
        output_port.write(silence(num_samples))
    elif isinstance(value, tuple):
        output_port.write(np.asarray(value, dtype=np.float32))
    else:
        output_port.write(value)
