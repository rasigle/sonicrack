"""Shared envelope curve labels and normalization for ADSR / Decay."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

EnvelopeCurveName = Literal["linear", "exponential", "polynomial"]

ENVELOPE_CURVE_ITEMS: tuple[str, ...] = ("Linear", "Exponential", "Polynomial")

ENVELOPE_CURVE_TOOLTIP_ADSR = (
    "Segment interpolation for attack, decay, and release.\n"
    "Linear: straight ramps.\n"
    "Exponential: analog RC shape — natural VCA feel (recommended).\n"
    "Polynomial: smooth S-curve (Hermite smoothstep)."
)

ENVELOPE_CURVE_TOOLTIP_DECAY = (
    "Segment interpolation for attack and decay.\n"
    "Linear: straight ramps.\n"
    "Exponential: analog RC shape — natural pluck tails (recommended).\n"
    "Polynomial: smooth S-curve (Hermite smoothstep)."
)


def normalize_envelope_curve(value: str) -> EnvelopeCurveName:
    """Map a UI or patch curve label to the soniclab curve name."""
    lowered = value.lower().strip()
    if lowered == "polynomial":
        return "polynomial"
    if lowered == "linear":
        return "linear"
    return "exponential"


def apply_envelope_curve(component: object, value: str) -> EnvelopeCurveName:
    """Set ``component.curve`` when the installed soniclab supports it."""
    curve = normalize_envelope_curve(value)
    if hasattr(component, "curve"):
        component.curve = curve
    return curve


def register_envelope_curve_choice(
    module: object,
    on_changed: Callable[[str], None],
    *,
    tooltip: str,
) -> object:
    """Register the shared Curve context-menu choice on an envelope module."""
    return module.register_menu_choice(  # type: ignore[attr-defined]
        "curve",
        "Curve",
        ENVELOPE_CURVE_ITEMS,
        "Exponential",
        on_changed=on_changed,
        tooltip=tooltip,
    )
