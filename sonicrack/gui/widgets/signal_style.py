"""Shared visual styling for port signal kinds (cables and jacks)."""

from __future__ import annotations

from PyQt6.QtGui import QColor

from sonicrack.patching.port import PortSignal, normalize_port_signal

# Distinct palette tuned for the dark patch canvas.
# Audio / V/Oct / Gate / Trigger are the primary cable identities.
_SIGNAL_COLORS: dict[PortSignal, QColor] = {
    PortSignal.AUDIO: QColor(255, 159, 67),  # Orange
    PortSignal.PITCH_CV: QColor(74, 158, 255),  # Blue (V/Oct)
    PortSignal.GATE: QColor(245, 215, 110),  # Gold / yellow
    PortSignal.TRIGGER: QColor(255, 94, 122),  # Coral / pink-red
    PortSignal.CONTROL_CV: QColor(179, 136, 255),  # Purple
    PortSignal.FREQUENCY_HZ: QColor(46, 204, 113),  # Green
    PortSignal.UNKNOWN: QColor(136, 136, 136),  # Gray
}

_SIGNAL_LABELS: dict[PortSignal, str] = {
    PortSignal.AUDIO: "Audio",
    PortSignal.PITCH_CV: "V/Oct",
    PortSignal.GATE: "Gate",
    PortSignal.TRIGGER: "Trigger",
    PortSignal.CONTROL_CV: "Control CV",
    PortSignal.FREQUENCY_HZ: "Frequency (Hz)",
    PortSignal.UNKNOWN: "Unknown",
}

_SELECTION_COLOR = QColor(255, 220, 80)


def color_for_port_signal(signal: PortSignal | str | None) -> QColor:
    """Return the base display color for a port signal kind."""
    kind = normalize_port_signal(signal)
    return QColor(_SIGNAL_COLORS.get(kind, _SIGNAL_COLORS[PortSignal.UNKNOWN]))


def label_for_port_signal(signal: PortSignal | str | None) -> str:
    """Return a short human-readable label for a port signal kind."""
    kind = normalize_port_signal(signal)
    return _SIGNAL_LABELS.get(kind, _SIGNAL_LABELS[PortSignal.UNKNOWN])


def cable_color_for_signal(
    signal: PortSignal | str | None,
    *,
    selected: bool = False,
    hovered: bool = False,
) -> QColor:
    """Return the cable stroke color for the given signal and interaction state."""
    if selected:
        return QColor(_SELECTION_COLOR)

    color = color_for_port_signal(signal)
    if hovered:
        return color.lighter(130)
    return color


def port_color_for_signal(
    signal: PortSignal | str | None,
    *,
    hovered: bool = False,
) -> QColor:
    """Return the jack fill color for the given signal and hover state."""
    color = color_for_port_signal(signal)
    if hovered:
        return color.lighter(120)
    return color
