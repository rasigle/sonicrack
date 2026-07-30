"""Tests for port/cable signal color styling."""

from __future__ import annotations

from PyQt6.QtGui import QColor

from sonicrack.gui.widgets.signal_style import (
    cable_color_for_signal,
    color_for_port_signal,
    label_for_port_signal,
    port_color_for_signal,
)
from sonicrack.patching.port import PortSignal


def test_signal_colors_are_distinct_for_primary_kinds():
    audio = color_for_port_signal(PortSignal.AUDIO)
    pitch = color_for_port_signal(PortSignal.PITCH_CV)
    gate = color_for_port_signal(PortSignal.GATE)
    trigger = color_for_port_signal(PortSignal.TRIGGER)

    colors = {
        (audio.red(), audio.green(), audio.blue()),
        (pitch.red(), pitch.green(), pitch.blue()),
        (gate.red(), gate.green(), gate.blue()),
        (trigger.red(), trigger.green(), trigger.blue()),
    }
    assert len(colors) == 4


def test_signal_labels_match_user_facing_names():
    assert label_for_port_signal(PortSignal.AUDIO) == "Audio"
    assert label_for_port_signal(PortSignal.PITCH_CV) == "V/Oct"
    assert label_for_port_signal(PortSignal.GATE) == "Gate"
    assert label_for_port_signal(PortSignal.TRIGGER) == "Trigger"


def test_cable_selection_uses_highlight_color():
    selected = cable_color_for_signal(PortSignal.AUDIO, selected=True)
    normal = cable_color_for_signal(PortSignal.AUDIO)

    assert selected == QColor(255, 220, 80)
    assert normal == color_for_port_signal(PortSignal.AUDIO)


def test_port_hover_lightens_signal_color():
    normal = port_color_for_signal(PortSignal.GATE)
    hovered = port_color_for_signal(PortSignal.GATE, hovered=True)

    assert hovered != normal
    assert hovered.lightness() >= normal.lightness()
