"""Tests for module cable connect/disconnect lifecycle bugs."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QKeyEvent

from sonicrack.gui.modules.modifier.volume import VolumeModule
from sonicrack.gui.modules.source.oscillator import OscillatorModule
from sonicrack.gui.widgets.patch_canvas import PatchCanvas
from sonicrack.gui.widgets.port_widget import PortWidget
from sonicrack.patching.port import Port


def _attach_main_window_like_handlers(canvas: PatchCanvas) -> list[tuple[str, bool]]:
    """Mimic MainWindow connect/disconnect side effects for unit tests."""
    notifications: list[tuple[str, bool]] = []

    def on_connected(start: PortWidget, target: PortWidget) -> None:
        start.port.connect(target.port)
        for module, port in (
            (target.parent_module, target),
            (start.parent_module, start),
        ):
            handler = getattr(module, "on_port_connection_changed", None)
            if callable(handler):
                connected = port.port.is_connected
                handler(port.port_name, connected)
                notifications.append((port.port_name, connected))

    def on_disconnected(start: PortWidget, target: PortWidget) -> None:
        if start is None or target is None:
            return
        start.port.disconnect(target.port)
        for module, port in (
            (target.parent_module, target),
            (start.parent_module, start),
        ):
            handler = getattr(module, "on_port_connection_changed", None)
            if callable(handler):
                connected = port.port.is_connected
                handler(port.port_name, connected)
                notifications.append((port.port_name, connected))

    canvas.cable_connected.connect(on_connected)
    canvas.cable_disconnected.connect(on_disconnected)
    return notifications


def test_create_connection_rejects_cycles(qapp: Any):
    del qapp
    canvas = PatchCanvas()
    first = VolumeModule()
    second = VolumeModule()
    canvas.add_module(first)
    canvas.add_module(second)

    assert canvas.create_connection(first.output_ports[0], second.input_ports[0])
    cycle = canvas._would_create_cycle(second.output_ports[0], first.input_ports[0])
    assert cycle is not None
    assert (
        canvas.create_connection(second.output_ports[0], first.input_ports[0]) is None
    )
    assert len(canvas.get_connections()) == 1


def test_create_connection_rejects_duplicate_cables(qapp: Any):
    del qapp
    canvas = PatchCanvas()
    source = OscillatorModule()
    target = VolumeModule()
    canvas.add_module(source)
    canvas.add_module(target)

    first = canvas.create_connection(source.output_ports[0], target.input_ports[0])
    second = canvas.create_connection(source.output_ports[0], target.input_ports[0])

    assert first is not None
    assert second is first
    assert len(target.input_ports[0].cables) == 1
    assert len(canvas.get_connections()) == 1


def test_bulk_cable_delete_notifies_all_modulated_targets(qapp: Any):
    del qapp
    canvas = PatchCanvas()
    lfo1 = OscillatorModule()
    lfo2 = OscillatorModule()
    vol1 = VolumeModule()
    vol2 = VolumeModule()
    for module in (lfo1, lfo2, vol1, vol2):
        canvas.add_module(module)

    _attach_main_window_like_handlers(canvas)

    c1 = canvas.create_connection(lfo1.output_ports[0], vol1.input_ports[1])
    c2 = canvas.create_connection(lfo2.output_ports[0], vol2.input_ports[1])
    assert c1 is not None and c2 is not None
    assert vol1._is_modulated is True
    assert vol2._is_modulated is True

    c1.setSelected(True)
    c2.setSelected(True)
    canvas.keyPressEvent(
        QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier
        )
    )

    assert vol1._is_modulated is False
    assert vol2._is_modulated is False
    assert type(vol1.component).__name__ == "Volume"
    assert type(vol2.component).__name__ == "Volume"
    assert not vol1.mod_port.is_connected
    assert not vol2.mod_port.is_connected


def test_delete_from_patch_demotes_remaining_modulated_module(qapp: Any):
    del qapp
    canvas = PatchCanvas()
    lfo = OscillatorModule()
    volume = VolumeModule()
    canvas.add_module(lfo)
    canvas.add_module(volume)

    _attach_main_window_like_handlers(canvas)

    cable = canvas.create_connection(lfo.output_ports[0], volume.input_ports[1])
    assert cable is not None
    assert volume._is_modulated is True
    assert type(volume.component).__name__ == "ModulatedVolume"

    lfo.delete_from_patch()

    assert volume._is_modulated is False
    assert type(volume.component).__name__ == "Volume"
    assert not volume.mod_port.is_connected
    assert len(volume.input_ports[1].cables) == 0
    assert lfo not in canvas.get_modules()


def test_port_widget_connected_to_follows_cables(qapp: Any):
    del qapp
    canvas = PatchCanvas()
    source = OscillatorModule()
    target = VolumeModule()
    canvas.add_module(source)
    canvas.add_module(target)

    assert source.output_ports[0].connected_to is None
    cable = canvas.create_connection(source.output_ports[0], target.input_ports[0])
    assert cable is not None

    assert source.output_ports[0].connected_to is target.input_ports[0]
    assert target.input_ports[0].connected_to is source.output_ports[0]


def test_partial_disconnect_preserves_multi_fanout_value():
    output = Port("output", "Out")
    input1 = Port("input", "In1")
    input2 = Port("input", "In2")
    output.connect(input1)
    output.connect(input2)
    output.value = 0.75

    output.disconnect(input1)

    assert len(output.connected_to) == 1
    assert input2 in output.connected_to
    assert output.value == 0.75  # still live fanout
    assert not input1.is_connected
    assert input1.value == 0.0  # peer fully cleared


def test_disconnect_all_clears_lonely_peers():
    output = Port("output", "Out")
    input1 = Port("input", "In1")
    output.connect(input1)
    output.value = 1.0
    input1.value = 2.0

    output.disconnect()

    assert not output.is_connected
    assert not input1.is_connected
    assert output.value == 0.0
    assert input1.value == 0.0


def test_key_delete_module_demotes_modulated_neighbor(qapp: Any):
    del qapp
    canvas = PatchCanvas()
    lfo = OscillatorModule()
    volume = VolumeModule()
    canvas.add_module(lfo)
    canvas.add_module(volume)
    _attach_main_window_like_handlers(canvas)

    assert canvas.create_connection(lfo.output_ports[0], volume.input_ports[1])
    assert volume._is_modulated is True

    lfo.setSelected(True)
    canvas.keyPressEvent(
        QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier
        )
    )

    assert volume._is_modulated is False
    assert type(volume.component).__name__ == "Volume"


def test_cable_geometry_tracks_module_move(qapp: Any):
    """Moving a connected module must refresh the cable endpoint cache."""
    del qapp
    canvas = PatchCanvas()
    source = OscillatorModule()
    target = VolumeModule()
    source.setPos(0, 0)
    target.setPos(300, 0)
    canvas.add_module(source)
    canvas.add_module(target)

    cable = canvas.create_connection(source.output_ports[0], target.input_ports[0])
    assert cable is not None

    start_before = cable._geom_start
    end_before = cable._geom_end
    assert start_before != end_before

    target.setPos(500, 120)

    # Cached endpoints must match live port scene positions after the move.
    assert cable._geom_start == source.output_ports[0].get_scene_pos()
    assert cable._geom_end == target.input_ports[0].get_scene_pos()
    assert cable._geom_end != end_before
    assert cable.boundingRect().contains(cable._geom_start)
    assert cable.boundingRect().contains(cable._geom_end)


def test_cable_temp_drag_updates_geometry_cache(qapp: Any):
    del qapp
    canvas = PatchCanvas()
    source = OscillatorModule()
    source.setPos(0, 0)
    canvas.add_module(source)

    from PyQt6.QtCore import QPointF

    from sonicrack.gui.widgets.cable_widget import Cable

    cable = Cable(source.output_ports[0])
    canvas._scene.addItem(cable)

    cable.set_temp_end_pos(QPointF(200, 80))
    assert cable._geom_end == QPointF(200, 80)
    assert cable.boundingRect().contains(QPointF(200, 80))

    cable.set_temp_end_pos(QPointF(40, 250))
    assert cable._geom_end == QPointF(40, 250)
    assert cable.boundingRect().contains(QPointF(40, 250))


def test_clear_all_disconnects_port_models(qapp: Any):
    del qapp
    canvas = PatchCanvas()
    source = OscillatorModule()
    target = VolumeModule()
    canvas.add_module(source)
    canvas.add_module(target)
    _attach_main_window_like_handlers(canvas)

    assert canvas.create_connection(source.output_ports[0], target.input_ports[0])
    assert source.output_ports[0].port.is_connected
    assert target.input_ports[0].port.is_connected

    out_port = source.output_ports[0].port
    in_port = target.input_ports[0].port
    canvas.clear_all()

    assert not out_port.is_connected
    assert not in_port.is_connected
    assert canvas.get_modules() == []
    assert canvas.get_connections() == []


def test_runtime_demotes_volume_when_mod_disconnected_without_notify(qapp: Any):
    del qapp

    volume = VolumeModule()
    lfo = OscillatorModule()
    lfo.sine_port.connect(volume.mod_port)
    volume.on_port_connection_changed("Mod", True)
    assert volume._is_modulated is True

    # Sever model without UI notification.
    lfo.sine_port.disconnect(volume.mod_port)
    assert volume._is_modulated is True

    volume.process_runtime(64, {})
    assert volume._is_modulated is False
    assert type(volume.component).__name__ == "Volume"
