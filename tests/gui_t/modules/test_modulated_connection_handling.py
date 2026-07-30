"""Test connection handling optimization for modulated modules.

This test validates that components are created/switched during connection events,
not during process_runtime(), to keep the audio rendering path fast.
"""

from typing import Any

import pytest

from sonicrack.gui.modules.modifier.clipper_mod import ClipperModulatedModule
from sonicrack.gui.modules.modifier.pan_mod import PannerModule
from sonicrack.gui.modules.modifier.volume_mod import VolumeModule
from sonicrack.gui.modules.source.oscillator import OscillatorModule
from sonicrack.runtime.specs import RuntimeParameters


@pytest.fixture
def qapp(qapp: Any):
    """Qt application fixture."""
    return qapp


def test_panner_component_prepared_on_connection(qapp: Any):
    """Test that Panner component is prepared when modulation is connected."""
    del qapp

    panner = PannerModule()
    lfo = OscillatorModule()

    # Initially, panner has unmodulated component
    assert panner.component is not None
    assert panner._is_modulated is False
    assert type(panner.component).__name__ == "Panner"

    # Connect modulation
    lfo.sine_port.connect(panner.mod_port)
    panner.on_port_connection_changed("Mod", True)

    # After connection, component switches to ModulatedPanner
    assert panner.component is not None
    assert type(panner.component).__name__ == "ModulatedPanner"
    assert panner._is_modulated is True

    # Disconnect modulation
    lfo.sine_port.disconnect(panner.mod_port)
    panner.on_port_connection_changed("Mod", False)

    # After disconnection, component switches back to Panner
    assert panner.component is not None
    assert type(panner.component).__name__ == "Panner"
    assert panner._is_modulated is False


def test_volume_component_prepared_on_connection(qapp: Any):
    """Test that Volume component is prepared when modulation is connected."""
    del qapp

    volume = VolumeModule()
    lfo = OscillatorModule()

    # Initially, volume has unmodulated component
    assert volume.component is not None
    assert volume._is_modulated is False
    assert type(volume.component).__name__ == "Volume"

    # Connect modulation
    lfo.sine_port.connect(volume.mod_port)
    volume.on_port_connection_changed("Mod", True)

    # After connection, component switches to ModulatedVolume
    assert volume.component is not None
    assert type(volume.component).__name__ == "ModulatedVolume"
    assert volume._is_modulated is True

    # Disconnect modulation
    lfo.sine_port.disconnect(volume.mod_port)
    volume.on_port_connection_changed("Mod", False)

    # After disconnection, component switches back to Volume
    assert volume.component is not None
    assert type(volume.component).__name__ == "Volume"
    assert volume._is_modulated is False


def test_clipper_component_prepared_on_connection(qapp: Any):
    """Test that Clipper component is prepared when modulation is connected."""
    del qapp

    clipper = ClipperModulatedModule()
    lfo = OscillatorModule()

    # Initially, clipper has unmodulated component
    assert clipper.component is not None
    assert clipper._is_modulated is False
    assert type(clipper.component).__name__ == "Clipper"

    # Connect modulation
    lfo.sine_port.connect(clipper.mod_port)
    clipper.on_port_connection_changed("Mod", True)

    # After connection, component switches to ModulatedClipper
    assert clipper.component is not None
    assert type(clipper.component).__name__ == "ModulatedClipper"
    assert clipper._is_modulated is True

    # Disconnect modulation
    lfo.sine_port.disconnect(clipper.mod_port)
    clipper.on_port_connection_changed("Mod", False)

    # After disconnection, component switches back to Clipper
    assert clipper.component is not None
    assert type(clipper.component).__name__ == "Clipper"
    assert clipper._is_modulated is False


def test_process_runtime_does_not_create_component(qapp: Any):
    """Test that process_runtime() does not create components - they're pre-created."""
    del qapp

    audio_source = OscillatorModule()
    panner = PannerModule()

    # Connect audio
    audio_source.sine_port.connect(panner.in_port)

    # Store initial component reference
    initial_component = panner.component
    assert initial_component is not None

    # Process several buffers
    params: RuntimeParameters = {}
    for _ in range(10):
        audio_source.process_runtime(256, params)
        panner.process_runtime(256, params)

    # Component should be the same instance (not recreated)
    assert panner.component is initial_component, (
        "Component should not be recreated during process_runtime()"
    )


def test_knob_disabled_when_modulation_connected(qapp: Any):
    """Test that control knob remains enabled and changes function when modulated."""
    del qapp

    panner = PannerModule()
    lfo = OscillatorModule()

    # Initially, knob is enabled
    assert panner.pan_knob.isEnabled()

    # Connect modulation
    lfo.sine_port.connect(panner.mod_port)
    panner.on_port_connection_changed("Mod", True)

    # Knob should remain enabled but tooltip changes to indicate depth control
    assert panner.pan_knob.isEnabled(), "Knob should remain enabled when modulated"
    modulated_tooltip = panner.pan_knob.toolTip()
    assert "Modulation Depth" in modulated_tooltip, (
        "Tooltip should indicate depth control"
    )

    # Disconnect modulation
    lfo.sine_port.disconnect(panner.mod_port)
    panner.on_port_connection_changed("Mod", False)

    # Knob should still be enabled with original tooltip
    assert panner.pan_knob.isEnabled()
    # Tooltip may differ slightly but should not mention depth
    assert "Modulation Depth" not in panner.pan_knob.toolTip()


def test_on_port_connection_changed_ignores_other_ports(qapp: Any):
    """Test that on_port_connection_changed only reacts to Mod port."""
    del qapp

    panner = PannerModule()
    initial_component = panner.component
    initial_modulated_state = panner._is_modulated

    # Call with different port name
    panner.on_port_connection_changed("In", True)

    # Should not change component or state
    assert panner.component is initial_component
    assert panner._is_modulated == initial_modulated_state

    panner.on_port_connection_changed("Out", True)

    # Should still not change component or state
    assert panner.component is initial_component
    assert panner._is_modulated == initial_modulated_state
