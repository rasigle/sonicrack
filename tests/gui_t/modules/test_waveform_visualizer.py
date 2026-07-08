"""Waveform visualizer performance-oriented behavior tests."""

from __future__ import annotations

from typing import Any

import numpy as np

from sonicrack.gui.core.port import Port, PortType
from sonicrack.gui.modules.visualization.waveform import WaveformDisplay, WaveformModule


def test_waveform_display_uses_bounded_line_only_paths(qapp: Any) -> None:
    del qapp
    display = WaveformDisplay()
    display.resize(480, 200)
    samples = np.sin(np.linspace(0.0, 80.0 * np.pi, 8192)).astype(np.float32)

    display.set_samples(samples)

    assert display._mono_path is not None
    assert display._target_point_count() == 240
    assert not hasattr(display, "_mono_fill_path")
    assert not hasattr(display, "_left_fill_path")
    assert not hasattr(display, "_right_fill_path")


def test_waveform_display_sanitizes_visible_points(qapp: Any) -> None:
    del qapp
    display = WaveformDisplay()
    display.resize(320, 180)
    samples = np.array([0.0, np.nan, np.inf, -np.inf, 2.0, -2.0], dtype=np.float32)

    display.set_samples(samples)

    assert display._mono_path is not None
    assert display._has_signal is True


def test_waveform_display_handles_stereo_line_paths(qapp: Any) -> None:
    del qapp
    display = WaveformDisplay()
    display.resize(480, 200)
    mono = np.sin(np.linspace(0.0, 20.0 * np.pi, 4096)).astype(np.float32)
    stereo = np.column_stack((mono, -mono)).astype(np.float32)

    display.set_samples(stereo)

    assert display.is_stereo is True
    assert display._left_path is not None
    assert display._right_path is not None


def test_waveform_module_exposes_single_stereo_input(qapp: Any) -> None:
    del qapp
    module = WaveformModule()
    try:
        assert list(module.inputs) == ["In"]
        assert len(module.input_ports) == 1
        assert module.in_port_l is module.in_port
        assert module.in_port_r is module.in_port
    finally:
        module.shutdown()


def test_waveform_module_displays_connected_stereo_signal(qapp: Any) -> None:
    del qapp
    module = WaveformModule()
    mono = np.sin(np.linspace(0.0, 20.0 * np.pi, 1024)).astype(np.float32)
    stereo = np.column_stack((mono, -mono)).astype(np.float32)

    try:
        module._update_samples(stereo)

        assert module.waveform_display.is_stereo is True
        assert module.waveform_display._left_path is not None
        assert module.waveform_display._right_path is not None
        assert module.samples_label.text() == "Samples: 1024 x 2"
    finally:
        module.shutdown()


def test_waveform_module_treats_two_mono_cables_as_stereo(qapp: Any) -> None:
    del qapp
    module = WaveformModule()
    left_source = Port(PortType.OUTPUT, "Left")
    right_source = Port(PortType.OUTPUT, "Right")
    left = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
    right = -left

    try:
        module.in_port.connect(left_source)
        module.in_port.connect(right_source)
        left_source.write(left)
        right_source.write(right)

        samples = module._read_input_samples(256)

        assert samples is not None
        assert samples.shape == (256, 2)
        np.testing.assert_allclose(samples[:, 0], left)
        np.testing.assert_allclose(samples[:, 1], right)
    finally:
        module.shutdown()
