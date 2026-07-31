"""Waveform visualizer performance-oriented behavior tests."""

from __future__ import annotations

from typing import Any

import numpy as np

from sonicrack.gui.modules.visualization.spectrum import SpectrumModule
from sonicrack.gui.modules.visualization.visualizer_utils import (
    process_visualizer_passthrough,
)
from sonicrack.gui.modules.visualization.waveform import WaveformDisplay, WaveformModule
from sonicrack.patching.port import Port, PortType
from sonicrack.runtime.helpers import EMPTY_PARAMETERS


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
        assert list(module.outputs) == ["Out"]
        assert len(module.input_ports) == 1
        assert len(module.output_ports) == 1
        assert module.in_port_l is module.in_port
        assert module.in_port_r is module.in_port
        assert module.is_processing_module is True
        assert module.get_required_inputs() == ["In"]
    finally:
        module.shutdown()


def test_waveform_module_passthrough_copies_input_unchanged(qapp: Any) -> None:
    del qapp
    module = WaveformModule()
    source = Port(PortType.OUTPUT, "Src")
    mono = np.linspace(-0.5, 0.5, 128, dtype=np.float32)

    try:
        module.in_port.connect(source)
        source.write(mono)

        module.process_runtime(128, EMPTY_PARAMETERS)

        out = module.out_port.value
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, mono)
        # Output owns a copy so downstream mutation cannot touch the source.
        assert out is not mono
        assert out is not source.value
    finally:
        module.shutdown()


def test_waveform_module_passthrough_writes_silence_when_disconnected(
    qapp: Any,
) -> None:
    del qapp
    module = WaveformModule()
    try:
        module.process_runtime(64, EMPTY_PARAMETERS)
        out = module.out_port.value
        assert isinstance(out, np.ndarray)
        assert out.shape == (64,)
        np.testing.assert_array_equal(out, np.zeros(64, dtype=np.float32))
    finally:
        module.shutdown()


def test_spectrum_module_exposes_passthrough_output(qapp: Any) -> None:
    del qapp
    module = SpectrumModule()
    source = Port(PortType.OUTPUT, "Src")
    mono = np.sin(np.linspace(0.0, 4.0 * np.pi, 256)).astype(np.float32)

    try:
        assert list(module.inputs) == ["In"]
        assert list(module.outputs) == ["Out"]
        assert module.is_processing_module is True
        assert module.get_required_inputs() == ["In"]

        module.in_port.connect(source)
        source.write(mono)
        module.process_runtime(256, EMPTY_PARAMETERS)

        out = module.out_port.value
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, mono)
    finally:
        module.shutdown()


def test_process_visualizer_passthrough_helper(qapp: Any) -> None:
    del qapp
    in_port = Port(PortType.INPUT, "In")
    out_port = Port(PortType.OUTPUT, "Out")
    source = Port(PortType.OUTPUT, "Src")
    samples = np.array([0.1, -0.2, 0.3], dtype=np.float32)

    in_port.connect(source)
    source.write(samples)
    process_visualizer_passthrough(in_port, out_port, 3)

    np.testing.assert_array_equal(out_port.value, samples)


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
