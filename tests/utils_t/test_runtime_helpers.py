"""Unit tests for shared runtime helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from sonicrack.runtime.helpers import (
    EMPTY_PARAMETERS,
    apply_cv_influence,
    as_samples,
    float_parameter,
    gate_transition_indices,
    parameter,
    ramp_if_changed,
    read_optional_port,
    silence,
    str_parameter,
    write_silence_if_disconnected,
)


class TestSilenceAndAsSamples:
    def test_silence_shape_and_dtype(self):
        buf = silence(8)
        assert buf.shape == (8,)
        assert buf.dtype == np.float32
        assert np.allclose(buf, 0.0)

    def test_silence_cache_returns_same_readonly_buffer(self):
        a = silence(16)
        b = silence(16)
        assert a is b
        assert not a.flags.writeable

    def test_silence_writable_is_unique(self):
        a = silence(16, writable=True)
        b = silence(16, writable=True)
        assert a is not b
        assert a.flags.writeable

    def test_as_samples_none_and_scalar(self):
        assert np.allclose(as_samples(None, 4), np.zeros(4, dtype=np.float32))
        assert np.allclose(as_samples(0.5, 4), np.full(4, 0.5, dtype=np.float32))

    def test_as_samples_pads_and_truncates(self):
        short = as_samples(np.array([1.0, 2.0], dtype=np.float32), 4)
        long = as_samples(np.arange(6, dtype=np.float32), 4)
        assert short.tolist() == [1.0, 2.0, 0.0, 0.0]
        assert long.tolist() == [0.0, 1.0, 2.0, 3.0]


class TestOptionalPortAndSilence:
    def test_read_optional_port_none_when_disconnected(self):
        port = SimpleNamespace(is_connected=False)
        assert read_optional_port(None, 8) is None
        assert read_optional_port(port, 8) is None

    def test_read_optional_port_reads_when_connected(self):
        samples = np.linspace(0.0, 1.0, 8, dtype=np.float32)

        class _Port:
            is_connected = True

            def read(self, num_samples: int):
                assert num_samples == 8
                return samples

        result = read_optional_port(_Port(), 8)
        assert result is not None
        assert np.allclose(result, samples)

    def test_write_silence_if_disconnected(self):
        written: list[np.ndarray] = []
        out = SimpleNamespace(write=lambda value: written.append(np.asarray(value)))

        connected = SimpleNamespace(is_connected=True)
        assert write_silence_if_disconnected(connected, out, 4) is False
        assert written == []

        disconnected = SimpleNamespace(is_connected=False)
        assert write_silence_if_disconnected(disconnected, out, 4) is True
        assert len(written) == 1
        assert np.allclose(written[0], np.zeros(4, dtype=np.float32))


class TestRampAndCvInfluence:
    def test_ramp_if_changed_none_when_equal(self):
        assert ramp_if_changed(1.0, 1.0, 8) is None

    def test_ramp_if_changed_linear(self):
        ramp = ramp_if_changed(0.0, 1.0, 5)
        assert ramp is not None
        np.testing.assert_allclose(ramp, np.linspace(0.0, 1.0, 5, dtype=np.float32))

    def test_apply_cv_influence_extremes(self):
        # influence 0 -> pure gain
        assert float(apply_cv_influence(0.25, 0.0, output_gain=0.8)) == pytest.approx(
            0.8
        )
        # influence 1 -> CV * gain
        assert float(apply_cv_influence(0.5, 1.0, output_gain=0.8)) == pytest.approx(
            0.4
        )

    def test_apply_cv_influence_clamps(self):
        assert float(apply_cv_influence(2.0, 1.0, output_gain=1.0)) == pytest.approx(
            1.0
        )
        values = apply_cv_influence(
            np.array([-1.0, 0.5, 2.0], dtype=np.float32),
            1.0,
            output_gain=1.0,
        )
        assert np.allclose(values, [0.0, 0.5, 1.0])

    def test_apply_cv_influence_accepts_gain_curve(self):
        cv = np.ones(4, dtype=np.float32)
        gain = np.linspace(0.0, 1.0, 4, dtype=np.float32)
        values = apply_cv_influence(cv, 1.0, output_gain=gain)
        np.testing.assert_allclose(values, gain)


class TestGateAndParameters:
    def test_gate_transition_indices_empty(self):
        ons, offs, final = gate_transition_indices(np.array([]), 0.0)
        assert ons.size == 0
        assert offs.size == 0
        assert final == 0.0

    def test_gate_transition_indices_detects_edges(self):
        gate = np.array([0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32)
        ons, offs, final = gate_transition_indices(gate, previous_gate=0.0)
        assert ons.tolist() == [2]
        assert offs.tolist() == [4]
        assert final == pytest.approx(0.0)

    def test_parameter_helpers(self):
        params = {"cutoff": "440", "mode": "low"}
        assert parameter(params, "cutoff", lambda: 0) == "440"
        assert parameter(params, "missing", lambda: 7) == 7
        assert float_parameter(params, "cutoff", lambda: 0) == pytest.approx(440.0)
        assert str_parameter(params, "mode", lambda: "high") == "low"
        assert str_parameter({}, "mode", lambda: "high") == "high"

    def test_empty_parameters_is_immutable_mapping(self):
        assert dict(EMPTY_PARAMETERS) == {}
        with pytest.raises(TypeError):
            EMPTY_PARAMETERS["x"] = 1  # type: ignore[index]

    def test_write_output_variants(self):
        from sonicrack.runtime.helpers import write_output

        written: list[object] = []
        port = SimpleNamespace(write=lambda value: written.append(value))

        write_output(port, None, 4)  # type: ignore[arg-type]
        assert np.allclose(written[-1], np.zeros(4, dtype=np.float32))

        write_output(port, (0.1, 0.2), 2)  # type: ignore[arg-type]
        assert np.allclose(written[-1], np.array([0.1, 0.2], dtype=np.float32))

        samples = np.ones(3, dtype=np.float32)
        write_output(port, samples, 3)  # type: ignore[arg-type]
        assert written[-1] is samples

    def test_silence_zero_length_and_writable_padding_path(self):
        assert silence(0).shape == (0,)
        writable = silence(4, writable=True)
        writable[0] = 1.0
        assert writable[0] == 1.0
