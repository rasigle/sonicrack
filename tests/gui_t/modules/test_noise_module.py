"""Tests for the GUI noise module runtime behavior."""

from __future__ import annotations

import pytest

from sonicrack.gui.modules.source.noise import NoiseModule


def test_noise_runtime_updates_gain_from_parameters(qapp):
    del qapp
    module = NoiseModule()

    module.process_runtime(16, {"noise_type": "White", "gain_db": -60.0})
    assert module.component is not None
    assert module.component.gain_db == pytest.approx(-60.0)

    module.process_runtime(16, {"noise_type": "White", "gain_db": 0.0})
    assert module.component.gain_db == pytest.approx(0.0)
