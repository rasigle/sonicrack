import math

import numpy as np
import pytest

from src.engine.utils.math import db_to_linear, linear_to_db, squish_val


class TestOscillatorDBControl:
    """Test dB control in Oscillator class."""

    def test_db_to_linear_conversion(self):
        """Test dB to linear amplitude conversion."""
        assert db_to_linear(0) == pytest.approx(1.0)
        assert db_to_linear(-6) == pytest.approx(0.5, rel=0.01)
        assert db_to_linear(-20) == pytest.approx(0.1)
        assert db_to_linear(6) == pytest.approx(2.0, rel=0.01)
        assert db_to_linear(-40) == pytest.approx(0.01)

    def test_linear_to_db_conversion(self):
        """Test linear to dB conversion."""
        assert linear_to_db(1.0) == pytest.approx(0.0)
        assert linear_to_db(0.5) == pytest.approx(-6.0, rel=0.1)
        assert linear_to_db(0.1) == pytest.approx(-20.0)
        assert linear_to_db(2.0) == pytest.approx(6.0, rel=0.1)
        assert linear_to_db(0.0) == -math.inf

    def test_converts_db_to_linear_correctly(self):
        assert db_to_linear(0) == 1.0
        assert np.isclose(db_to_linear(-6), 0.5011872336272722)
        assert np.isclose(db_to_linear(-20), 0.1)
        assert np.isclose(db_to_linear(6), 1.99526231)

    def test_handles_large_negative_db_values(self):
        assert db_to_linear(-100) == 10 ** (-100 / 20.0)

    def test_converts_linear_to_db_correctly(self):
        assert linear_to_db(1.0) == 0.0
        assert np.isclose(linear_to_db(0.5), -6.020599913279624)
        assert np.isclose(linear_to_db(0.1), -20.0)

    def test_handles_zero_or_negative_linear_values(self):
        assert linear_to_db(0.0) == -math.inf
        assert linear_to_db(-1.0) == -math.inf


class TestSquishVal:
    def test_maps_value_to_range_correctly(self):
        assert squish_val(0, 0, 10) == 5.0
        assert squish_val(-1, 0, 10) == 0.0
        assert squish_val(1, 0, 10) == 10.0

    def test_handles_custom_min_and_max_ranges(self):
        assert squish_val(0, -5, 5) == 0.0
        assert squish_val(-1, -5, 5) == -5.0
        assert squish_val(1, -5, 5) == 5.0

    def test_handles_edge_cases_for_squish_val(self):
        assert squish_val(0, 0, 0) == 0.0
        assert squish_val(0, -1, -1) == -1.0
        assert squish_val(0, 1, 1) == 1.0
