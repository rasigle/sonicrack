import pytest

from src.utils.math import db_to_linear, linear_to_db


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
        assert linear_to_db(0.0) == float("-inf")
