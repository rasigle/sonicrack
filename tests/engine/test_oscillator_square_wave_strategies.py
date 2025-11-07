"""Tests for square wave generation strategies.

This module tests all square wave formulations to ensure correctness,
performance, and proper factory behavior.
"""

import numpy as np
import pytest

from src.engine.oscillator_square import (
    SquareWaveFactory,
    SquareWaveStrategy,
    IdealSquareStrategy,
    BandlimitedSquareStrategy,
    SoftSquareStrategy,
    ComparatorSquareStrategy,
)


class TestSquareWaveFactory:
    """Test the square wave factory."""

    def test_factory_creates_ideal_strategy(self):
        """Factory should create ideal strategy."""
        strategy = SquareWaveFactory.create("ideal")
        assert isinstance(strategy, IdealSquareStrategy)

    def test_factory_creates_bandlimited_strategy(self):
        """Factory should create bandlimited strategy."""
        strategy = SquareWaveFactory.create("bandlimited", sample_rate=48000)
        assert isinstance(strategy, BandlimitedSquareStrategy)
        assert strategy.sample_rate == 48000

    def test_factory_creates_soft_strategy(self):
        """Factory should create soft strategy."""
        strategy = SquareWaveFactory.create("soft", smoothness=15.0)
        assert isinstance(strategy, SoftSquareStrategy)
        assert strategy.smoothness == 15.0

    def test_factory_creates_comparator_strategy(self):
        """Factory should create comparator strategy."""
        strategy = SquareWaveFactory.create("comparator", hysteresis=0.05)
        assert isinstance(strategy, ComparatorSquareStrategy)
        assert strategy.hysteresis == 0.05

    def test_factory_unknown_mode_raises(self):
        """Factory should raise error for unknown mode."""
        with pytest.raises(ValueError, match="Unknown square wave mode"):
            SquareWaveFactory.create("invalid_mode")

    def test_factory_get_available_modes(self):
        """Factory should return list of available modes."""
        modes = SquareWaveFactory.get_available_modes()
        assert isinstance(modes, list)
        assert "ideal" in modes
        assert "bandlimited" in modes
        assert "soft" in modes
        assert "comparator" in modes

    def test_factory_register_custom_strategy(self):
        """Factory should allow registering custom strategies."""

        class CustomStrategy(SquareWaveStrategy):
            def generate_sample(self, phase, pulsewidth_threshold, low_value, high_value):
                return 0.5

            def generate_samples(self, phases, pulsewidth_threshold, low_value, high_value):
                return np.full_like(phases, 0.5)

        SquareWaveFactory.register_strategy("custom", CustomStrategy)

        strategy = SquareWaveFactory.create("custom")
        assert isinstance(strategy, CustomStrategy)

        # Verify it works
        assert strategy.generate_sample(0, np.pi, -1, 1) == 0.5

    def test_factory_register_invalid_strategy_raises(self):
        """Factory should reject non-strategy classes."""

        class NotAStrategy:
            pass

        with pytest.raises(TypeError):
            SquareWaveFactory.register_strategy("bad", NotAStrategy)


class TestIdealSquareStrategy:
    """Test ideal square wave strategy."""

    @pytest.fixture
    def strategy(self):
        return IdealSquareStrategy()

    def test_ideal_50_percent_duty_cycle(self, strategy):
        """Ideal strategy should produce 50% duty cycle at threshold=π."""
        # Test single sample
        assert strategy.generate_sample(0, np.pi, -1, 1) == 1  # High
        assert strategy.generate_sample(np.pi, np.pi, -1, 1) == -1  # Low
        assert strategy.generate_sample(2*np.pi - 0.01, np.pi, -1, 1) == -1  # Low

    def test_ideal_25_percent_duty_cycle(self, strategy):
        """Ideal strategy should produce 25% duty cycle at threshold=π/2."""
        threshold = np.pi / 2
        assert strategy.generate_sample(0, threshold, -1, 1) == 1  # High
        assert strategy.generate_sample(threshold - 0.01, threshold, -1, 1) == 1  # Still high
        assert strategy.generate_sample(threshold, threshold, -1, 1) == -1  # Low
        assert strategy.generate_sample(np.pi, threshold, -1, 1) == -1  # Low

    def test_ideal_vectorized_matches_sample_by_sample(self, strategy):
        """Vectorized and sample-by-sample should produce same results."""
        phases = np.linspace(0, 2*np.pi, 100)
        threshold = np.pi

        # Vectorized
        vectorized = strategy.generate_samples(phases, threshold, -1, 1)

        # Sample by sample
        samples = [strategy.generate_sample(p, threshold, -1, 1) for p in phases]

        np.testing.assert_array_equal(vectorized, samples)

    def test_ideal_custom_output_range(self, strategy):
        """Ideal strategy should respect custom output ranges."""
        result = strategy.generate_sample(0, np.pi, 0, 10)
        assert result == 10  # High value

        result = strategy.generate_sample(np.pi, np.pi, 0, 10)
        assert result == 0  # Low value


class TestBandlimitedSquareStrategy:
    """Test bandlimited square wave strategy."""

    @pytest.fixture
    def strategy(self):
        return BandlimitedSquareStrategy(sample_rate=44100)

    def test_bandlimited_has_blep_table(self, strategy):
        """Bandlimited strategy should have BLEP table."""
        assert hasattr(strategy, 'blep_table')
        assert len(strategy.blep_table) > 0
        assert isinstance(strategy.blep_table, np.ndarray)

    def test_bandlimited_blep_table_properties(self, strategy):
        """BLEP table should start at 0 and end at 1."""
        assert strategy.blep_table[0] == pytest.approx(0, abs=0.01)
        assert strategy.blep_table[-1] == pytest.approx(1, abs=0.01)
        # Should be monotonically increasing
        assert np.all(np.diff(strategy.blep_table) >= 0)

    def test_bandlimited_reduces_high_frequency_content(self, strategy):
        """Bandlimited square should have less high-frequency content than ideal."""
        # Generate one period
        phases = np.linspace(0, 2*np.pi, 1000)
        threshold = np.pi

        # Compare with ideal
        ideal = IdealSquareStrategy()
        ideal_samples = ideal.generate_samples(phases, threshold, -1, 1)
        bandlimited_samples = strategy.generate_samples(phases, threshold, -1, 1)

        # Check that transitions are smoother (less instant jumps)
        ideal_diffs = np.abs(np.diff(ideal_samples))
        bandlimited_diffs = np.abs(np.diff(bandlimited_samples))

        # Ideal should have sharper transitions
        assert np.max(ideal_diffs) >= np.max(bandlimited_diffs)


class TestSoftSquareStrategy:
    """Test soft square wave strategy."""

    @pytest.fixture
    def strategy(self):
        return SoftSquareStrategy(smoothness=10.0)

    def test_soft_has_smooth_transitions(self, strategy):
        """Soft strategy should have smooth transitions around threshold."""
        threshold = np.pi

        # Sample around threshold
        phases = np.linspace(threshold - 0.5, threshold + 0.5, 100)
        samples = strategy.generate_samples(phases, threshold, -1, 1)

        # Should have gradual transition (no instant jumps)
        diffs = np.abs(np.diff(samples))
        assert np.max(diffs) < 0.3  # No large jumps

        # Should be monotonic in transition region
        # (values should gradually change from high to low)

    def test_soft_extremes_approach_limits(self, strategy):
        """Soft strategy should approach high/low values away from threshold."""
        threshold = np.pi

        # Far from threshold (high side)
        high = strategy.generate_sample(0, threshold, -1, 1)
        assert high > 0.5  # Should be closer to 1

        # Far from threshold (low side)
        low = strategy.generate_sample(2*np.pi - 0.1, threshold, -1, 1)
        assert low < -0.5  # Should be closer to -1

    def test_soft_smoothness_parameter_effect(self):
        """Higher smoothness should create sharper transitions."""
        smooth_low = SoftSquareStrategy(smoothness=5.0)
        smooth_high = SoftSquareStrategy(smoothness=50.0)

        threshold = np.pi
        phase_near_threshold = threshold + 0.1

        # Both should be on low side, but high smoothness should be lower
        low_smooth = smooth_low.generate_sample(phase_near_threshold, threshold, -1, 1)
        high_smooth = smooth_high.generate_sample(phase_near_threshold, threshold, -1, 1)

        # Higher smoothness = sharper = closer to ideal value
        assert high_smooth < low_smooth


class TestComparatorSquareStrategy:
    """Test comparator square wave strategy."""

    @pytest.fixture
    def strategy(self):
        return ComparatorSquareStrategy(hysteresis=0.01)

    def test_comparator_has_state(self, strategy):
        """Comparator strategy should maintain state."""
        assert hasattr(strategy, 'last_state')
        assert strategy.last_state in [0, 1]

    def test_comparator_hysteresis_prevents_rapid_switching(self, strategy):
        """Hysteresis should prevent rapid state changes."""
        threshold = np.pi

        # Generate samples near threshold
        phases = np.linspace(threshold - 0.05, threshold + 0.05, 20)
        samples = []

        for phase in phases:
            sample = strategy.generate_sample(phase, threshold, -1, 1)
            samples.append(sample)

        # Count transitions
        transitions = np.sum(np.abs(np.diff(samples)) > 1)

        # Should have few transitions due to hysteresis
        # (ideal would switch back and forth many times)
        assert transitions <= 2  # At most one up and one down

    def test_comparator_vectorized_applies_hysteresis(self, strategy):
        """Vectorized comparator should apply hysteresis smoothing."""
        threshold = np.pi
        phases = np.linspace(0, 2*np.pi, 100)

        samples = strategy.generate_samples(phases, threshold, -1, 1)

        # Should have defined high and low regions
        assert np.max(samples) > 0
        assert np.min(samples) < 0

        # With hysteresis, there should be smooth transition regions
        # (not all samples are exactly ±1)
        unique_values = len(np.unique(samples))
        assert unique_values > 2  # More than just high and low


class TestStrategyConsistency:
    """Test consistency across all strategies."""

    @pytest.fixture(params=["ideal", "bandlimited", "soft", "comparator"])
    def strategy_name(self, request):
        return request.param

    def test_all_strategies_respect_output_range(self, strategy_name):
        """All strategies should respect custom output ranges."""
        kwargs = {}
        if strategy_name == "bandlimited":
            kwargs["sample_rate"] = 44100

        strategy = SquareWaveFactory.create(strategy_name, **kwargs)

        # Test custom range
        phases = np.linspace(0, 2*np.pi, 100)
        samples = strategy.generate_samples(phases, np.pi, 0, 10)

        # All samples should be within range
        assert np.all(samples >= 0)
        assert np.all(samples <= 10)

    def test_all_strategies_have_correct_average(self, strategy_name):
        """All strategies should have average near midpoint for 50% duty cycle."""
        kwargs = {}
        if strategy_name == "bandlimited":
            kwargs["sample_rate"] = 44100
        elif strategy_name == "soft":
            kwargs["smoothness"] = 10.0
        elif strategy_name == "comparator":
            kwargs["hysteresis"] = 0.01

        strategy = SquareWaveFactory.create(strategy_name, **kwargs)

        # Generate full period with 50% duty cycle
        phases = np.linspace(0, 2*np.pi, 1000)
        samples = strategy.generate_samples(phases, np.pi, -1, 1)

        # Average should be near 0 for 50% duty cycle
        avg = np.mean(samples)
        assert abs(avg) < 0.2  # Allow some deviation for smoothed versions


class TestPerformance:
    """Test performance characteristics."""

    def test_vectorized_faster_than_loop(self):
        """Vectorized generation should be faster than sample-by-sample."""
        import time

        strategy = IdealSquareStrategy()
        phases = np.linspace(0, 2*np.pi, 10000)
        threshold = np.pi

        # Vectorized
        start = time.perf_counter()
        for _ in range(100):
            strategy.generate_samples(phases, threshold, -1, 1)
        vectorized_time = time.perf_counter() - start

        # Sample by sample
        start = time.perf_counter()
        for _ in range(100):
            [strategy.generate_sample(p, threshold, -1, 1) for p in phases]
        loop_time = time.perf_counter() - start

        # Vectorized should be significantly faster
        assert vectorized_time < loop_time / 5  # At least 5x faster


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

