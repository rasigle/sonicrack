"""Tests for square wave generation strategies.

This module tests all square wave formulations to ensure correctness,
performance, and proper factory behavior.
"""

import numpy as np
import pytest

from src.engine.generators.oscillators.oscillator import (
    BandlimitedSquareStrategy,
    ComparatorSquareStrategy,
    IdealSquareStrategy,
    IdealSquareStrategySmoothing,
    SoftSquareStrategy,
    SquareOscillator,
    SquareWaveFactory,
    SquareWaveStrategy,
    VCVRackSquareStrategy,
)


class TestSquareWaveFactory:
    """Test the square wave factory."""

    def test_factory_creates_ideal_strategy(self):
        """Factory should create ideal strategy."""
        strategy = SquareWaveFactory.create("ideal")
        assert isinstance(strategy, IdealSquareStrategy)

    def test_factory_creates_soft_strategy(self):
        """Factory should create soft strategy."""
        strategy = SquareWaveFactory.create("soft", smoothness=15.0)
        assert isinstance(strategy, SoftSquareStrategy)
        assert strategy.smoothness == 15.0

    def test_factory_creates_bandlimited_strategy(self):
        """Factory should create bandlimited strategy."""
        strategy = SquareWaveFactory.create(
            "bandlimited", frequency=880, sample_rate=48000
        )
        assert isinstance(strategy, BandlimitedSquareStrategy)
        assert strategy.frequency == 880
        assert strategy.sample_rate == 48000

    def test_factory_creates_vcv_strategy(self):
        """Factory should create VCV Rack-style strategy."""
        strategy = SquareWaveFactory.create("vcv", frequency=880, sample_rate=48000)
        assert isinstance(strategy, VCVRackSquareStrategy)
        assert strategy.frequency == 880
        assert strategy.sample_rate == 48000

    def test_factory_creates_comparator_strategy(self):
        """Factory should create comparator strategy."""
        strategy = SquareWaveFactory.create("comparator", hysteresis=0.05)
        assert isinstance(strategy, ComparatorSquareStrategy)
        assert strategy.hysteresis == 0.05

    def test_factory_unknown_mode_raises(self):
        """Factory should raise error for unknown mode."""
        with pytest.raises(ValueError, match="Unknown square wave mode"):
            SquareWaveFactory.create("invalid_mode")

    def test_factory_rejects_unused_ideal_strategy_kwargs(self):
        """Factory should not silently ignore misspelled ideal strategy kwargs."""
        with pytest.raises(TypeError):
            SquareWaveFactory.create("ideal", smoothness=20.0)

    def test_factory_get_available_modes(self):
        """Factory should return list of available modes."""
        modes = SquareWaveFactory.get_available_modes()
        assert isinstance(modes, list)
        assert "ideal" in modes
        assert "soft" in modes
        assert "bandlimited" in modes
        assert "vcv" in modes
        assert "comparator" in modes

    def test_factory_register_custom_strategy(self):
        """Factory should allow registering custom strategies."""

        class CustomStrategy(SquareWaveStrategy):
            def generate_sample(
                self, phase, pulsewidth_threshold, low_value, high_value
            ):
                return 0.5

            def generate_samples(
                self, phases, pulsewidth_threshold, low_value, high_value
            ):
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
        assert strategy.generate_sample(2 * np.pi - 0.01, np.pi, -1, 1) == -1  # Low

    def test_ideal_25_percent_duty_cycle(self, strategy):
        """Ideal strategy should produce 25% duty cycle at threshold=π/2."""
        threshold = np.pi / 2
        assert strategy.generate_sample(0, threshold, -1, 1) == 1  # High
        assert (
            strategy.generate_sample(threshold - 0.01, threshold, -1, 1) == 1
        )  # Still high
        assert strategy.generate_sample(threshold, threshold, -1, 1) == -1  # Low
        assert strategy.generate_sample(np.pi, threshold, -1, 1) == -1  # Low

    def test_ideal_vectorized_matches_sample_by_sample(self, strategy):
        """Vectorized and sample-by-sample should produce same results."""
        phases = np.linspace(0, 2 * np.pi, 100)
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
        high = strategy.generate_sample(threshold / 2, threshold, -1, 1)
        assert high > 0.5  # Should be closer to 1

        # Far from threshold (low side)
        low = strategy.generate_sample(2 * np.pi - 0.1, threshold, -1, 1)
        assert low < -0.5  # Should be closer to -1

    def test_soft_smooths_phase_wrap_transition(self, strategy):
        """Soft strategy should smooth the rising edge at the period wrap."""
        threshold = np.pi
        epsilon = 1e-4

        before_wrap = strategy.generate_sample(2 * np.pi - epsilon, threshold, -1, 1)
        at_wrap = strategy.generate_sample(0, threshold, -1, 1)
        after_wrap = strategy.generate_sample(epsilon, threshold, -1, 1)

        assert abs(at_wrap - before_wrap) < 0.01
        assert abs(after_wrap - at_wrap) < 0.01

    def test_soft_smoothness_parameter_effect(self):
        """Higher smoothness should create sharper transitions."""
        smooth_low = SoftSquareStrategy(smoothness=5.0)
        smooth_high = SoftSquareStrategy(smoothness=50.0)

        threshold = np.pi
        phase_near_threshold = threshold + 0.1

        # Both should be on low side, but high smoothness should be lower
        low_smooth = smooth_low.generate_sample(phase_near_threshold, threshold, -1, 1)
        high_smooth = smooth_high.generate_sample(
            phase_near_threshold, threshold, -1, 1
        )

        # Higher smoothness = sharper = closer to ideal value
        assert high_smooth < low_smooth


class TestIdealSquareStrategySmoothing:
    """Test ideal square strategy with internal amplitude smoothing."""

    def test_vectorized_smoothing_spans_multiple_chunks(self):
        """Smoothing should span the configured ramp sample count."""
        strategy = IdealSquareStrategySmoothing(smoothing_time_ms=10, sample_rate=1000)
        strategy.reset_amplitude(0.0)
        strategy.set_amplitude(1.0)

        first_chunk = strategy.generate_samples(np.zeros(5), np.pi, -1, 1)
        second_chunk = strategy.generate_samples(np.zeros(5), np.pi, -1, 1)

        assert first_chunk[-1] == pytest.approx(0.5)
        assert second_chunk[-1] == pytest.approx(1.0)
        assert strategy._smoothing_samples_remaining == 0

    def test_iterator_and_vectorized_smoothing_match(self):
        """Single-sample and vectorized generation should consume the same envelope."""
        phases = np.zeros(10)
        vectorized = IdealSquareStrategySmoothing(
            smoothing_time_ms=10, sample_rate=1000
        )
        iterator = IdealSquareStrategySmoothing(smoothing_time_ms=10, sample_rate=1000)
        vectorized.reset_amplitude(0.0)
        iterator.reset_amplitude(0.0)
        vectorized.set_amplitude(1.0)
        iterator.set_amplitude(1.0)

        vectorized_samples = vectorized.generate_samples(phases, np.pi, -1, 1)
        iterator_samples = np.array(
            [iterator.generate_sample(0, np.pi, -1, 1) for _ in range(10)],
            dtype=np.float32,
        )

        np.testing.assert_allclose(iterator_samples, vectorized_samples)

    def test_oscillator_ideal_smooth_owns_amplitude_ramp(self):
        """The oscillator should wire ideal_smooth without double-applying amplitude."""
        osc = SquareOscillator(
            frequency=10,
            amplitude=1.0,
            gain_db=None,
            sample_rate=1000,
            mode="ideal_smooth",
            smoothing_time_ms=10,
        )

        osc.amplitude = 0.5
        samples = osc.get_samples_vectorized(10)

        assert osc._smoothing_samples_remaining == 0
        assert abs(samples[-1]) == pytest.approx(0.5)

    def test_oscillator_ideal_smooth_tracks_sample_rate_changes(self):
        """The oscillator should keep ideal_smooth sample timing in sync."""
        osc = SquareOscillator(
            frequency=10,
            amplitude=1.0,
            gain_db=None,
            sample_rate=1000,
            mode="ideal_smooth",
            smoothing_time_ms=10,
        )
        osc.sample_rate = 2000

        osc.amplitude = 0.5
        osc.get_samples_vectorized(10)

        assert osc._strategy.sample_rate == 2000
        assert osc._strategy._smoothing_samples_remaining == 10

    def test_square_iterator_smooths_amplitude_changes(self):
        """Iterator mode should ramp amplitude changes instead of jumping to target."""
        iterator_osc = SquareOscillator(
            frequency=10, amplitude=1.0, gain_db=None, sample_rate=1000
        )
        vectorized_osc = SquareOscillator(
            frequency=10, amplitude=1.0, gain_db=None, sample_rate=1000
        )

        iterator_osc.amplitude = 0.5
        vectorized_osc.amplitude = 0.5

        iterator_samples = np.array([next(iterator_osc) for _ in range(10)])
        vectorized_samples = vectorized_osc.get_samples_vectorized(10)

        np.testing.assert_allclose(iterator_samples, vectorized_samples, rtol=1e-6)
        assert abs(iterator_samples[0]) < 1.0


class TestBandlimitedSquareStrategy:
    """Test PolyBLEP bandlimited square wave strategy."""

    def test_bandlimited_vectorized_matches_sample_by_sample(self):
        """Vectorized and sample-by-sample output should match."""
        phases = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        vectorized = BandlimitedSquareStrategy(frequency=440, sample_rate=44100)
        iterator = BandlimitedSquareStrategy(frequency=440, sample_rate=44100)

        vectorized_samples = vectorized.generate_samples(phases, np.pi, -1, 1)
        iterator_samples = np.array(
            [iterator.generate_sample(p, np.pi, -1, 1) for p in phases],
            dtype=np.float32,
        )

        np.testing.assert_allclose(iterator_samples, vectorized_samples)

    def test_bandlimited_respects_custom_output_range(self):
        """Bandlimited strategy should scale generated values to the requested range."""
        strategy = BandlimitedSquareStrategy(frequency=440, sample_rate=44100)
        phases = np.linspace(0.1, np.pi - 0.1, 20)
        samples = strategy.generate_samples(phases, np.pi, 0, 10)

        assert np.all(samples >= 0)
        assert np.all(samples <= 10)
        assert np.max(samples) > 9

    def test_oscillator_syncs_bandlimited_frequency_and_sample_rate(self):
        """SquareOscillator should keep the bandlimited strategy context current."""
        osc = SquareOscillator(
            frequency=440,
            sample_rate=44100,
            mode="bandlimited",
            gain_db=0,
        )

        osc.frequency = 880
        osc.sample_rate = 48000

        assert osc._strategy.frequency == 880
        assert osc._strategy.sample_rate == 48000

    def test_bandlimited_reduces_high_frequency_content(self):
        """Bandlimited mode should reduce near-Nyquist energy versus ideal mode."""
        n_samples = 4096
        sample_rate = 44100
        frequency = 4000
        ideal = SquareOscillator(
            frequency=frequency, sample_rate=sample_rate, mode="ideal", gain_db=0
        ).get_samples_vectorized(n_samples)
        bandlimited = SquareOscillator(
            frequency=frequency,
            sample_rate=sample_rate,
            mode="bandlimited",
            gain_db=0,
        ).get_samples_vectorized(n_samples)

        ideal_fft = np.abs(np.fft.rfft(ideal))
        bandlimited_fft = np.abs(np.fft.rfft(bandlimited))
        freqs = np.fft.rfftfreq(n_samples, d=1 / sample_rate)
        high_band = freqs > sample_rate * 0.35

        assert np.mean(bandlimited_fft[high_band]) < np.mean(ideal_fft[high_band])


class TestVCVRackSquareStrategy:
    """Test VCV Rack Fundamental-style square wave strategy."""

    def test_vcv_vectorized_matches_sample_by_sample(self):
        """Stateful VCV mode should produce identical iterator/vectorized samples."""
        iterator_osc = SquareOscillator(
            frequency=440,
            sample_rate=44100,
            mode="vcv",
            gain_db=0,
        )
        vectorized_osc = SquareOscillator(
            frequency=440,
            sample_rate=44100,
            mode="vcv",
            gain_db=0,
        )

        iterator_samples = np.array([next(iterator_osc) for _ in range(2048)])
        vectorized_samples = vectorized_osc.get_samples_vectorized(2048)

        np.testing.assert_allclose(iterator_samples, vectorized_samples, atol=1e-6)

    def test_vcv_has_minblep_ringing(self):
        """VCV mode should show minBLEP edge ringing rather than hard clipping."""
        osc = SquareOscillator(
            frequency=440,
            sample_rate=44100,
            mode="vcv",
            gain_db=0,
            dc_block=False,
        )
        samples = osc.get_samples_vectorized(2048)

        assert np.max(samples) > 1.05
        assert np.min(samples) < -1.05

    def test_vcv_reverse_phase_edges_are_symmetric(self):
        """Reverse phase traversal should sign the minBLEP edges symmetrically."""
        osc = SquareOscillator(
            frequency=-10,
            sample_rate=1000,
            phase=90,
            mode="vcv",
            gain_db=0,
            dc_block=False,
        )
        samples = osc.get_samples_vectorized(250)
        edge_indices = np.flatnonzero(
            np.signbit(samples[:-1]) != np.signbit(samples[1:])
        )

        falling_edge = edge_indices[2]
        rising_edge = edge_indices[3]
        falling_window = samples[falling_edge - 4 : falling_edge + 24]
        rising_window = samples[rising_edge - 4 : rising_edge + 24]

        np.testing.assert_allclose(falling_window, -rising_window, atol=1e-6)

    def test_vcv_clamps_pulse_width_like_vcv_rack(self):
        """VCV mode should internally clamp pulse width to 1%-99%."""
        for pulsewidth, expected in (
            (0.0, 0.01),
            (1.0, 0.99),
        ):
            osc = SquareOscillator(
                frequency=10,
                sample_rate=10000,
                mode="vcv",
                pulsewidth=pulsewidth,
                gain_db=0,
                dc_block=False,
            )
            samples = osc.get_samples_vectorized(10000)

            assert np.mean(samples > 0) == pytest.approx(expected, abs=0.002)

    def test_vcv_dc_block_reduces_long_term_bias(self):
        """VCV mode should apply the VCV-style high-pass/DC-blocking filter."""
        blocked = SquareOscillator(
            frequency=20,
            sample_rate=44100,
            mode="vcv",
            pulsewidth=0.8,
            gain_db=0,
        ).get_samples_vectorized(44100)
        unblocked = SquareOscillator(
            frequency=20,
            sample_rate=44100,
            mode="vcv",
            pulsewidth=0.8,
            gain_db=0,
            dc_block=False,
        ).get_samples_vectorized(44100)

        assert abs(np.mean(blocked)) < abs(np.mean(unblocked))

    def test_vcv_reduces_high_frequency_content(self):
        """VCV minBLEP mode should reduce near-Nyquist energy versus ideal mode."""
        n_samples = 4096
        sample_rate = 44100
        frequency = 4000
        ideal = SquareOscillator(
            frequency=frequency, sample_rate=sample_rate, mode="ideal", gain_db=0
        ).get_samples_vectorized(n_samples)
        vcv = SquareOscillator(
            frequency=frequency, sample_rate=sample_rate, mode="vcv", gain_db=0
        ).get_samples_vectorized(n_samples)

        ideal_fft = np.abs(np.fft.rfft(ideal))
        vcv_fft = np.abs(np.fft.rfft(vcv))
        freqs = np.fft.rfftfreq(n_samples, d=1 / sample_rate)
        high_band = freqs > sample_rate * 0.35

        assert np.mean(vcv_fft[high_band]) < np.mean(ideal_fft[high_band])

    def test_vcv_reset_restores_state(self):
        """Resetting oscillator iteration should clear VCV buffers and filters."""
        osc = SquareOscillator(
            frequency=440,
            sample_rate=44100,
            mode="vcv",
            gain_db=0,
        )
        first = osc.get_samples_vectorized(512)
        osc.get_samples_vectorized(2048)
        second = osc.get_samples(512, reset=True, mode="vectorized")

        np.testing.assert_allclose(first, second)


class TestComparatorSquareStrategy:
    """Test sine-comparator square wave strategy."""

    def test_comparator_matches_ideal_without_hysteresis(self):
        """Comparator without hysteresis should follow the requested duty cycle."""
        strategy = ComparatorSquareStrategy(hysteresis=0.0)
        phases = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
        samples = strategy.generate_samples(phases, np.pi, -1, 1)

        high_ratio = np.mean(samples > 0)
        assert high_ratio == pytest.approx(0.5, abs=0.01)

    def test_comparator_supports_pulsewidth(self):
        """Comparator mode should honor non-50% pulse widths."""
        osc = SquareOscillator(
            frequency=10,
            pulsewidth=0.25,
            sample_rate=10000,
            mode="comparator",
            gain_db=0,
        )
        samples = osc.get_samples_vectorized(10000)

        assert np.mean(samples > 0) == pytest.approx(0.25, abs=0.02)

    def test_comparator_iterator_and_vectorized_match(self):
        """Stateful comparator generation should be mode-consistent."""
        iterator_osc = SquareOscillator(
            frequency=440,
            sample_rate=44100,
            mode="comparator",
            hysteresis=0.02,
            gain_db=0,
        )
        vectorized_osc = SquareOscillator(
            frequency=440,
            sample_rate=44100,
            mode="comparator",
            hysteresis=0.02,
            gain_db=0,
        )

        iterator_samples = np.array([next(iterator_osc) for _ in range(1000)])
        vectorized_samples = vectorized_osc.get_samples_vectorized(1000)

        np.testing.assert_allclose(iterator_samples, vectorized_samples)

    def test_comparator_reset_restores_state(self):
        """Resetting oscillator iteration should reset comparator hysteresis state."""
        osc = SquareOscillator(
            frequency=440,
            sample_rate=44100,
            mode="comparator",
            hysteresis=0.05,
            gain_db=0,
        )
        first = osc.get_samples_vectorized(128)
        osc.get_samples_vectorized(256)
        second = osc.get_samples(128, reset=True, mode="vectorized")

        np.testing.assert_allclose(first, second)


class TestStrategyConsistency:
    """Test consistency across all strategies."""

    @pytest.fixture(params=SquareOscillator.get_available_modes())
    def strategy_name(self, request):
        return request.param

    def test_all_strategies_respect_output_range(self, strategy_name):
        """All strategies should respect custom output ranges."""
        if strategy_name == "vcv":
            # "VCV-style minBLEP ringing intentionally overshoots edges"
            return

        kwargs = {}
        if strategy_name == "bandlimited":
            kwargs["sample_rate"] = 44100

        strategy = SquareWaveFactory.create(strategy_name, **kwargs)

        # Test custom range
        phases = np.linspace(0, 2 * np.pi, 100)
        samples = strategy.generate_samples(phases, np.pi, 0, 10)

        # All samples should be within range
        assert np.all(samples >= 0)
        assert np.all(samples <= 10)

    def test_all_strategies_have_correct_average(self, strategy_name):
        """All strategies should have average near midpoint for 50% duty cycle."""
        if strategy_name == "vcv":
            # "VCV-style strategy needs a longer buffer for DC settling"
            return

        kwargs = {}
        if strategy_name == "soft":
            kwargs["smoothness"] = 10.0

        strategy = SquareWaveFactory.create(strategy_name, **kwargs)

        # Generate full period with 50% duty cycle
        phases = np.linspace(0, 2 * np.pi, 1000)
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
        phases = np.linspace(0, 2 * np.pi, 10000)
        threshold = np.pi

        # Vectorized
        start = time.perf_counter()
        for _ in range(100):
            strategy.generate_samples(phases, threshold, -1, 1)
        vectorized_time = time.perf_counter() - start

        # Sample by sample
        start = time.perf_counter()
        for _ in range(100):
            _ = [strategy.generate_sample(p, threshold, -1, 1) for p in phases]
        loop_time = time.perf_counter() - start

        # Vectorized should be significantly faster
        assert vectorized_time < loop_time / 5  # At least 5x faster


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
