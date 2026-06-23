"""Compatibility re-exports for square-wave strategy classes."""

from src.engine.oscillator_square import (
    BandlimitedSquareStrategy,
    ComparatorSquareStrategy,
    IdealSquareStrategy,
    IdealSquareStrategySmoothing,
    SoftSquareStrategy,
    SquareWaveFactory,
    SquareWaveMode,
    SquareWaveStrategy,
)

__all__ = [
    "SquareWaveMode",
    "SquareWaveStrategy",
    "BandlimitedSquareStrategy",
    "ComparatorSquareStrategy",
    "IdealSquareStrategy",
    "IdealSquareStrategySmoothing",
    "SoftSquareStrategy",
    "SquareWaveFactory",
]
