"""Compatibility re-exports for square-wave strategy classes."""

from src.engine.oscillator_square import (
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
    "IdealSquareStrategy",
    "IdealSquareStrategySmoothing",
    "SoftSquareStrategy",
    "SquareWaveFactory",
]
