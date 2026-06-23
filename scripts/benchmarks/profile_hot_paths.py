"""Profile the current measured DSP hot paths.

Run from the repository root:

    uv run python scripts/benchmarks/profile_hot_paths.py
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
from collections.abc import Callable

import numpy as np

from src.engine import Reverb, SineOscillator, SquareOscillator

ProfileTarget = Callable[[], np.ndarray]


def _targets(buffer_size: int) -> dict[str, ProfileTarget]:
    reverb = Reverb(
        SineOscillator(frequency=330, gain_db=-12),
        room_size=0.65,
        damping=0.35,
        mix=0.4,
    )
    square = SquareOscillator(frequency=137, gain_db=-12, mode="vcv")
    return {
        "reverb": lambda: reverb.get_samples_vectorized(buffer_size),
        "square_vcv": lambda: square.get_samples_vectorized(buffer_size),
    }


def profile_target(target: ProfileTarget, iterations: int, top: int) -> str:
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(iterations):
        target()
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumtime")
    stats.print_stats(top)
    return stream.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer-size", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    for name, target in _targets(args.buffer_size).items():
        print(f"== {name} ==")
        print(profile_target(target, args.iterations, args.top))


if __name__ == "__main__":
    main()
