#!/usr/bin/env python
"""Click detection tool for audio quality testing.

This tool analyzes audio output to detect clicks, pops, and discontinuities
that indicate buffer underruns or parameter changes without proper smoothing.

Usage:
    python scripts/benchmarks/test_no_clicks.py <audio_file.wav>
    python scripts/benchmarks/test_no_clicks.py --live <patch_file.apr>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def detect_clicks(
    audio: np.ndarray,
    threshold: float = 0.5,
    sample_rate: int = 44100,
) -> dict[str, any]:
    """Detect clicks in audio signal.

    Args:
        audio: Audio samples as numpy array
        threshold: Amplitude threshold for click detection
        sample_rate: Sample rate in Hz

    Returns:
        Dictionary containing click detection results
    """
    # Calculate sample-to-sample differences
    diff = np.abs(np.diff(audio))

    # Find discontinuities above threshold
    click_indices = np.where(diff > threshold)[0]

    # Calculate click severity
    if len(click_indices) > 0:
        click_amplitudes = diff[click_indices]
        max_amplitude = np.max(click_amplitudes)
        mean_amplitude = np.mean(click_amplitudes)
    else:
        max_amplitude = 0.0
        mean_amplitude = 0.0

    # Calculate time positions of clicks
    click_times = click_indices / sample_rate if len(click_indices) > 0 else []

    # Analyze audio statistics
    audio_rms = np.sqrt(np.mean(audio**2))
    audio_peak = np.max(np.abs(audio))

    return {
        "click_count": len(click_indices),
        "click_indices": click_indices.tolist(),
        "click_times": click_times.tolist() if len(click_times) > 0 else [],
        "max_click_amplitude": float(max_amplitude),
        "mean_click_amplitude": float(mean_amplitude),
        "threshold": threshold,
        "audio_rms": float(audio_rms),
        "audio_peak": float(audio_peak),
        "sample_count": len(audio),
        "duration_sec": len(audio) / sample_rate,
    }


def print_click_report(results: dict[str, any]) -> None:
    """Print click detection results in a readable format."""
    print(f"\n{'=' * 70}")
    print(f"CLICK DETECTION REPORT")
    print(f"{'=' * 70}")
    print(f"Audio Duration:    {results['duration_sec']:.2f} seconds")
    print(f"Sample Count:      {results['sample_count']:,}")
    print(f"")
    print(f"Audio Statistics:")
    print(f"  RMS Level:       {results['audio_rms']:.4f}")
    print(f"  Peak Level:      {results['audio_peak']:.4f}")
    print(f"")
    print(f"Click Detection:")
    print(f"  Threshold:       {results['threshold']:.2f}")
    print(f"  Clicks Found:    {results['click_count']}")

    if results["click_count"] > 0:
        print(f"  Max Amplitude:   {results['max_click_amplitude']:.4f}")
        print(f"  Mean Amplitude:  {results['mean_click_amplitude']:.4f}")
        print(f"")
        print(f"Click Positions (first 10):")
        for i, (idx, time) in enumerate(
            zip(results["click_indices"][:10], results["click_times"][:10])
        ):
            print(f"    {i+1}. Sample {idx:,} @ {time:.3f}s")
        if results["click_count"] > 10:
            print(f"    ... and {results['click_count'] - 10} more")

    # Status indicator
    if results["click_count"] == 0:
        status = "✅ PASS - No clicks detected"
    elif results["click_count"] < 5:
        status = "⚠️  WARNING - Few clicks detected"
    else:
        status = "❌ FAIL - Many clicks detected"

    print(f"")
    print(f"Status: {status}")
    print(f"{'=' * 70}\n")


def generate_test_audio(
    duration_sec: float = 10.0,
    sample_rate: int = 44100,
) -> np.ndarray:
    """Generate test audio with known characteristics.

    Args:
        duration_sec: Duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        Test audio samples
    """
    num_samples = int(duration_sec * sample_rate)

    # Generate a mix of frequencies
    t = np.arange(num_samples) / sample_rate
    audio = (
        0.3 * np.sin(2 * np.pi * 440 * t)  # A4
        + 0.2 * np.sin(2 * np.pi * 880 * t)  # A5
        + 0.1 * np.sin(2 * np.pi * 220 * t)  # A3
    )

    return audio.astype(np.float32)


def generate_click_audio(
    duration_sec: float = 10.0,
    sample_rate: int = 44100,
    num_clicks: int = 5,
) -> np.ndarray:
    """Generate test audio with intentional clicks.

    Args:
        duration_sec: Duration in seconds
        sample_rate: Sample rate in Hz
        num_clicks: Number of clicks to inject

    Returns:
        Test audio with clicks
    """
    audio = generate_test_audio(duration_sec, sample_rate)

    # Add some clicks at random positions
    num_samples = len(audio)
    click_positions = np.random.choice(
        range(1000, num_samples - 1000), size=num_clicks, replace=False
    )

    for pos in click_positions:
        # Create discontinuity
        audio[pos] = audio[pos - 1] + 0.7 * np.sign(np.random.randn())

    return audio


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Detect clicks and discontinuities in audio"
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        type=Path,
        help="Path to audio file to analyze (.wav, .npy)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Click detection threshold (default: 0.5)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Sample rate in Hz (default: 44100)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run self-test with generated audio",
    )
    parser.add_argument(
        "--test-with-clicks",
        action="store_true",
        help="Run self-test with intentional clicks",
    )

    args = parser.parse_args()

    if args.test:
        # Test with clean audio
        print("Testing with clean audio...")
        audio = generate_test_audio(duration_sec=5.0, sample_rate=args.sample_rate)
        results = detect_clicks(audio, args.threshold, args.sample_rate)
        print_click_report(results)
        return 0 if results["click_count"] == 0 else 1

    elif args.test_with_clicks:
        # Test with audio containing clicks
        print("Testing with audio containing 5 intentional clicks...")
        audio = generate_click_audio(
            duration_sec=5.0, sample_rate=args.sample_rate, num_clicks=5
        )
        results = detect_clicks(audio, args.threshold, args.sample_rate)
        print_click_report(results)
        return 0 if results["click_count"] >= 5 else 1

    elif args.audio_file:
        # Analyze audio file
        if not args.audio_file.exists():
            print(f"Error: Audio file not found: {args.audio_file}")
            return 1

        # Load audio based on file extension
        if args.audio_file.suffix == ".npy":
            audio = np.load(args.audio_file)
        elif args.audio_file.suffix in [".wav", ".wave"]:
            # TODO: Add scipy.io.wavfile support when available
            print(f"Error: WAV file support not yet implemented")
            print(f"       Use .npy files or add scipy dependency")
            return 1
        else:
            print(f"Error: Unsupported file format: {args.audio_file.suffix}")
            print(f"       Supported: .npy (numpy array)")
            return 1

        results = detect_clicks(audio, args.threshold, args.sample_rate)
        print_click_report(results)
        return 0 if results["click_count"] == 0 else 1

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
