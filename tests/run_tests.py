"""Run all unit tests for the audio engine.

This script discovers and runs all tests in the tests directory.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py -v           # Verbose output
    python run_tests.py TestSineOscillator  # Run specific test class
"""

import sys
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_all_tests(
    verbosity: int = 2, pattern: str = "test_*.py"
) -> unittest.TestResult:
    """Discover and run all tests.

    Args:
        verbosity: Verbosity level (0=quiet, 1=normal, 2=verbose).
        pattern: Pattern to match test files.

    Returns:
        unittest.TestResult: Test results.
    """
    # Discover tests
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(start_dir, pattern=pattern)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    return result


def main() -> int:
    """Main entry point.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    # Parse simple command-line arguments
    verbosity = 2
    pattern = "test_*.py"

    if "-v" in sys.argv or "--verbose" in sys.argv:
        verbosity = 2
    if "-q" in sys.argv or "--quiet" in sys.argv:
        verbosity = 0

    # Check for specific test pattern
    for arg in sys.argv[1:]:
        if arg.startswith("Test") or arg.startswith("test"):
            pattern = f"*{arg}*.py"

    print("=" * 70)
    print("Running Audio Engine Unit Tests")
    print("=" * 70)
    print()

    result = run_all_tests(verbosity=verbosity, pattern=pattern)

    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print()

    if result.wasSuccessful():
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
