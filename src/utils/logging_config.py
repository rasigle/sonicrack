"""Logging configuration for the AudioPlayground engine.

This module provides centralized logging configuration for all engine components.
Use get_logger() to obtain a configured logger for your module.

Example:
    >>> from src.utils.logging_config import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Oscillator initialized with frequency 440 Hz")
"""

import logging
import sys
from pathlib import Path

from src.constants import LOG_DIRECTORY

# Default log level
DEFAULT_LOG_LEVEL = logging.INFO

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)

def setup_logging(
    level: int = DEFAULT_LOG_LEVEL,
    log_file: str | None = None,
    console_output: bool = True,
    detailed: bool = False,
) -> None:
    """Configure logging for the entire application.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO).
        log_file: Optional path to log file. If None, no file logging.
        console_output: If True, log to console (stderr).
        detailed: If True, use detailed format with file/line info.

    Example:
        >>> setup_logging(level=logging.DEBUG, log_file="audio_engine.log")
    """
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Set root logger to DEBUG if file logging is enabled (allows file to capture all
    # levels). Otherwise set to the requested level
    root_logger.setLevel(logging.DEBUG if log_file else level)

    # Choose format
    log_format = DETAILED_FORMAT if detailed else LOG_FORMAT
    formatter = logging.Formatter(log_format)

    # Console handler - filters to requested level
    if console_output:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler - always captures DEBUG and above
    if log_file:
        if not Path(log_file).is_absolute():
            file_path = LOG_DIRECTORY / log_file
            LOG_DIRECTORY.mkdir(exist_ok=True)
        else:
            file_path = Path(log_file)
        file_handler = logging.FileHandler(file_path, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # File gets all DEBUG messages
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logging.debug(
        f"Set up logging: level={level}, log_file={log_file}, "
        f"console_output={console_output}, detailed={detailed}"
    )


def get_logger(name: str, level: int | None = None) -> logging.Logger:
    """Get a logger instance for a specific module.

    Args:
        name: Logger name, typically __name__ from the calling module.
        level: Optional logging level. If None, uses root logger level.

    Returns:
        logging.Logger: Configured logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.debug("Debug message")
        >>> logger.info("Info message")
        >>> logger.warning("Warning message")
        >>> logger.error("Error message")
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger


# Convenience function for engine modules
def get_engine_logger(module_name: str) -> logging.Logger:
    """Get a logger specifically for engine modules.

    Args:
        module_name: Name of the engine module.

    Returns:
        logging.Logger: Configured logger with 'engine.' prefix.

    Example:
        >>> # In src/engine/oscillator.py
        >>> logger = get_engine_logger("oscillator")
        >>> logger.info("Oscillator initialized")
    """
    return get_logger(f"engine.{module_name}")


if __name__ == "__main__":
    # Example usage
    setup_logging(level=logging.DEBUG, log_file="test.log", detailed=True)

    logger = get_logger(__name__)
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    engine_logger = get_engine_logger("test_module")
    engine_logger.info("Engine module message")
