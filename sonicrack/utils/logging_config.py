"""Logging configuration for the SonicRack engine.

Example:
    >>> import logging
    >>>
    >>> logger = logging.getLogger(__name__)
    >>> logger.info("Oscillator initialized with frequency 440 Hz")
"""

import logging
import sys
from pathlib import Path

from sonicrack.constants import LOG_DIRECTORY

# Default log level
DEFAULT_LOG_LEVEL = logging.INFO

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)


def setup_logging(
    level: int = DEFAULT_LOG_LEVEL,
    log_file: str | Path | None = None,
    console_output: bool = True,
    detailed: bool = False,
    file_level: int = logging.DEBUG,
    file_mode: str = "w",
) -> None:
    """Configure logging for the entire application.

    Args:
        level: Console logging level, e.g. logging.DEBUG or logging.INFO.
        log_file: Optional path to log file. Relative paths are placed in LOG_DIRECTORY.
        console_output: If True, log to stderr.
        detailed: If True, include file and line information.
        file_level: Logging level for file output.
        file_mode: File open mode. Use "w" to overwrite or "a" to append.
    """
    root_logger = logging.getLogger()

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    root_logger.setLevel(min(level, file_level) if log_file else level)

    log_format = DETAILED_FORMAT if detailed else LOG_FORMAT
    formatter = logging.Formatter(log_format)

    if console_output:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)

        if not log_path.is_absolute():
            log_path = LOG_DIRECTORY / log_path

        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            log_path,
            mode=file_mode,
            encoding="utf-8",
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logging.debug(
        "Set up logging: level=%s, log_file=%s, console_output=%s, detailed=%s",
        level,
        log_file,
        console_output,
        detailed,
    )


if __name__ == "__main__":
    # Example usage
    setup_logging(level=logging.DEBUG, log_file="test.log", detailed=True)

    logger = logging.getLogger(__name__)
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
