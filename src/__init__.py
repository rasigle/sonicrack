from src.utils.logging_config import setup_logging

from ._version import version, version_info, __version__

__all__ = [
    "__version__", "version", "version_info"
]

setup_logging()
