"""Loguru logging configuration for the KRM pipeline.

Configures the global ``loguru`` logger with two sinks:

* **Terminal (stderr)** — colored, human-readable, for interactive runs.
* **File** — ``logs/krm.log``, rotating (10 MB), gzip-compressed, retained
  for 14 days, for audit/history.

Call :func:`setup_logging` once at the CLI entry point. The log level can be
overridden with the ``KRM_LOG_LEVEL`` environment variable (default ``INFO``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger

#: Colored format for the terminal sink.
_TERMINAL_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

#: Plain format for the file sink (no ANSI color codes).
_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "{name}:{function}:{line} - {message}"
)


def setup_logging(
    log_dir: str | Path = "logs",
    level: str | None = None,
) -> None:
    """Configure loguru with terminal + rotating file sinks.

    Idempotent — removes any existing handlers before re-adding.

    Args:
        log_dir: Directory for the rotating log file (created if missing).
        level: Minimum log level. Defaults to ``KRM_LOG_LEVEL`` env var,
            falling back to ``"INFO"``.
    """
    level = level or os.environ.get("KRM_LOG_LEVEL", "INFO")

    logger.remove()  # drop loguru's default stderr handler

    # Terminal (stderr), colored.
    logger.add(sys.stderr, level=level, format=_TERMINAL_FORMAT, colorize=True)

    # File, rotating + compressed + retained.
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "krm.log",
        level=level,
        format=_FILE_FORMAT,
        rotation="10 MB",
        retention="14 days",
        compression="gz",
        enqueue=True,
    )
