"""Logging helpers shared by the CLI and the experiment runners."""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def configure_logging(verbose: bool = False) -> None:
    """Install a single stream handler on the root logger.

    Calling this more than once replaces the previous handler instead of
    stacking duplicates, which keeps repeated CLI invocations inside one
    Python process readable.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Qiskit is chatty at DEBUG level and drowns out our own progress lines.
    logging.getLogger("qiskit").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a package-scoped logger."""
    return logging.getLogger(name)
