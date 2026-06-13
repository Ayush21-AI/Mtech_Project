"""Shared utility helpers (logging, reproducibility)."""

import logging
import os
import random
from typing import Any

import numpy as np


def setup_logging(level: int = logging.INFO, fmt: str | None = None) -> logging.Logger:
    """Configure a consistent root logger for the project.

    Parameters
    ----------
    level: int
        Logging level (default ``logging.INFO``).
    fmt: str | None
        Optional custom format string.

    Returns
    -------
    logging.Logger
        The configured ``cifar10_eda`` logger.
    """
    if fmt is None:
        fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

    logger = logging.getLogger("cifar10_eda")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)

    return logger


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Sets Python ``random``, ``numpy``, and ``PYTHONHASHSEED`` so that
    downstream sampling and visualizations are deterministic.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
