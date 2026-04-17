"""Reproducibility utilities for random seed management.

Provides functions to set random seeds across PyTorch, NumPy,
and Python's random module for reproducible experiments.
"""

from __future__ import annotations


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for Python random, NumPy, and PyTorch to ensure
    reproducible results across runs.

    Args:
        seed: Random seed value.

    Raises:
        ValueError: If seed is negative.
    """
    pass
