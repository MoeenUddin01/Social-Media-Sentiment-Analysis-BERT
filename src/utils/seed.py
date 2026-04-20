"""Reproducibility utilities for random seed management.

Provides functions to set random seeds across PyTorch, NumPy,
and Python's random module for reproducible experiments.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for Python random, NumPy, and PyTorch to ensure
    reproducible results across runs.

    Args:
        seed: Random seed value.

    Raises:
        ValueError: If seed is negative.
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Seed set to {seed}")


def seed_worker(worker_id: int) -> None:
    """Seed each DataLoader worker for reproducibility.

    Args:
        worker_id: Worker ID from DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device() -> torch.device:
    """Returns the best available device.

    Checks cuda → mps → cpu in order.
    Logs which device was selected.

    Returns:
        torch.device: The best available device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using device: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("✅ Using device: CPU")
    return device


__all__ = ["set_seed", "seed_worker", "get_device"]
