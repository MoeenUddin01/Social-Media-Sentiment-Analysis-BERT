"""Training callbacks for BERT fine-tuning.

Provides early stopping, model checkpointing, and logging
callbacks for the training process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class EarlyStopping:
    """Early stopping callback to halt training when validation metrics stop improving.

    Monitors a specified metric and stops training if it doesn't improve
    for a given number of epochs (patience).

    Attributes:
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum change to qualify as improvement.
        mode: 'min' or 'max' for the monitored metric.
        best_metric: Best value seen so far.
        counter: Number of epochs without improvement.
        should_stop: Flag indicating training should stop.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
        monitor: str = "val_loss",
    ) -> None:
        """Initialize EarlyStopping callback.

        Args:
            patience: Number of epochs to wait for improvement before stopping.
            min_delta: Minimum change to count as improvement.
            mode: 'min' to minimize metric, 'max' to maximize.
            monitor: Metric name to monitor (e.g., 'val_loss', 'val_accuracy').

        Raises:
            ValueError: If mode is not 'min' or 'max'.
        """
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.monitor = monitor
        self.best_metric: float | None = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, epoch: int, metrics: dict) -> bool:
        """Check if training should stop.

        Args:
            epoch: Current epoch number (0-indexed).
            metrics: Dictionary of metric values.

        Returns:
            True if training should stop, False otherwise.

        Raises:
            KeyError: If monitored metric is not in metrics dict.
        """
        if self.monitor not in metrics:
            raise KeyError(f"Monitored metric '{self.monitor}' not found in metrics")

        current = float(metrics[self.monitor])

        if self.best_metric is None:
            self.best_metric = current
            return False

        if self.mode == "min":
            improved = current < (self.best_metric - self.min_delta)
        else:
            improved = current > (self.best_metric + self.min_delta)

        if improved:
            self.best_metric = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class ModelCheckpoint:
    """Model checkpointing callback to save model during training.

    Saves model checkpoints based on improvement of a monitored metric.
    Can save best model only or every epoch.

    Attributes:
        monitor: Metric name to monitor for saving.
        mode: 'min' or 'max' for the monitored metric.
        save_best_only: Whether to only save improved models.
        save_dir: Directory to save checkpoints.
        best_metric: Best value seen so far.
        should_save: Flag indicating model should be saved this epoch.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        save_dir: str | Path = "artifacts/checkpoints",
    ) -> None:
        """Initialize ModelCheckpoint callback.

        Args:
            monitor: Metric name to monitor (e.g., 'val_loss', 'val_accuracy').
            mode: 'min' to minimize metric, 'max' to maximize.
            save_best_only: If True, only save when metric improves.
            save_dir: Directory path to save checkpoints.

        Raises:
            ValueError: If mode is not 'min' or 'max'.
        """
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_dir = save_dir
        self.best_metric: float | None = None
        self.should_save = False

    def __call__(self, epoch: int, metrics: dict) -> bool:
        """Determine if checkpoint should be saved.

        Args:
            epoch: Current epoch number (0-indexed).
            metrics: Dictionary of metric values.

        Returns:
            True if model should be saved this epoch, False otherwise.

        Raises:
            KeyError: If monitored metric is not in metrics dict.
        """
        if self.monitor not in metrics:
            raise KeyError(f"Monitored metric '{self.monitor}' not found in metrics")

        current = float(metrics[self.monitor])

        if self.best_metric is None:
            self.best_metric = current
            self.should_save = True
            return True

        if self.mode == "min":
            improved = current < (self.best_metric - 1e-7)
        else:
            improved = current > (self.best_metric + 1e-7)

        if improved:
            self.best_metric = current
            self.should_save = True
            return True

        if not self.save_best_only:
            self.should_save = True
            return True

        self.should_save = False
        return False
