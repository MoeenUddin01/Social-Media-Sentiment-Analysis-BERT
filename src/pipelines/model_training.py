"""Training pipeline for BERT sentiment classification.

Provides a structured training loop with evaluation callbacks,
early stopping, and model checkpointing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader

from src.models.evaluator import ModelEvaluator

if TYPE_CHECKING:
    from src.models.bert_classifier import BertSentimentClassifier


def train_epoch(
    model: BertSentimentClassifier,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    evaluator: ModelEvaluator | None = None,
    val_loader: DataLoader | None = None,
) -> dict[str, float]:
    """Train for one epoch with optional validation and callbacks.

    Args:
        model: The BERT classifier model.
        train_loader: DataLoader for training data.
        optimizer: Optimizer for parameter updates.
        device: Device to run training on.
        evaluator: Optional ModelEvaluator for validation metrics.
        val_loader: Optional validation DataLoader for evaluation.

    Returns:
        Dictionary with training metrics and optional validation metrics.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    criterion = torch.nn.CrossEntropyLoss()

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        logits = model(inputs)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    metrics: dict[str, float] = {
        "train_loss": avg_loss,
        "train_accuracy": accuracy,
    }

    # Validation step with evaluator
    if evaluator is not None and val_loader is not None:
        val_metrics = evaluator.evaluate(val_loader)
        metrics["val_loss"] = val_metrics["loss"]
        metrics["val_accuracy"] = val_metrics["accuracy"]
        metrics["val_macro_f1"] = val_metrics["macro_f1"]
        metrics["val_weighted_f1"] = val_metrics["weighted_f1"]

        # Pass val_metrics dict to callbacks (EarlyStopping and ModelCheckpoint)
        # These callbacks can use metrics for model selection
        _run_callbacks(metrics, val_metrics)

    return metrics


def _run_callbacks(
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
) -> None:
    """Run training callbacks with validation metrics.

    Args:
        train_metrics: Dictionary of training metrics.
        val_metrics: Dictionary of validation metrics from evaluator.
    """
    # Callback implementations (EarlyStopping, ModelCheckpoint) can access
    # val_metrics dict which includes:
    # - accuracy, macro_f1, weighted_f1, precision, recall, confusion_matrix
    pass
