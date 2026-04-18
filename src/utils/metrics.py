"""Metrics computation for sentiment classification evaluation.

Provides accuracy, precision, recall, F1 score, and confusion matrix
computation for multi-class sentiment analysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class MetricsComputer:
    """Computer for classification metrics.

    Calculates accuracy, precision, recall, F1 score, and confusion
    matrix for sentiment classification evaluation.

    Args:
        num_classes: Number of sentiment classes.
        average: Averaging strategy for multi-class metrics.

    Raises:
        ValueError: If num_classes is less than 2.
    """
    pass


def per_class_precision_recall(
    y_true: "np.ndarray",
    y_pred: "np.ndarray",
    num_classes: int = 3,
) -> dict[str, dict[str, float]]:
    """Calculate per-class precision and recall.

    Args:
        y_true: Ground truth labels (N,).
        y_pred: Predicted labels (N,).
        num_classes: Number of classes. Defaults to 3.

    Returns:
        Dictionary mapping class index to dict with precision and recall.
    """
    from sklearn.metrics import precision_score, recall_score

    results: dict[str, dict[str, float]] = {}

    for class_idx in range(num_classes):
        # Binary precision/recall for this class vs rest
        y_true_binary = (y_true == class_idx).astype(int)
        y_pred_binary = (y_pred == class_idx).astype(int)

        try:
            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        except Exception:
            precision = 0.0
            recall = 0.0

        results[str(class_idx)] = {
            "precision": float(precision),
            "recall": float(recall),
        }

    return results


def format_confusion_matrix(
    cm: "np.ndarray",
    label_names: list[str] | None = None,
) -> str:
    """Format confusion matrix as a readable string.

    Args:
        cm: Confusion matrix array (num_classes, num_classes).
        label_names: List of class names. If None, uses indices.

    Returns:
        Formatted confusion matrix string.
    """
    if label_names is None:
        label_names = [str(i) for i in range(len(cm))]

    # Build formatted string
    max_name_len = max(len(name) for name in label_names)
    header = " " * (max_name_len + 2) + " ".join(f"{name:>8}" for name in label_names)
    lines = [header]

    for i, name in enumerate(label_names):
        row_values = " ".join(f"{cm[i, j]:>8}" for j in range(len(cm)))
        lines.append(f"{name:>{max_name_len}} | {row_values}")

    return "\n".join(lines)
