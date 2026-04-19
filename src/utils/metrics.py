"""Metrics computation for sentiment classification evaluation.

Provides accuracy, precision, recall, F1 score, and confusion matrix
computation for multi-class sentiment analysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class SentimentMetrics:
    """Computer for sentiment classification metrics.

    Calculates accuracy, precision, recall, F1 score, and confusion
    matrix for multi-class sentiment analysis.

    All methods are static and operate on numpy arrays of predictions
    and labels.
    """

    @staticmethod
    def compute_all(
        y_true: "np.ndarray",
        y_pred: "np.ndarray",
        y_probs: "np.ndarray" | None = None,
    ) -> dict[str, float]:
        """Compute all classification metrics.

        Args:
            y_true: Ground truth labels (N,).
            y_pred: Predicted labels (N,).
            y_probs: Optional probability predictions (N, num_classes).

        Returns:
            Dictionary containing:
                - accuracy: Overall accuracy
                - macro_f1: Macro-averaged F1 score
                - weighted_f1: Weighted-averaged F1 score
                - precision: Macro-averaged precision
                - recall: Macro-averaged recall
                - per_class_precision: Dict of per-class precision scores
                - per_class_recall: Dict of per-class recall scores
                - per_class_f1: Dict of per-class F1 scores
                - confusion_matrix: Confusion matrix as numpy array
        """
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )

        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        weighted_f1 = f1_score(y_true, y_pred, average="weighted")
        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")
        cm = confusion_matrix(y_true, y_pred)

        # Per-class metrics
        per_class_f1 = f1_score(y_true, y_pred, average=None)
        per_class_precision = precision_score(y_true, y_pred, average=None)
        per_class_recall = recall_score(y_true, y_pred, average=None)

        num_classes = len(per_class_f1)

        return {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "precision": float(precision),
            "recall": float(recall),
            "per_class_precision": {
                str(i): float(per_class_precision[i]) for i in range(num_classes)
            },
            "per_class_recall": {
                str(i): float(per_class_recall[i]) for i in range(num_classes)
            },
            "per_class_f1": {
                str(i): float(per_class_f1[i]) for i in range(num_classes)
            },
            "confusion_matrix": cm,
        }


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
