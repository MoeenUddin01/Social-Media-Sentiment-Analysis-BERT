"""Metrics computation for sentiment classification evaluation.

Provides accuracy, precision, recall, F1 score, and confusion matrix
computation for multi-class sentiment analysis.
"""

from __future__ import annotations


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
