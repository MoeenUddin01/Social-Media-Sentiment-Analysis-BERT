"""Model evaluation utilities for sentiment classification.

Provides the Evaluator class for raw prediction and metric computation.
This is a lightweight helper class used by ModelEvaluator and TrainingPipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from src.models.bert_classifier import BertSentimentClassifier
from src.utils.logger import get_logger
from src.utils.metrics import SentimentMetrics

if TYPE_CHECKING:
    from logging import Logger

    from torch.utils.data import DataLoader


class Evaluator:
    """Lightweight evaluator for raw prediction and metric computation.

    Handles model inference on batches and computes classification metrics.
    This is a helper class used by ModelEvaluator and TrainingPipeline.

    Attributes:
        model: BERT sentiment classifier model.
        device: Device to run evaluation on.
        label_map: Mapping from integer labels to class names.
        logger: Logger instance.
    """

    def __init__(
        self,
        model: BertSentimentClassifier,
        device: torch.device,
        label_map: dict[int, str] | None = None,
    ) -> None:
        """Initialize the Evaluator.

        Args:
            model: BERT sentiment classifier model.
            device: Device to run evaluation on (CPU or CUDA).
            label_map: Mapping from integer labels to class names.
                Defaults to {0: "negative", 1: "neutral", 2: "positive"}.
        """
        self.model = model
        self.device = device
        self.label_map = label_map or {0: "negative", 1: "neutral", 2: "positive"}
        self.logger: Logger = get_logger(__name__)

        # Put model in eval mode
        self.model.eval()

    def predict_proba(
        self,
        dataloader: DataLoader,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate predictions with probabilities.

        Runs the model on the dataloader with torch.no_grad() and returns
        predictions, labels, and softmax probabilities.

        Args:
            dataloader: DataLoader containing data to predict on.

        Returns:
            Tuple of (all_preds, all_labels, all_probs) as numpy arrays:
                - all_preds: Predicted class indices (N,)
                - all_labels: True labels (N,)
                - all_probs: Softmax probabilities for all classes (N, num_classes)
        """
        all_preds: list[int] = []
        all_labels: list[int] = []
        all_probs: list[np.ndarray] = []

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
                logits = self.model(inputs)
                probs = F.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy())

        return (
            np.array(all_preds),
            np.array(all_labels),
            np.array(all_probs),
        )

    def compute_metrics(
        self,
        all_preds: np.ndarray,
        all_labels: np.ndarray,
        all_probs: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute classification metrics.

        Delegates to SentimentMetrics.compute_all() for metric calculation.

        Args:
            all_preds: Predicted class indices (N,).
            all_labels: True labels (N,).
            all_probs: Optional probability predictions (N, num_classes).

        Returns:
            Dictionary containing accuracy, macro_f1, weighted_f1,
            per-class precision, recall, f1, and confusion matrix.
        """
        return SentimentMetrics.compute_all(all_labels, all_preds, all_probs)

    def full_eval(self, dataloader: DataLoader) -> dict[str, float]:
        """Run full evaluation on a dataloader.

        Chains predict_proba -> compute_metrics for a single-call evaluation.
        Used by ModelEvaluator and TrainingPipeline.

        Args:
            dataloader: DataLoader containing evaluation data.

        Returns:
            Dictionary with all computed metrics.
        """
        self.logger.info("Running full evaluation...")

        all_preds, all_labels, all_probs = self.predict_proba(dataloader)
        metrics = self.compute_metrics(all_preds, all_labels, all_probs)

        self.logger.info(
            f"Evaluation complete - Accuracy: {metrics['accuracy']:.4f}, "
            f"Macro F1: {metrics['macro_f1']:.4f}"
        )

        return metrics

