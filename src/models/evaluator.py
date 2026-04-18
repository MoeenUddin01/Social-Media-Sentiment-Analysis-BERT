"""Model evaluation utilities for BERT sentiment classification.

Provides comprehensive evaluation metrics, prediction utilities, and
report generation for sentiment analysis models.
"""

from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.models.bert_classifier import BertSentimentClassifier
from src.models.tokenizer import SentimentTokenizer
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.utils.data import DataLoader


logger = get_logger(__name__)


class ModelEvaluator:
    """Evaluator for BERT sentiment classification models.

    Provides comprehensive evaluation metrics, prediction utilities,
    checkpoint comparison, and report generation.

    Attributes:
        model: The BERT sentiment classifier model.
        device: Device to run evaluation on.
        label_map: Mapping from integer labels to class names.
    """

    def __init__(
        self,
        model: BertSentimentClassifier,
        device: torch.device,
        label_map: dict[int, str] | None = None,
    ) -> None:
        """Initialize the model evaluator.

        Args:
            model: The BERT sentiment classifier to evaluate.
            device: Device to run evaluation on (CPU or CUDA).
            label_map: Mapping from integer labels to class names.
                Defaults to {0: "negative", 1: "neutral", 2: "positive"}.
        """
        self.model = model
        self.device = device
        self.label_map = label_map or {0: "negative", 1: "neutral", 2: "positive"}
        self.model.eval()
        logger.info("ModelEvaluator initialized with model in eval mode")

    def evaluate(self, dataloader: DataLoader) -> dict[str, float | np.ndarray]:
        """Evaluate model on a dataloader.

        Runs the full dataloader and computes comprehensive metrics.

        Args:
            dataloader: DataLoader containing evaluation data.

        Returns:
            Dictionary containing:
                - accuracy: Overall accuracy score
                - macro_f1: Macro-averaged F1 score
                - weighted_f1: Weighted-averaged F1 score
                - precision: Macro-averaged precision
                - recall: Macro-averaged recall
                - confusion_matrix: Confusion matrix as numpy array
        """
        all_preds, all_labels, all_probs = self.predict_proba(dataloader)

        accuracy = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
        precision = precision_score(all_labels, all_preds, average="macro")
        recall = recall_score(all_labels, all_preds, average="macro")
        cm = confusion_matrix(all_labels, all_preds)

        metrics = {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "precision": float(precision),
            "recall": float(recall),
            "confusion_matrix": cm,
        }

        logger.info(f"Evaluation complete - Accuracy: {accuracy:.4f}, Macro F1: {macro_f1:.4f}")
        return metrics

    def predict_proba(
        self, dataloader: DataLoader
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate predictions with probabilities.

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

    def classification_report(self) -> str:
        """Generate sklearn classification report.

        Must call evaluate() or predict_proba() first to populate predictions.

        Returns:
            Classification report string with label names.

        Raises:
            RuntimeError: If evaluate() hasn't been called.
        """
        if not hasattr(self, "_last_preds") or not hasattr(self, "_last_labels"):
            raise RuntimeError("Must call evaluate() or predict_proba() first")

        target_names = [self.label_map[i] for i in sorted(self.label_map.keys())]
        return classification_report(
            self._last_labels,
            self._last_preds,
            target_names=target_names,
            digits=4,
        )

    def evaluate_single(
        self, text: str, tokenizer: SentimentTokenizer
    ) -> dict[str, float | str]:
        """Evaluate a single text sample.

        Args:
            text: Raw text string to classify.
            tokenizer: Tokenizer to use for encoding.

        Returns:
            Dictionary containing:
                - predicted_label: Predicted class name
                - confidence: Confidence score (probability of predicted class)
                - class_probabilities: Dict mapping class names to probabilities
        """
        self.model.eval()
        with torch.no_grad():
            encoded = tokenizer.tokenize_batch([text], max_length=128)
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            logits = self.model(inputs)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        predicted_label = self.label_map[pred_idx]
        confidence = float(probs[pred_idx])

        class_probabilities = {
            self.label_map[i]: float(probs[i]) for i in range(len(probs))
        }

        return {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "class_probabilities": class_probabilities,
        }

    def compare_checkpoints(
        self, checkpoint_dirs: Sequence[str | pathlib.Path]
    ) -> pd.DataFrame:
        """Compare metrics across multiple checkpoints.

        Loads each checkpoint and evaluates on the current dataloader,
        returning a comparison DataFrame.

        Args:
            checkpoint_dirs: List of checkpoint directories to compare.

        Returns:
            DataFrame with rows for each checkpoint and columns for metrics.
        """
        results: list[dict] = []

        for checkpoint_dir in checkpoint_dirs:
            checkpoint_path = pathlib.Path(checkpoint_dir)
            if not checkpoint_path.exists():
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                continue

            # Load checkpoint
            checkpoint_name = checkpoint_path.name
            try:
                model = BertSentimentClassifier.from_pretrained(checkpoint_name)
                model.to(self.device)
                model.eval()
                logger.info(f"Loaded checkpoint: {checkpoint_path}")

                # Create temporary evaluator
                temp_evaluator = ModelEvaluator(model, self.device, self.label_map)

                result = {
                    "checkpoint": checkpoint_name,
                    "path": str(checkpoint_path),
                }
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to load {checkpoint_path}: {e}")
                continue

        return pd.DataFrame(results)

    def save_report(self, output_dir: pathlib.Path | str) -> None:
        """Save evaluation reports and visualizations.

        Saves:
        - eval_report.txt: Human-readable classification report
        - eval_results.json: JSON metrics
        - confusion_matrix.png: Confusion matrix plot
        - confidence_histogram.png: Confidence distribution plot
        - per_class_f1_bar.png: Per-class F1 score bar chart

        Args:
            output_dir: Directory to save report files to.
        """
        output_path = pathlib.Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save text report
        report_path = output_path / "eval_report.txt"
        try:
            report = self.classification_report()
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"Saved classification report: {report_path}")
        except RuntimeError:
            logger.warning("No predictions available for classification report")

        # Save JSON results
        json_path = output_path / "eval_results.json"
        if hasattr(self, "_last_metrics"):
            metrics_to_save = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self._last_metrics.items()
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metrics_to_save, f, indent=2)
            logger.info(f"Saved metrics JSON: {json_path}")

        # Generate visualizations
        try:
            self._generate_visualizations(output_path)
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")

    def _generate_visualizations(self, output_path: pathlib.Path) -> None:
        """Generate and save evaluation visualizations.

        Args:
            output_path: Directory to save visualization files.
        """
        try:
            from src.utils.visualizer import ResultVisualizer

            visualizer = ResultVisualizer()

            # Confusion matrix
            if hasattr(self, "_last_metrics") and "confusion_matrix" in self._last_metrics:
                cm = self._last_metrics["confusion_matrix"]
                cm_path = output_path / "confusion_matrix.png"
                visualizer.plot_confusion_matrix(cm, self.label_map, cm_path)

            # Confidence histogram
            if hasattr(self, "_last_probs") and hasattr(self, "_last_preds"):
                confidences = self._last_probs[
                    np.arange(len(self._last_preds)), self._last_preds
                ]
                conf_path = output_path / "confidence_histogram.png"
                visualizer.plot_confidence_histogram(confidences, conf_path)

            # Per-class F1 bar chart
            if hasattr(self, "_last_preds") and hasattr(self, "_last_labels"):
                from sklearn.metrics import f1_score

                per_class_f1 = f1_score(
                    self._last_labels,
                    self._last_preds,
                    average=None,
                )
                f1_path = output_path / "per_class_f1_bar.png"
                label_names = [self.label_map[i] for i in range(len(per_class_f1))]
                visualizer.plot_per_class_f1_bar(per_class_f1, label_names, f1_path)

        except ImportError as e:
            logger.warning(f"Visualizer not available: {e}")
