"""Model evaluation pipeline for BERT sentiment classification.

Provides EvaluationPipeline for running evaluations, generating reports,
and comparing multiple model runs.
"""

from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.bert_classifier import BertSentimentClassifier
from src.models.evaluator import ModelEvaluator
from src.models.tokenizer import SentimentTokenizer
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from logging import Logger


class EvaluationPipeline:
    """Evaluation pipeline for BERT sentiment classification.

    Loads a trained model from checkpoint, runs evaluation on test data,
    generates reports, and compares multiple model runs.

    Attributes:
        checkpoint_dir: Path to the checkpoint directory.
        device: Device to run evaluation on.
        model: Loaded BERT sentiment classifier.
        label_map: Mapping from integer labels to class names.
        evaluator: ModelEvaluator instance for metrics and reporting.
        logger: Logger instance.
    """

    def __init__(
        self,
        checkpoint_dir: pathlib.Path | str,
        device: torch.device,
    ) -> None:
        """Initialize the evaluation pipeline.

        Args:
            checkpoint_dir: Path to the checkpoint directory containing
                model checkpoint, label_map.json, and other artifacts.
            device: Device to run evaluation on (CPU or CUDA).

        Raises:
            FileNotFoundError: If checkpoint directory or label_map.json not found.
        """
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.device = device
        self.logger: Logger = get_logger(__name__)

        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {self.checkpoint_dir}"
            )

        # Load model via load_for_inference
        self.model = BertSentimentClassifier.load_for_inference(self.checkpoint_dir)
        self.model.to(self.device)

        # Load label_map from checkpoint directory
        label_map_path = self.checkpoint_dir / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path, encoding="utf-8") as f:
                self.label_map: dict[int, str] = {
                    int(k): v for k, v in json.load(f).items()
                }
        else:
            self.label_map = {0: "negative", 1: "neutral", 2: "positive"}

        # Instantiate ModelEvaluator
        self.evaluator = ModelEvaluator(self.model, self.device, self.label_map)

        self.logger.info(f"EvaluationPipeline initialized from {checkpoint_dir}")

    def run(self, test_loader: DataLoader) -> dict[str, float]:
        """Run evaluation on test dataloader.

        Args:
            test_loader: DataLoader containing test data.

        Returns:
            Dictionary of evaluation metrics.
        """
        self.logger.info("Running evaluation...")

        # Delegate to ModelEvaluator.evaluate()
        metrics = self.evaluator.evaluate(test_loader)

        # Save report
        self.save_report(self.checkpoint_dir)

        return metrics

    def run_single(self, text: str) -> dict[str, float | str]:
        """Evaluate a single text sample.

        Args:
            text: Raw text string to classify.

        Returns:
            Dictionary containing:
                - predicted_label: Predicted class name
                - confidence: Confidence score (probability of predicted class)
                - class_probabilities: Dict mapping class names to probabilities
        """
        # Initialize tokenizer
        tokenizer = SentimentTokenizer()

        # Delegate to ModelEvaluator.evaluate_single()
        return self.evaluator.evaluate_single(text, tokenizer)

    def compare_runs(self, checkpoint_dirs: list[str]) -> pd.DataFrame:
        """Compare metrics across multiple model runs.

        Args:
            checkpoint_dirs: List of checkpoint directories to compare.

        Returns:
            DataFrame with metrics for each checkpoint.
        """
        # Delegate to ModelEvaluator.compare_checkpoints()
        return self.evaluator.compare_checkpoints(checkpoint_dirs)

    def save_report(self, output_dir: pathlib.Path | str) -> None:
        """Save evaluation report and plots.

        Args:
            output_dir: Directory to save evaluation reports.
        """
        # Delegate to ModelEvaluator.save_report()
        self.evaluator.save_report(output_dir)

    @staticmethod
    def load_best_checkpoint(checkpoint_root: str = "artifacts/checkpoints") -> str:
        """Load the best checkpoint from registry.

        Args:
            checkpoint_root: Root directory containing registry.json.

        Returns:
            Path to the best checkpoint directory.

        Raises:
            FileNotFoundError: If registry.json not found.
            KeyError: If best_checkpoint entry not found in registry.
        """
        registry_path = pathlib.Path(checkpoint_root) / "registry.json"

        if not registry_path.exists():
            raise FileNotFoundError(f"Registry file not found: {registry_path}")

        with open(registry_path, encoding="utf-8") as f:
            registry = json.load(f)

        best_checkpoint = registry.get("best_checkpoint")
        if best_checkpoint is None:
            raise KeyError("No 'best_checkpoint' entry found in registry")

        return str(best_checkpoint.get("path", ""))

