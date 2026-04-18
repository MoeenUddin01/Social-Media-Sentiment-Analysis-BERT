"""Model evaluation pipeline for BERT sentiment classification.

Provides a thin wrapper around ModelEvaluator for running evaluations
and generating reports.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from src.data.dataset import SentimentDataset
from src.models.bert_classifier import BertSentimentClassifier
from src.models.evaluator import ModelEvaluator
from src.models.tokenizer import SentimentTokenizer

if TYPE_CHECKING:
    from src.models.bert_classifier import BertSentimentClassifier


class EvaluationPipeline:
    """Evaluation pipeline as a thin wrapper around ModelEvaluator.

    Delegates all prediction logic to ModelEvaluator, keeping only
    pipeline orchestration in this class.
    """

    def __init__(
        self,
        model: BertSentimentClassifier,
        device: torch.device,
        label_map: dict[int, str] | None = None,
    ) -> None:
        """Initialize the evaluation pipeline.

        Args:
            model: The BERT sentiment classifier to evaluate.
            device: Device to run evaluation on.
            label_map: Mapping from integer labels to class names.
        """
        self.model = model
        self.device = device
        self.label_map = label_map or {0: "negative", 1: "neutral", 2: "positive"}
        self.evaluator = ModelEvaluator(model, device, self.label_map)

    def run_evaluation(
        self,
        dataloader: DataLoader,
        output_dir: pathlib.Path | str | None = None,
    ) -> dict[str, float]:
        """Run full evaluation and save reports.

        Args:
            dataloader: DataLoader containing evaluation data.
            output_dir: Directory to save evaluation reports.

        Returns:
            Dictionary of evaluation metrics.
        """
        # Delegate to ModelEvaluator.evaluate()
        metrics = self.evaluator.evaluate(dataloader)

        # Save reports if output directory provided
        if output_dir is not None:
            self.evaluator.save_report(output_dir)

        return metrics

    def evaluate_single(self, text: str, tokenizer: SentimentTokenizer) -> dict[str, float | str]:
        """Evaluate a single text sample.

        Args:
            text: Raw text string to classify.
            tokenizer: Tokenizer for encoding.

        Returns:
            Dictionary with prediction results.
        """
        # Delegate to ModelEvaluator.evaluate_single()
        return self.evaluator.evaluate_single(text, tokenizer)


def run_evaluation_pipeline(
    checkpoint_path: pathlib.Path | str,
    test_data_path: pathlib.Path | str,
    output_dir: pathlib.Path | str | None = None,
) -> dict[str, float]:
    """Convenience function to run full evaluation pipeline.

    Args:
        checkpoint_path: Path to model checkpoint file.
        test_data_path: Path to test data CSV file.
        output_dir: Directory to save evaluation reports.

    Returns:
        Dictionary of evaluation metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model via BertSentimentClassifier.from_pretrained
    checkpoint_path_obj = pathlib.Path(checkpoint_path)
    model = BertSentimentClassifier.from_pretrained(
        checkpoint_name=checkpoint_path_obj.name,
        checkpoints_dir=checkpoint_path_obj.parent,
    )
    model.to(device)

    # Initialize tokenizer
    tokenizer = SentimentTokenizer()

    # Create test dataloader
    test_dataset = SentimentDataset(
        data=str(test_data_path),
        tokenizer=tokenizer.tokenizer,
        max_length=128,
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Run evaluation pipeline
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    pipeline = EvaluationPipeline(model, device, label_map)

    if output_dir is None:
        output_dir = checkpoint_path_obj.parent

    return pipeline.run_evaluation(test_loader, output_dir)
