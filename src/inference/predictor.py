"""Inference predictor for BERT sentiment classification.

Loads trained models and makes predictions on new text data.
Follows the exact load order to ensure prediction works correctly:
1. Load preprocessor config
2. Load tokenizer
3. Load model with weights
4. Load label map
"""

from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING

import torch

from src.data.preprocessor import TextPreprocessor
from src.models.bert_classifier import BertSentimentClassifier
from src.models.tokenizer import SentimentTokenizer

if TYPE_CHECKING:
    from transformers import BatchEncoding


class SentimentPredictor:
    """Predictor for sentiment analysis on new text data.

    Loads all artifacts from a checkpoint directory and provides
    a predict() method for inference on raw text.

    Attributes:
        preprocessor: TextPreprocessor with training-time settings.
        tokenizer: SentimentTokenizer for encoding text.
        model: BertSentimentClassifier with loaded weights.
        label_map: Bidirectional label mapping dict.
        device: Device to run inference on.
    """

    def __init__(
        self,
        checkpoint_dir: pathlib.Path | str,
        device: torch.device | None = None,
    ) -> None:
        """Initialize predictor by loading all artifacts from checkpoint.

        Follows the exact 4-step load order:
        1. Load preprocessor via TextPreprocessor.load_config()
        2. Load tokenizer from checkpoint_dir/tokenizer/
        3. Load model via BertSentimentClassifier.load_for_inference()
        4. Load label_map.json for output conversion

        Args:
            checkpoint_dir: Directory containing checkpoint artifacts.
            device: Device to run inference on. If None, uses CPU.

        Raises:
            FileNotFoundError: If required artifacts are missing.
        """
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)

        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {self.checkpoint_dir}"
            )

        # Step 1: Load preprocessor config
        preprocessor_config_path = self.checkpoint_dir / "preprocessor_config.json"
        if preprocessor_config_path.exists():
            self.preprocessor = TextPreprocessor.load_config(preprocessor_config_path)
        else:
            # Use default if config not found
            self.preprocessor = TextPreprocessor()

        # Step 2: Load tokenizer
        tokenizer_path = self.checkpoint_dir / "tokenizer"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        self.tokenizer = SentimentTokenizer.load(tokenizer_path)

        # Step 3: Load model
        self.model = BertSentimentClassifier.load_for_inference(self.checkpoint_dir)

        # Step 4: Load label map (already attached to model, but verify)
        label_map_path = self.checkpoint_dir / "label_map.json"
        if not label_map_path.exists():
            raise FileNotFoundError(f"Label map not found: {label_map_path}")

        with open(label_map_path, encoding="utf-8") as f:
            self.label_map = json.load(f)

        # Set device
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict[str, str | float]:
        """Predict sentiment for a single text input.

        Args:
            text: Raw text string to classify.

        Returns:
            Dictionary containing:
                - predicted_label: Human-readable label ("positive", "negative", "neutral")
                - confidence: Confidence score (0.0 to 1.0)
                - probabilities: Dict of label probabilities
        """
        # Clean text using preprocessor
        import pandas as pd

        df = pd.DataFrame({"text": [text]})
        cleaned_df = self.preprocessor.preprocess(df)
        cleaned_text = cleaned_df["cleaned_text"].iloc[0]

        # Tokenize
        encoded = self.tokenizer.tokenize_batch([cleaned_text], max_length=128)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # Predict
        inputs: BatchEncoding = {"input_ids": input_ids, "attention_mask": attention_mask}

        with torch.no_grad():
            logits = self.model(inputs)
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()

        # Convert to label
        int_to_label = self.label_map["int_to_label"]
        predicted_label = int_to_label[str(predicted_class)]
        confidence = probabilities[0][predicted_class].item()

        # Build probability dict
        probs_dict = {
            int_to_label[str(i)]: probabilities[0][i].item()
            for i in range(len(int_to_label))
        }

        return {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "probabilities": probs_dict,
        }

    def predict_batch(self, texts: list[str]) -> list[dict[str, str | float]]:
        """Predict sentiment for a batch of texts.

        Args:
            texts: List of raw text strings to classify.

        Returns:
            List of prediction dictionaries.
        """
        return [self.predict(text) for text in texts]


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predictor.py <checkpoint_dir> [text]")
        sys.exit(1)

    checkpoint_dir = sys.argv[1]
    text = sys.argv[2] if len(sys.argv) > 2 else "This product is amazing!"

    predictor = SentimentPredictor(checkpoint_dir)
    result = predictor.predict(text)

    print(f"Text: {text}")
    print(f"Predicted: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Probabilities: {result['probabilities']}")
