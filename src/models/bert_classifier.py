"""BERT-based sentiment classifier implementation.

Provides a PyTorch module for fine-tuning pre-trained BERT models
on sentiment classification tasks.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import torch
from torch import nn
from transformers import BertModel

if TYPE_CHECKING:
    from transformers import BatchEncoding


# Label mapping for sentiment classification
LABELS = ["positive", "negative", "neutral"]
NUM_LABELS = len(LABELS)


class BertSentimentClassifier(nn.Module):
    """BERT-based sentiment classifier.

    Uses bert-base-uncased as the backbone with a custom classification
    head consisting of Dropout(0.3) followed by a Linear layer
    mapping from 768 hidden dimensions to num_labels.

    Attributes:
        bert: Pre-trained BERT model (bert-base-uncased).
        dropout: Dropout layer with p=0.3.
        classifier: Linear layer mapping 768 -> num_labels.
    """

    def __init__(self, num_labels: int = NUM_LABELS) -> None:
        """Initialize the classifier.

        Args:
            num_labels: Number of output labels. Defaults to 3
                (positive, negative, neutral).
        """
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, inputs: BatchEncoding) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            inputs: Tokenized inputs (input_ids, attention_mask, etc.)
                from the BERT tokenizer.

        Returns:
            Logits tensor of shape (batch_size, num_labels).
        """
        outputs = self.bert(**inputs)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_name: str,
        checkpoints_dir: pathlib.Path | str | None = None,
        num_labels: int = NUM_LABELS,
    ) -> "BertSentimentClassifier":
        """Load a pretrained model from a checkpoint.

        Args:
            checkpoint_name: Name of the checkpoint file (e.g., 'model.pt').
            checkpoints_dir: Directory containing checkpoints. Defaults to
                artifacts/checkpoints/ relative to project root.
            num_labels: Number of output labels. Defaults to 3.

        Returns:
            Loaded BertSentimentClassifier instance.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        if checkpoints_dir is None:
            # Get project root (3 levels up: src/models/ -> src/ -> project/)
            project_root = pathlib.Path(__file__).parent.parent.parent
            checkpoints_dir = project_root / "artifacts" / "checkpoints"
        else:
            checkpoints_dir = pathlib.Path(checkpoints_dir)

        model = cls(num_labels=num_labels)
        checkpoint_path = checkpoints_dir / checkpoint_name

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        state_dict = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        model.load_state_dict(state_dict)
        return model

    def save(
        self,
        checkpoint_name: str,
        checkpoints_dir: pathlib.Path | str | None = None,
    ) -> pathlib.Path:
        """Save model checkpoint to disk.

        Args:
            checkpoint_name: Name of the checkpoint file (e.g., 'model.pt').
            checkpoints_dir: Directory to save checkpoint. Defaults to
                artifacts/checkpoints/ relative to project root.

        Returns:
            Path where the checkpoint was saved.
        """
        if checkpoints_dir is None:
            # Get project root (3 levels up: src/models/ -> src/ -> project/)
            project_root = pathlib.Path(__file__).parent.parent.parent
            checkpoints_dir = project_root / "artifacts" / "checkpoints"
        else:
            checkpoints_dir = pathlib.Path(checkpoints_dir)

        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoints_dir / checkpoint_name

        torch.save(self.state_dict(), checkpoint_path)
        return checkpoint_path

    @classmethod
    def load_for_inference(
        cls,
        checkpoint_dir: pathlib.Path | str,
        checkpoint_name: str | None = None,
    ) -> "BertSentimentClassifier":
        """Load model for inference with all required artifacts.

        Step 1: Load model_config.json to rebuild architecture
        Step 2: Load model.pt to load trained weights
        Step 3: Set model to eval mode
        Step 4: Load and attach label_map.json for output conversion

        Args:
            checkpoint_dir: Directory containing checkpoint artifacts.
            checkpoint_name: Name of model file. If None, uses 'model.pt'.

        Returns:
            Model ready for prediction with attached label_map attribute.

        Raises:
            FileNotFoundError: If required files not found in checkpoint_dir.
        """
        import json

        checkpoint_dir = pathlib.Path(checkpoint_dir)

        # Step 1: Load model_config.json to rebuild architecture
        model_config_path = checkpoint_dir / "model_config.json"
        if not model_config_path.exists():
            raise FileNotFoundError(f"model_config.json not found in {checkpoint_dir}")

        with open(model_config_path, encoding="utf-8") as f:
            model_config = json.load(f)

        num_labels = model_config.get("num_labels", NUM_LABELS)

        # Step 2: Create model and load weights
        model = cls(num_labels=num_labels)

        if checkpoint_name is None:
            checkpoint_name = "model.pt"

        checkpoint_path = checkpoint_dir / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state_dict = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        model.load_state_dict(state_dict)

        # Step 3: Set to eval mode (disables dropout)
        model.eval()

        # Step 4: Load label_map.json for output conversion
        label_map_path = checkpoint_dir / "label_map.json"
        if not label_map_path.exists():
            raise FileNotFoundError(f"label_map.json not found in {checkpoint_dir}")

        with open(label_map_path, encoding="utf-8") as f:
            label_map = json.load(f)

        # Attach label maps to model for easy access during prediction
        model.label_map = label_map
        model.int_to_label = label_map["int_to_label"]
        model.label_to_int = label_map["label_to_int"]

        return model
