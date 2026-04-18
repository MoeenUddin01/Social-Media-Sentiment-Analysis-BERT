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
        num_labels: int = NUM_LABELS,
    ) -> "BertSentimentClassifier":
        """Load model for inference with eval mode.

        Convenience method that loads a checkpoint and puts the model
        in eval mode automatically.

        Args:
            checkpoint_dir: Directory containing the checkpoint file.
            checkpoint_name: Name of checkpoint file. If None, finds
                first .pt file in checkpoint_dir.
            num_labels: Number of output labels. Defaults to 3.

        Returns:
            Loaded model in eval mode.

        Raises:
            FileNotFoundError: If checkpoint not found.
        """
        checkpoint_dir = pathlib.Path(checkpoint_dir)

        # Find checkpoint file if not specified
        if checkpoint_name is None:
            checkpoint_files = list(checkpoint_dir.glob("*.pt"))
            if not checkpoint_files:
                raise FileNotFoundError(
                    f"No checkpoint files found in {checkpoint_dir}"
                )
            checkpoint_path = checkpoint_files[0]
            checkpoint_name = checkpoint_path.name

        # Load model via from_pretrained
        model = cls.from_pretrained(
            checkpoint_name=checkpoint_name,
            checkpoints_dir=checkpoint_dir,
            num_labels=num_labels,
        )

        # Put in eval mode for inference
        model.eval()
        return model
