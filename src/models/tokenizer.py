"""Tokenizer management for BERT models.

Provides utilities for loading, saving, and managing HuggingFace tokenizers
with consistent configuration.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import torch
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from transformers import BatchEncoding


class SentimentTokenizer:
    """BERT tokenizer wrapper for sentiment analysis.

    Wraps HuggingFace AutoTokenizer for bert-base-uncased with methods
    for batch tokenization, saving, and loading. Returns tensors suitable
    for model input.

    Attributes:
        tokenizer: The underlying HuggingFace AutoTokenizer instance.
        model_name: Name of the pre-trained model used.
    """

    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        """Initialize the tokenizer.

        Args:
            model_name: Pre-trained model name to load tokenizer from.
                Defaults to "bert-base-uncased".
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, text: str, max_length: int = 128, **kwargs) -> BatchEncoding:
        """Tokenize a single text string.

        Makes the tokenizer callable like the underlying HuggingFace tokenizer.
        Used by SentimentDataset for single-sample tokenization.

        Args:
            text: Text string to tokenize.
            max_length: Maximum sequence length. Defaults to 128.
            **kwargs: Additional arguments passed to the tokenizer.

        Returns:
            BatchEncoding with 'input_ids' and 'attention_mask'.
        """
        return self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            **kwargs
        )

    def tokenize_batch(
        self,
        texts: list[str],
        max_length: int = 128,
    ) -> BatchEncoding:
        """Tokenize a batch of texts.

        Applies padding and truncation to ensure consistent tensor shapes.
        Returns input_ids and attention_mask as tensors.

        Args:
            texts: List of text strings to tokenize.
            max_length: Maximum sequence length. Defaults to 128.

        Returns:
            BatchEncoding with 'input_ids' and 'attention_mask' tensors.
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return encoded

    def save(
        self, path: pathlib.Path | str | None = None
    ) -> pathlib.Path:
        """Save the tokenizer to disk.

        Args:
            path: Directory path to save the tokenizer files.
                Defaults to artifacts/tokenizers/{model_name}/.

        Returns:
            Path where the tokenizer was saved.
        """
        if path is None:
            # Get project root (3 levels up from this file: src/models/)
            project_root = pathlib.Path(__file__).parent.parent.parent
            save_path = project_root / "artifacts" / "tokenizers" / self.model_name.replace("/", "_")
        else:
            save_path = pathlib.Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(save_path)
        return save_path

    @classmethod
    def load(
        cls, path: pathlib.Path | str | None = None, model_name: str = "bert-base-uncased"
    ) -> "SentimentTokenizer":
        """Load a tokenizer from disk.

        Args:
            path: Directory path containing saved tokenizer files.
                Defaults to artifacts/tokenizers/{model_name}/.
            model_name: Model name for default path construction.
                Defaults to "bert-base-uncased".

        Returns:
            Loaded SentimentTokenizer instance.

        Raises:
            FileNotFoundError: If the tokenizer files do not exist at path.
        """
        if path is None:
            # Get project root (3 levels up from this file: src/models/)
            project_root = pathlib.Path(__file__).parent.parent.parent
            load_path = project_root / "artifacts" / "tokenizers" / model_name.replace("/", "_")
        else:
            load_path = pathlib.Path(path)

        if not load_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at: {load_path}")

        instance = cls.__new__(cls)
        instance.model_name = str(load_path)
        instance.tokenizer = AutoTokenizer.from_pretrained(load_path)
        return instance
