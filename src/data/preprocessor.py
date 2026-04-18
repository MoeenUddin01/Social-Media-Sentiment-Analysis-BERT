"""Text preprocessing and data augmentation for social media.

Provides utilities for cleaning social media text, handling emojis,
URLs, mentions, and applying augmentation techniques.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight

from src.data.cleaner import TweetCleaner

if TYPE_CHECKING:
    import numpy as np


class TextPreprocessor:
    """Preprocessor for social media text with TweetCleaner integration.

    Chains TweetCleaner methods with emoji removal and lowercasing to produce
    cleaned text suitable for BERT tokenization. Handles class imbalance
    through weighted loss computation.
    """

    def __init__(self, lowercase: bool = True, remove_emojis: bool = True) -> None:
        """Initialize the preprocessor with TweetCleaner instance.

        Args:
            lowercase: Whether to convert text to lowercase.
            remove_emojis: Whether to remove emojis from text.
        """
        self.cleaner = TweetCleaner()
        self.lowercase = lowercase
        self.remove_emojis = remove_emojis
        # Unicode ranges for emoji removal
        self._emoji_pattern = re.compile(
            r"["
            r"\U0001F600-\U0001F64F"  # emoticons
            r"\U0001F300-\U0001F5FF"  # symbols & pictographs
            r"\U0001F680-\U0001F6FF"  # transport & map symbols
            r"\U0001F1E0-\U0001F1FF"  # flags
            r"\U00002702-\U000027B0"  # dingbats
            r"\U000024C2-\U0001F251"
            r"]+",
            flags=re.UNICODE,
        )

    def _clean_text(self, text: str) -> str:
        """Apply all cleaning steps to a single text string.

        Pipeline: remove URLs → mentions → hashtags → emojis →
        special chars → lowercase → normalize whitespace.

        Args:
            text: Raw input text to clean.

        Returns:
            Cleaned text string.
        """
        text = self.cleaner.remove_urls(text)
        text = self.cleaner.remove_mentions(text)
        text = self.cleaner.remove_hashtags(text)

        if self.remove_emojis:
            text = self._emoji_pattern.sub("", text)

        text = self.cleaner.remove_special_chars(text)

        if self.lowercase:
            text = text.lower()

        text = self.cleaner.normalize_whitespace(text)

        return text

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text in a DataFrame and return augmented DataFrame.

        Expects 'text' and 'label' columns. Returns DataFrame with
        additional 'cleaned_text' column preserving original 'text'.

        Args:
            df: DataFrame with 'text' (str) and 'label' columns.

        Returns:
            DataFrame with added 'cleaned_text' column.

        Raises:
            KeyError: If 'text' or 'label' columns are missing.
        """
        if "text" not in df.columns:
            raise KeyError("DataFrame must contain 'text' column")
        if "label" not in df.columns:
            raise KeyError("DataFrame must contain 'label' column")

        result = df.copy()
        result["cleaned_text"] = result["text"].apply(self._clean_text)

        return result

    def get_class_weights(
        self,
        labels: pd.Series | "np.ndarray",
    ) -> torch.Tensor:
        """Compute class weights for imbalanced datasets.

        Uses sklearn's balanced class weight calculation.
        Returns PyTorch tensor for use with CrossEntropyLoss.

        Args:
            labels: Array-like of class labels.

        Returns:
            PyTorch tensor of class weights shape (n_classes,).
        """
        import numpy as np

        classes = np.unique(labels)
        weights = compute_class_weight(
            "balanced",
            classes=classes,
            y=labels,
        )

        return torch.tensor(weights, dtype=torch.float32)
