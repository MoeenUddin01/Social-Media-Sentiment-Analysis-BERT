"""PyTorch Dataset for sentiment analysis.

Provides SentimentDataset for loading processed data with BERT tokenization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


class SentimentDataset(Dataset):
    """PyTorch Dataset for sentiment classification.

    Loads data from a DataFrame or CSV file in dataset/processed/,
    tokenizes text using a HuggingFace tokenizer, and returns tensors
    suitable for BERT model training.
    """

    def __init__(
        self,
        data: pd.DataFrame | str,
        tokenizer: "PreTrainedTokenizer",
        max_length: int = 128,
        text_column: str = "cleaned_text",
        label_column: str = "label",
    ) -> None:
        """Initialize the dataset.

        Args:
            data: DataFrame with text and labels, or path to CSV in dataset/processed/.
            tokenizer: HuggingFace tokenizer (e.g., BertTokenizer).
            max_length: Maximum sequence length for tokenization.
            text_column: Column name containing text to tokenize.
            label_column: Column name containing labels.

        Raises:
            FileNotFoundError: If data is a path and the file does not exist.
            KeyError: If required columns are missing from the DataFrame.
        """
        import os

        if isinstance(data, str):
            # If path is relative, look in dataset/processed/
            if not os.path.isabs(data):
                project_root = os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                )
                data = os.path.join(project_root, "dataset", "processed", data)
            self.df = pd.read_csv(data)
        else:
            self.df = data.copy()

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column

        if self.text_column not in self.df.columns:
            raise KeyError(f"Column '{text_column}' not found in DataFrame")
        if self.label_column not in self.df.columns:
            raise KeyError(f"Column '{label_column}' not found in DataFrame")

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            Number of samples.
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Dictionary containing:
                - 'input_ids': Token IDs tensor (max_length,)
                - 'attention_mask': Attention mask tensor (max_length,)
                - 'labels': Label tensor (scalar)
        """
        row = self.df.iloc[idx]
        text = str(row[self.text_column])
        label = int(row[self.label_column])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
