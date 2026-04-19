"""Data preprocessing pipeline for BERT sentiment analysis.

Provides the DataPipeline class for end-to-end data loading, cleaning,
splitting, and DataLoader creation for training and inference.
"""

from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data.cleaner import TweetCleaner
from src.data.dataset import SentimentDataset
from src.data.preprocessor import TextPreprocessor
from src.models.tokenizer import SentimentTokenizer
from src.utils.logger import get_logger
from src.utils.seed import seed_worker

if TYPE_CHECKING:
    from logging import Logger


class DataPipeline:
    """End-to-end data preprocessing pipeline for sentiment analysis.

    Orchestrates the full data workflow: loading raw CSV, cleaning text,
    splitting into train/val/test sets, and wrapping in PyTorch DataLoaders.

    Attributes:
        config: Configuration dictionary with data and training parameters.
        tokenizer: SentimentTokenizer for text tokenization.
        raw_dir: Path to raw data directory.
        processed_dir: Path to processed data directory.
        logger: Logger instance for pipeline events.
        cleaner: TweetCleaner instance for text cleaning.
        preprocessor: TextPreprocessor instance for text preprocessing.
    """

    def __init__(
        self,
        config: dict,
        tokenizer: SentimentTokenizer,
    ) -> None:
        """Initialize the DataPipeline.

        Args:
            config: Configuration dictionary loaded from config.yaml.
                Must contain 'data' section with raw_dir, processed_dir,
                train_split, val_split, test_split, batch_size, num_workers,
                and pin_memory keys.
            tokenizer: SentimentTokenizer instance for text tokenization.

        Raises:
            KeyError: If required keys are missing from config.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.logger: Logger = get_logger(__name__)

        data_config = config.get("data", {})
        self.raw_dir = pathlib.Path(data_config.get("raw_dir", "dataset/raw"))
        self.processed_dir = pathlib.Path(
            data_config.get("processed_dir", "dataset/processed")
        )

        self.cleaner = TweetCleaner()
        self.preprocessor = TextPreprocessor()

    def load_raw(self, filename: str) -> pd.DataFrame:
        """Load raw data from CSV file.

        Reads CSV from dataset/raw/ directory and validates required columns.
        Logs information about missing values and duplicate rows.

        Args:
            filename: Name of the CSV file in the raw directory.

        Returns:
            DataFrame with raw data containing 'text' and 'label' columns.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            KeyError: If required columns ('text', 'label') are missing.
        """
        file_path = self.raw_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {file_path}")

        df = pd.read_csv(file_path)

        required_columns = ["text", "label"]
        for col in required_columns:
            if col not in df.columns:
                raise KeyError(f"Required column '{col}' not found in {filename}")

        missing_count = df.isnull().sum().sum()
        duplicate_count = df.duplicated().sum()

        self.logger.info(f"Loaded {len(df)} rows from {filename}")
        self.logger.info(
            f"Missing values: {missing_count}, Duplicates: {duplicate_count}"
        )

        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the DataFrame.

        Passes DataFrame through TweetCleaner and TextPreprocessor,
        then saves the cleaned output to dataset/processed/.

        Args:
            df: Raw DataFrame with 'text' and 'label' columns.

        Returns:
            Cleaned DataFrame with additional 'cleaned_text' column.

        Raises:
            KeyError: If required columns are missing from DataFrame.
        """
        self.logger.info("Starting data cleaning...")

        cleaned_df = self.preprocessor.preprocess(df)

        self.processed_dir.mkdir(parents=True, exist_ok=True)

        output_path = self.processed_dir / "cleaned_data.csv"
        cleaned_df.to_csv(output_path, index=False)

        self.logger.info(f"Cleaned data saved to {output_path}")
        self.logger.info(f"Cleaned {len(cleaned_df)} rows")

        return cleaned_df

    def build_datasets(
        self,
        df: pd.DataFrame,
    ) -> tuple[SentimentDataset, SentimentDataset, SentimentDataset]:
        """Split DataFrame and wrap in SentimentDataset instances.

        Splits data into train/val/test sets using ratios from config.yaml,
        then wraps each split in a SentimentDataset.

        Args:
            df: Cleaned DataFrame with 'cleaned_text' and 'label' columns.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset).

        Raises:
            KeyError: If required split ratios are missing from config.
        """
        data_config = self.config.get("data", {})

        train_ratio = data_config.get("train_split", 0.8)
        val_ratio = data_config.get("val_split", 0.1)
        test_ratio = data_config.get("test_split", 0.1)

        max_length = self.config.get("training", {}).get("max_length", 128)

        temp_df, test_df = train_test_split(
            df,
            test_size=test_ratio,
            random_state=42,
            stratify=df["label"],
        )

        val_size = val_ratio / (train_ratio + val_ratio)
        train_df, val_df = train_test_split(
            temp_df,
            test_size=val_size,
            random_state=42,
            stratify=temp_df["label"],
        )

        self.logger.info(
            f"Split data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

        train_dataset = SentimentDataset(
            data=train_df,
            tokenizer=self.tokenizer.tokenizer,
            max_length=max_length,
        )
        val_dataset = SentimentDataset(
            data=val_df,
            tokenizer=self.tokenizer.tokenizer,
            max_length=max_length,
        )
        test_dataset = SentimentDataset(
            data=test_df,
            tokenizer=self.tokenizer.tokenizer,
            max_length=max_length,
        )

        return train_dataset, val_dataset, test_dataset

    def build_dataloaders(
        self,
        train_dataset: SentimentDataset,
        val_dataset: SentimentDataset,
        test_dataset: SentimentDataset,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Wrap datasets in PyTorch DataLoaders.

        Creates DataLoaders for train/val/test sets using batch_size,
        num_workers, and pin_memory settings from config.yaml.
        Uses seed_worker for reproducible data loading.

        Args:
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            test_dataset: Test dataset.

        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        data_config = self.config.get("data", {})

        batch_size = data_config.get("batch_size", 32)
        num_workers = data_config.get("num_workers", 4)
        pin_memory = data_config.get("pin_memory", True)

        generator = torch.Generator()
        generator.manual_seed(42)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            generator=generator,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            generator=generator,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            generator=generator,
        )

        self.logger.info(
            f"Created DataLoaders: batch_size={batch_size}, "
            f"num_workers={num_workers}, pin_memory={pin_memory}"
        )

        return train_loader, val_loader, test_loader

    def run(self, filename: str) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Execute the full data pipeline.

        Chains load_raw → clean → build_datasets → build_dataloaders
        as a single entry point for training scripts.

        Args:
            filename: Name of the raw CSV file to process.

        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        self.logger.info(f"Starting data pipeline for {filename}")

        df = self.load_raw(filename)
        cleaned_df = self.clean(df)
        train_dataset, val_dataset, test_dataset = self.build_datasets(cleaned_df)
        train_loader, val_loader, test_loader = self.build_dataloaders(
            train_dataset, val_dataset, test_dataset
        )

        self.logger.info("Data pipeline completed successfully")

        return train_loader, val_loader, test_loader

    def load_checkpoint(self, checkpoint_dir: str | pathlib.Path) -> "DataPipeline":
        """Load a DataPipeline from a checkpoint for inference.

        Reads the registry.json in the checkpoint directory to find the
        best checkpoint, then loads the tokenizer from the checkpoint.

        Args:
            checkpoint_dir: Path to the checkpoint directory containing
                registry.json and tokenizer/ subdirectory.

        Returns:
            DataPipeline instance with loaded tokenizer, ready for inference.

        Raises:
            FileNotFoundError: If registry.json or tokenizer directory
                does not exist.
            KeyError: If best_checkpoint entry is missing from registry.
        """
        checkpoint_path = pathlib.Path(checkpoint_dir)
        registry_path = checkpoint_path / "registry.json"

        if not registry_path.exists():
            raise FileNotFoundError(f"Registry file not found: {registry_path}")

        with open(registry_path, encoding="utf-8") as f:
            registry = json.load(f)

        best_checkpoint = registry.get("best_checkpoint")
        if best_checkpoint is None:
            raise KeyError("No 'best_checkpoint' entry found in registry.json")

        tokenizer_path = checkpoint_path / "tokenizer"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_path}")

        self.tokenizer = SentimentTokenizer.load(tokenizer_path)

        self.logger.info(
            f"Loaded checkpoint pipeline from {checkpoint_dir}, "
            f"best checkpoint: {best_checkpoint}"
        )

        return self
