# BERT Sentiment Analysis Project - Architecture Documentation

## Project Overview

This project provides a complete pipeline for fine-tuning BERT models on social media sentiment classification tasks.

## Architecture

### Source Layout (`src/`)

- **`data/`** - Data loading and preprocessing
  - `loaders.py` - Dataset/DataLoader wrappers for HF datasets
  - `preprocessor.py` - Text cleaning, tokenization, augmentation
  - `raw/` - Original dataset storage
  - `processed/` - Preprocessed data ready for training

- **`models/`** - BERT model components
  - `bert_classifier.py` - BERT + classification head implementation
  - `fine_tuner.py` - HuggingFace Trainer wrapper
  - `tokenizer.py` - Tokenizer management utilities
  - `config.py` - Model configuration dataclass
  - `checkpoints/` - Saved model checkpoints

- **`training/`** - Training utilities
  - `trainer.py` - Custom PyTorch training loop
  - `evaluator.py` - Model evaluation metrics
  - `scheduler.py` - LR schedulers (linear, cosine, warmup)
  - `callbacks.py` - Early stopping, checkpointing, logging

- **`inference/`** - Inference and serving
  - `predictor.py` - Single/batch prediction interface
  - `pipeline.py` - End-to-end inference pipeline
  - `batch_infer.py` - Batch processing utilities
  - `api_server.py` - FastAPI serving endpoint

- **`utils/`** - Shared utilities
  - `metrics.py` - Accuracy, F1, confusion matrix
  - `visualizer.py` - Training curves, plots
  - `logger.py` - Structured logging setup
  - `seed.py` - Reproducibility helpers

## Configuration System

YAML-based configuration in `config/`:
- `base.yaml` - Shared defaults
- `train.yaml` - Training hyperparameters
- `infer.yaml` - Inference settings
- `model.yaml` - Model architecture
- `logging.yaml` - Logging configuration

## Code Conventions

- All Python files: module docstring + `from __future__ import annotations`
- Type hints required for all function signatures
- Google-style docstrings with Args, Returns, Raises sections
- Maximum line length: 88 characters
- File size: 300-500 lines (split if larger)
