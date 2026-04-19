# BERT Sentiment Analysis Project - Architecture Documentation

## Project Overview

BERT-based sentiment analysis on social media using PyTorch and HuggingFace Transformers.
Fine-tune BERT models for multi-class sentiment classification on tweets and social posts.

## Architecture

### Root Files

| File | Responsibility |
|------|--------------|
| `main.py` | Entry stub (replace with orchestration logic) |
| `pyproject.toml` | Dependencies: torch, transformers, fastapi, uvicorn, wandb |
| `config.ymal` → `config.yaml` | Single YAML configuration file |
| `README.md` | Project documentation |
| `.env.example` | Template for API keys (WANDB, Twitter, HuggingFace) |

### src/data/ — Data Loading and Preprocessing

| File | Responsibility |
|------|--------------|
| `__init__.py` | Exports: DataLoaderFactory, SentimentDataset, TextPreprocessor, DataAugmenter, TweetCleaner |
| `loaders.py` | Dataset/DataLoader factory for PyTorch |
| `cleaner.py` | `TweetCleaner` class: remove URLs, mentions, hashtags, special chars, normalize whitespace |
| `dataset.py` | PyTorch Dataset definitions (currently stub) |
| `preprocessor.py` | Text preprocessing transforms and augmentation |

### src/models/ — BERT Model Implementation

| File | Responsibility |
|------|--------------|
| `__init__.py` | Exports: BERTClassifier, BERTFineTuner, BERTTokenizer |
| `bert_classifier.py` | BERT + classification head architecture |
| `fine_tuner.py` | HuggingFace Trainer wrapper for fine-tuning |
| `tokenizer.py` | Tokenizer management and encoding utilities |
| `train.py` | Training step implementation (currently stub) |
| `checkpoints/` | Saved model weights (.gitkeep present) |

### src/pipelines/ — Training and Evaluation Orchestration

| File | Responsibility |
|------|--------------|
| `__init__.py` | Exports: DataPipeline, Trainer, Evaluator, Scheduler, Callbacks |
| `data_preprocessin.py` → `data_preprocessing.py` | Data pipeline orchestration |
| `model_training.py` | Full training pipeline (currently stub) |
| `model_evaluation.py` | Evaluation pipeline (currently stub) |
| `callbacks.py` | Early stopping, model checkpointing |
| `scheduler.py` | Learning rate scheduling (linear, cosine, warmup) |
| `evaluator.py` | Metrics computation wrapper |

### src/utils/ — Shared Utilities

| File | Responsibility |
|------|--------------|
| `__init__.py` | Exports: Metrics, Visualizer, Logger, set_seed |
| `metrics.py` | Accuracy, F1 score, confusion matrix |
| `visualizer.py` | Training curves, confusion matrix plots |
| `logger.py` | Structured logging with wandb integration |
| `seed.py` | Reproducibility: `set_seed()` for deterministic runs |

## Conventions

| Rule | Requirement |
|------|-------------|
| Docstrings | Google-style with Args, Returns, Raises sections |
| Type hints | Required on all function signatures |
| Magic numbers | **Forbidden** — all values in `config.yaml` only |
| Random seed | `set_seed()` required before any training run |
| Line length | Maximum 88 characters |
| Imports | Three blocks: stdlib, third-party, internal (separated by blank lines) |
| Module docstring | Every file must have module-level docstring |
| Future annotations | `from __future__ import annotations` at top of every file |

## Data Flow

```
                  dataset/raw/
                  (CSV/JSON)
                       ↓
    ┌──────────────────┴──────────────────┐
    │  DataPipeline                         │
    │  src/pipelines/data_preprocessin.py     │
    └──────────────────┬──────────────────┘
                       ↓
               dataset/processed/
               (cleaned, tokenized)
                       ↓
    ┌──────────────────┴──────────────────┐
    │  SentimentDataset                     │
    │  src/data/dataset.py                  │
    └──────────────────┬──────────────────┘
                       ↓
    ┌──────────────────┴──────────────────┐
    │  DataLoader                           │
    │  src/data/loaders.py                  │
    └──────────────────┬──────────────────┘
                       ↓
    ┌──────────────────┴──────────────────┐
    │  BERTClassifier                       │
    │  src/models/bert_classifier.py        │
    └───────────────────────────────────────┘
```

## DagsHub/MLflow Logging Flow

```
    TrainingPipeline.fit()
           ↓
    dagshub_logger.start_run()
           ↓
    ┌──────────────────────────────────────┐
    │  Epoch Loop                           │
    │   ↓                                  │
    │  train_epoch()                        │
    │   ↓                                  │
    │  ┌─────────────────────────────┐     │
    │  │  Batch Loop                  │     │
    │  │   ↓                          │     │
    │  │  log_batch_metrics() ────────┼─────┼──→ DagsHub (live charts)
    │  │     train/batch_loss         │     │
    │  │     train/batch_accuracy     │     │
    │  │   (step = epoch*batches+batch)│     │
    │  └─────────────────────────────┘     │
    │   ↓                                  │
    │  validate()                           │
    │   ↓                                  │
    │  log_epoch_metrics() ────────────────┼──→ DagsHub
    │     train/epoch_loss                 │
    │     val/loss                         │
    │     val/f1_macro                     │
    │     learning_rate                    │
    │   ↓                                  │
    │  log_confusion_matrix() ─────────────┼──→ DagsHub artifacts
    └──────────────────────────────────────┘
           ↓
    dagshub_logger.end_run()
           ↓
    ┌──────────────────────────────────────┐
    │  save_all_artifacts()                 │
    │   ↓                                  │
    │  log_plots() ────────────────────────┼──→ DagsHub
    │  log_model_artifact() ───────────────┼──→ DagsHub
    │  log_test_metrics() ─────────────────┼──→ DagsHub
    └──────────────────────────────────────┘
```

**Note:** `log_batch_metrics()` is the source of live updating charts that refresh every batch during training.

## Common Pitfalls

| Issue | Solution |
|-------|----------|
| **config.ymal typo** | File is named `config.ymal` — rename to `config.yaml` |
| **Checkpoints path** | Use `src/models/checkpoints/` (exists with `.gitkeep`) |
| **Class imbalance** | Social media datasets skew negative — use weighted loss or oversampling |
| **Empty stubs** | `dataset.py`, `train.py`, `data_preprocessin.py`, `model_training.py`, `model_evaluation.py` are empty — implement before use |
| **Import paths** | Always import from package root: `from src.data import TweetCleaner` not `from src.data.cleaner import TweetCleaner` |
| **Reproducibility** | Call `set_seed()` from `src.utils` before every training run |
| **Whitespace after cleaning** | Chain cleaners then call `normalize_whitespace()` last to avoid double spaces |
