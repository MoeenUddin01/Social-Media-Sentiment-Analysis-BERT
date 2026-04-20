#!/usr/bin/env python3
"""Test script to verify all imports work correctly."""

from __future__ import annotations

print("Testing imports...")

# Test data module
from src.data import SentimentDataset, TextPreprocessor, TweetCleaner
print("✅ src.data imports work")

# Test models module
from src.models import BertSentimentClassifier, ModelEvaluator, FineTuner, SentimentTokenizer
print("✅ src.models imports work")

# Test pipelines module
from src.pipelines import (
    EarlyStopping,
    ModelCheckpoint,
    DataPipeline,
    Evaluator,
    EvaluationPipeline,
    TrainingPipeline,
    get_optimizer,
    get_scheduler,
)
print("✅ src.pipelines imports work")

# Test utils module
from src.utils import (
    DagsHubLogger,
    get_logger,
    MetricsComputer,
    SentimentMetrics,
    get_device,
    seed_worker,
    set_seed,
    ResultVisualizer,
    TrainingVisualizer,
)
print("✅ src.utils imports work")

# Test direct imports
from src.pipelines.data_preprocessin import DataPipeline as DP
from src.models.tokenizer import SentimentTokenizer as ST
from src.utils.seed import get_device as gd
print("✅ Direct imports work")

print("\n🎉 All imports successful! Project is runnable.")
