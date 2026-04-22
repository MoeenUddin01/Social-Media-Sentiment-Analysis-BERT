"""Keggle-style beginner training script for BERT Sentiment Analysis.

This script provides a beginner-friendly, cell-by-cell approach to training
with clear progress bars and DAGs Hub integration for experiment tracking.

Simply run this in a notebook cell-by-cell fashion or as a script.

Charts shown in DAGs Hub experiment:
-------------------------------------
1. train/batch_loss - Live loss per batch (updates continuously)
2. train/batch_accuracy - Live accuracy per batch (updates continuously)
3. train/epoch_loss - Training loss at end of each epoch
4. train/epoch_accuracy - Training accuracy at end of each epoch
5. val/loss - Validation loss per epoch
6. val/accuracy - Validation accuracy per epoch
7. val/f1_macro - Macro F1 score per epoch
8. val/f1_weighted - Weighted F1 score per epoch
9. val/precision - Precision per epoch
10. val/recall - Recall per epoch
11. learning_rate - Learning rate schedule over time
12. confusion_matrices/ - Confusion matrix images per epoch
13. plots/confusion_matrix.png - Final confusion matrix
14. plots/training_curves.png - Training history visualization
15. plots/confidence_histogram.png - Prediction confidence distribution
16. plots/per_class_f1_bar.png - Per-class F1 scores bar chart
17. test/* metrics - Test set evaluation results

DAGs Hub Model Saving:
----------------------
The ModelCheckpoint callback is configured with `save_best_only=True` which means:
- Models are ONLY saved when validation accuracy improves
- If the current epoch's val_accuracy is NOT better than the best so far, no save
- This prevents wasting storage on worse-performing checkpoints

"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1: Setup - Install dependencies
# ═══════════════════════════════════════════════════════════════════════════════
# Uncomment and run if running on Keggle or fresh environment:
# !pip install torch transformers pandas scikit-learn pyyaml tqdm dagshub mlflow

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2: Imports
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import pathlib
import sys

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Add src to path
sys.path.insert(0, str(pathlib.Path().parent / "src"))

from src.data.dataset import SentimentDataset
from src.models.bert_classifier import BertSentimentClassifier
from src.models.tokenizer import SentimentTokenizer
from src.pipelines.callbacks import EarlyStopping, ModelCheckpoint
from src.pipelines.data_preprocessin import DataPipeline
from src.pipelines.model_training import TrainingPipeline
from src.pipelines.scheduler import get_scheduler
from src.utils.logger import DagsHubLogger, get_logger
from src.utils.seed import set_seed

print("✅ All imports successful!")
print(f"🖥️  PyTorch version: {torch.__version__}")
print(f"🖥️  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🖥️  CUDA device: {torch.cuda.get_device_name(0)}")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3: Configuration
# ═══════════════════════════════════════════════════════════════════════════════
# Load config
config_path = pathlib.Path("config.yaml")
with open(config_path, encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Update config for Keggle/beginner friendly settings
config["training"]["epochs"] = 10
config["training"]["num_epochs"] = 10
config["training"]["batch_size"] = 32
config["training"]["learning_rate"] = 2e-5
config["dagshub"]["enabled"] = True

# IMPORTANT: Set your DAGs Hub credentials here!
config["dagshub"]["repo_owner"] = "MoeenUddin01"  # ← Your DagsHub username
config["dagshub"]["repo_name"] = "Social-Media-Sentiment-Analysis-BERT"  # ← Your DagsHub repo name

print("📋 Configuration loaded:")
print(f"   Model: {config['model']['name']}")
print(f"   Epochs: {config['training']['epochs']}")
print(f"   Batch size: {config['training']['batch_size']}")
print(f"   Learning rate: {config['training']['learning_rate']}")
print(f"   DAGs Hub enabled: {config['dagshub']['enabled']}")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4: Set random seed for reproducibility
# ═══════════════════════════════════════════════════════════════════════════════
set_seed(42)
print("🎲 Random seed set to 42 for reproducibility")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5: Initialize tokenizer and data pipeline
# ═══════════════════════════════════════════════════════════════════════════════
print("🔤 Initializing tokenizer...")
tokenizer = SentimentTokenizer(
    model_name=config.get("model", {}).get("name", "bert-base-uncased")
)
print("✅ Tokenizer ready!")

print("\n📊 Setting up data pipeline...")
pipeline = DataPipeline(config, tokenizer)

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6: Load and process data (with progress display)
# ═══════════════════════════════════════════════════════════════════════════════
# Replace with your actual data file name
data_file = "tweets.csv"  # ← Change this to your data file in dataset/raw/

print(f"⏳ Loading and processing data from dataset/raw/{data_file}...")
print("   This may take a few minutes...")

# Run full data pipeline
train_loader, val_loader, test_loader = pipeline.run(data_file)

print("\n✅ Data pipeline complete!")
print(f"   📚 Training samples: {len(train_loader.dataset)}")
print(f"   ✅ Validation samples: {len(val_loader.dataset)}")
print(f"   🧪 Test samples: {len(test_loader.dataset)}")
print(f"   📦 Training batches: {len(train_loader)}")
print(f"   📦 Validation batches: {len(val_loader)}")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7: Initialize model
# ═══════════════════════════════════════════════════════════════════════════════
print("🤖 Initializing BERT model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Using device: {device}")

model = BertSentimentClassifier(
    num_labels=config["model"]["num_labels"],
).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"✅ Model loaded!")
print(f"   📊 Total parameters: {total_params:,}")
print(f"   🎯 Trainable parameters: {trainable_params:,}")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 8: Setup optimizer and scheduler
# ═══════════════════════════════════════════════════════════════════════════════
print("⚙️ Setting up optimizer and scheduler...")

optimizer = AdamW(
    model.parameters(),
    lr=config["training"]["learning_rate"],
    weight_decay=config["training"]["weight_decay"],
)

num_epochs = config["training"]["epochs"]
total_steps = len(train_loader) * num_epochs

scheduler = get_scheduler(
    optimizer=optimizer,
    num_training_steps=total_steps,
    config=config,
)

print(f"✅ Optimizer: AdamW (lr={config['training']['learning_rate']})")
print(f"✅ Scheduler: Linear with warmup ({config['training'].get('warmup_steps', 500)} steps)")
print(f"   Total training steps: {total_steps}")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 9: Setup callbacks and DAGs Hub logger
# ═══════════════════════════════════════════════════════════════════════════════
print("🔔 Setting up callbacks...")

# Early stopping - stops training if no improvement for 3 epochs
early_stopping = EarlyStopping(
    patience=3,
    min_delta=0.001,
    mode="max",
    monitor="val_accuracy",
)

# Model checkpoint - ONLY saves when validation accuracy improves
# save_best_only=True means: only save if current > best so far
checkpoint_callback = ModelCheckpoint(
    monitor="val_accuracy",  # Monitor validation accuracy
    mode="max",              # Higher is better
    save_best_only=True,     # ← KEY: Only save when metric improves!
    save_dir="artifacts/checkpoints",
)

print("✅ Early stopping: patience=3, monitor=val_accuracy")
print("✅ Model checkpoint: save_best_only=True (only saves when better!)")

# Initialize DAGs Hub logger if enabled
dagshub_logger = None
if config["dagshub"]["enabled"]:
    print("\n🔗 Initializing DAGs Hub logger...")
    try:
        dagshub_logger = DagsHubLogger(config)
        print("✅ DAGs Hub logger ready!")
        print(f"   Experiment: {config['dagshub']['experiment_name']}")
    except Exception as e:
        print(f"⚠️ Could not initialize DAGs Hub: {e}")
        print("   Continuing without DAGs Hub logging...")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 10: Initialize training pipeline
# ═══════════════════════════════════════════════════════════════════════════════
print("🏋️ Initializing training pipeline...")

# Use lightweight Evaluator from pipelines (avoids circular import)
from src.pipelines.evaluator import Evaluator

evaluator = Evaluator(model, device)

training_pipeline = TrainingPipeline(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    evaluator=evaluator,
    callbacks=[early_stopping, checkpoint_callback],
    device=device,
    config=config,
    dagshub_logger=dagshub_logger,
)

print("✅ Training pipeline ready!")
print("\n" + "=" * 70)
print("📊 Training will log these charts to DAGs Hub:")
print("   • train/batch_loss & train/batch_accuracy (live per batch)")
print("   • train/epoch_loss & train/epoch_accuracy (per epoch)")
print("   • val/loss, val/accuracy, val/f1_macro, val/f1_weighted (per epoch)")
print("   • val/precision, val/recall (per epoch)")
print("   • learning_rate (per epoch)")
print("   • confusion_matrices/ (images per epoch)")
print("   • plots/* (final evaluation charts)")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 11: TRAIN! (with live progress bars)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n🚀 STARTING TRAINING!\n")

history = training_pipeline.fit(num_epochs=num_epochs)

print("\n✅ Training complete!")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 12: Save all artifacts
# ═══════════════════════════════════════════════════════════════════════════════
print("💾 Saving training artifacts...")

checkpoint_dir = pathlib.Path("artifacts/checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Get timestamp for unique folder name
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = checkpoint_dir / f"run_{timestamp}"

# Save everything
training_pipeline.save_all_artifacts(
    checkpoint_dir=save_dir,
    run_metadata={"keggle_run": True, "timestamp": timestamp},
)

print(f"✅ All artifacts saved to: {save_dir}")
print("\n📁 Saved artifacts:")
print("   • model.pt - Best model weights")
print("   • tokenizer/ - Tokenizer files")
print("   • model_config.json - Model architecture")
print("   • label_map.json - Label mappings")
print("   • metrics.json - Training history")
print("   • run_info.json - Run metadata")
print("   • confusion_matrix.png, *.png - Evaluation plots")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 13: Final evaluation on test set
# ═══════════════════════════════════════════════════════════════════════════════
print("\n🧪 Running final evaluation on test set...")

from src.pipelines.model_evaluation import EvaluationPipeline

eval_pipeline = EvaluationPipeline(
    checkpoint_dir=save_dir,
    device=device,
)

test_metrics = eval_pipeline.run(test_loader)

print("\n" + "=" * 50)
print("📊 FINAL TEST RESULTS")
print("=" * 50)
print(f"   Accuracy:    {test_metrics['accuracy']:.4f}")
print(f"   Macro F1:    {test_metrics['macro_f1']:.4f}")
print(f"   Weighted F1: {test_metrics['weighted_f1']:.4f}")
print(f"   Precision:   {test_metrics['precision']:.4f}")
print(f"   Recall:      {test_metrics['recall']:.4f}")
print("=" * 50)

# Save test metrics to DAGs Hub
if dagshub_logger is not None:
    dagshub_logger.log_test_metrics(test_metrics)
    print("\n✅ Test metrics logged to DAGs Hub!")

# Save evaluation report
eval_pipeline.save_report(save_dir)
print(f"✅ Evaluation report saved to: {save_dir}")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 14: Summary
# ═══════════════════════════════════════════════════════════════════════════════
best_epoch = history["val_accuracy"].index(max(history["val_accuracy"]))
best_val_acc = max(history["val_accuracy"])
best_val_f1 = max(history["val_f1"])

print("\n" + "=" * 70)
print("🏆 TRAINING SUMMARY")
print("=" * 70)
print(f"   Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch + 1})")
print(f"   Best validation F1:       {best_val_f1:.4f}")
print(f"   Final test accuracy:      {test_metrics['accuracy']:.4f}")
print(f"   Total epochs trained:     {len(history['train_loss'])}")
print(f"   Early stopped:            {early_stopping.should_stop}")
print(f"   Checkpoint saved to:      {save_dir}")

if config["dagshub"]["enabled"] and dagshub_logger is not None:
    print(f"\n   📊 View experiment at:")
    print(f"   https://dagshub.com/{config['dagshub']['repo_owner']}/{config['dagshub']['repo_name']}/experiments")

print("=" * 70)
print("\n🎉 All done! Happy analyzing!")
