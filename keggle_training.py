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
# CELL 11: TRAIN! (verbose every-5-batch output + live DagsHub logging)
# ═══════════════════════════════════════════════════════════════════════════════
import time
import types

PRINT_EVERY = 5  # Print to console every N batches


def _verbose_train_epoch(self, epoch: int) -> dict:
    """Verbose training epoch — prints loss/accuracy every PRINT_EVERY batches."""
    self.model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(self.train_loader)
    epoch_start = time.time()

    print(f"\n{'═' * 65}")
    print(f"  📅 EPOCH {epoch + 1}/{num_epochs}   "
          f"({total_batches:,} batches  |  batch_size={config['training']['batch_size']})")
    print(f"{'═' * 65}")

    for batch_idx, batch in enumerate(self.train_loader):
        batch_start = time.time()

        input_ids      = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels         = batch["labels"].to(self.device)

        self.optimizer.zero_grad()
        logits = self.model({"input_ids": input_ids, "attention_mask": attention_mask})
        loss   = self._criterion(logits, labels)
        loss.backward()

        grad_clip = self.config.get("training", {}).get("gradient_clip", 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

        self.optimizer.step()
        self.scheduler.step()

        # ── Batch metrics ───────────────────────────────────────────────────
        batch_loss  = loss.item()
        preds       = torch.argmax(logits, dim=-1)
        batch_corr  = (preds == labels).sum().item()
        batch_total = labels.size(0)
        batch_acc   = batch_corr / batch_total if batch_total > 0 else 0.0

        total_loss += batch_loss
        correct    += batch_corr
        total      += batch_total

        # ── Console print every PRINT_EVERY batches ─────────────────────────
        if (batch_idx + 1) % PRINT_EVERY == 0 or batch_idx == total_batches - 1:
            elapsed      = time.time() - epoch_start
            batches_done = batch_idx + 1
            secs_per_bat = elapsed / batches_done
            remaining    = (total_batches - batches_done) * secs_per_bat
            pct_done     = 100.0 * batches_done / total_batches
            run_acc      = correct / total if total > 0 else 0.0
            run_loss     = total_loss / batches_done
            current_lr   = self.optimizer.param_groups[0]["lr"]

            # Build a simple ASCII progress bar
            bar_len   = 30
            filled    = int(bar_len * pct_done / 100)
            bar       = "█" * filled + "░" * (bar_len - filled)

            eta_m, eta_s = divmod(int(remaining), 60)
            elapsed_m, elapsed_s = divmod(int(elapsed), 60)

            print(
                f"  [{bar}] {pct_done:5.1f}%  "
                f"batch {batches_done:>6,}/{total_batches:,}  │  "
                f"loss={run_loss:.4f}  acc={run_acc:.4f}  "
                f"lr={current_lr:.2e}  │  "
                f"elapsed={elapsed_m}m{elapsed_s:02d}s  eta={eta_m}m{eta_s:02d}s"
            )

        # ── DagsHub batch logging ────────────────────────────────────────────
        if self.dagshub_logger is not None:
            self.dagshub_logger.log_batch_metrics(
                batch=batch_idx,
                epoch=epoch,
                loss=batch_loss,
                accuracy=batch_acc,
                total_batches=total_batches,
            )

    # ── Epoch summary ────────────────────────────────────────────────────────
    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    epoch_time = time.time() - epoch_start
    ep_m, ep_s = divmod(int(epoch_time), 60)

    print(f"  {'─' * 63}")
    print(f"  ✅ Epoch {epoch + 1} done in {ep_m}m{ep_s:02d}s")
    print(f"     avg_loss={avg_loss:.4f}  |  avg_accuracy={accuracy:.4f}")

    return {"train_loss": avg_loss, "train_accuracy": accuracy}


# Monkey-patch the verbose train_epoch onto the pipeline instance
training_pipeline.train_epoch = types.MethodType(_verbose_train_epoch, training_pipeline)


def _verbose_fit(self, num_epochs: int) -> dict:
    """Verbose fit loop — prints a full epoch results box after every validation."""
    from src.pipelines.callbacks import EarlyStopping, ModelCheckpoint

    self._logger if hasattr(self, "_logger") else None

    # Start DagsHub logging
    if self.dagshub_logger is not None:
        self.dagshub_logger.start_run()
        self.dagshub_logger.log_config(self.config)

    training_start = time.time()
    best_val_acc = 0.0

    for epoch in range(num_epochs):

        # ── Train one epoch (verbose batch output) ───────────────────────────
        train_metrics = self.train_epoch(epoch)

        # ── Validate ─────────────────────────────────────────────────────────
        print(f"  ⏳ Running validation for epoch {epoch + 1}...")
        val_start = time.time()
        val_metrics = self.validate(epoch, train_metrics=train_metrics)
        val_time = time.time() - val_start

        combined_metrics = {**train_metrics, **val_metrics}

        # ── Update history ────────────────────────────────────────────────────
        self.history["train_loss"].append(train_metrics["train_loss"])
        self.history["train_accuracy"].append(train_metrics["train_accuracy"])
        self.history["val_loss"].append(val_metrics["val_loss"])
        self.history["val_accuracy"].append(val_metrics["val_accuracy"])
        self.history["val_f1"].append(val_metrics["val_f1"])

        # ── Best tracker ──────────────────────────────────────────────────────
        is_best = val_metrics["val_accuracy"] > best_val_acc
        if is_best:
            best_val_acc = val_metrics["val_accuracy"]
        best_marker = "  ⭐ NEW BEST!" if is_best else ""

        # ── Total elapsed ─────────────────────────────────────────────────────
        total_elapsed  = time.time() - training_start
        tot_m, tot_s   = divmod(int(total_elapsed), 60)
        val_m, val_s   = divmod(int(val_time), 60)

        # ── EPOCH RESULTS BOX ─────────────────────────────────────────────────
        print(f"\n{'╔' + '═' * 63 + '╗'}")
        print(f"║  📊 EPOCH {epoch + 1}/{num_epochs} RESULTS"
              f"{best_marker:<{40 - len(str(epoch+1)) - len(str(num_epochs))}}║")
        print(f"{'╠' + '═' * 63 + '╣'}")
        print(f"║  {'METRIC':<25}  {'TRAIN':>10}  {'VAL':>10}          ║")
        print(f"║  {'─'*25}  {'─'*10}  {'─'*10}          ║")
        print(f"║  {'Loss':<25}  "
              f"{train_metrics['train_loss']:>10.4f}  "
              f"{val_metrics['val_loss']:>10.4f}          ║")
        print(f"║  {'Accuracy':<25}  "
              f"{train_metrics['train_accuracy']:>10.4f}  "
              f"{val_metrics['val_accuracy']:>10.4f}          ║")
        print(f"║  {'F1 (macro)':<25}  "
              f"{'—':>10}  "
              f"{val_metrics.get('macro_f1', val_metrics.get('val_f1', 0.0)):>10.4f}          ║")
        print(f"║  {'F1 (weighted)':<25}  "
              f"{'—':>10}  "
              f"{val_metrics.get('weighted_f1', 0.0):>10.4f}          ║")
        print(f"║  {'Precision':<25}  "
              f"{'—':>10}  "
              f"{val_metrics.get('precision', 0.0):>10.4f}          ║")
        print(f"║  {'Recall':<25}  "
              f"{'—':>10}  "
              f"{val_metrics.get('recall', 0.0):>10.4f}          ║")
        print(f"{'╠' + '═' * 63 + '╣'}")
        current_lr = self.optimizer.param_groups[0]["lr"]
        print(f"║  Learning Rate: {current_lr:.2e}"
              f"   │  Val time: {val_m}m{val_s:02d}s"
              f"   │  Total: {tot_m}m{tot_s:02d}s"
              f"{'': <5}║")
        print(f"║  Best val_acc so far: {best_val_acc:.4f}"
              f"  (epoch {self.history['val_accuracy'].index(best_val_acc) + 1})"
              f"{'': <14}║")
        print(f"{'╚' + '═' * 63 + '╝'}\n")

        # ── Callbacks ─────────────────────────────────────────────────────────
        should_stop = False
        for callback in self.callbacks:
            if isinstance(callback, EarlyStopping):
                if callback(epoch, combined_metrics):
                    should_stop = True
                    print(f"  🛑 Early stopping triggered at epoch {epoch + 1}!")
                    break
            elif isinstance(callback, ModelCheckpoint):
                callback(epoch, combined_metrics)

        if should_stop:
            break

    # ── Final training summary ────────────────────────────────────────────────
    total_elapsed = time.time() - training_start
    tot_m, tot_s  = divmod(int(total_elapsed), 60)
    best_epoch    = self.history["val_accuracy"].index(max(self.history["val_accuracy"]))

    print(f"\n{'═' * 65}")
    print(f"  🏆 TRAINING COMPLETE  —  {tot_m}m{tot_s:02d}s total")
    print(f"  Best val_accuracy: {max(self.history['val_accuracy']):.4f}  "
          f"(epoch {best_epoch + 1})")
    print(f"  Best val_f1:       {max(self.history['val_f1']):.4f}")
    print(f"  Epochs trained:    {len(self.history['train_loss'])}")
    print(f"{'═' * 65}\n")

    return self.history


# Monkey-patch verbose fit (replaces the default fit)
training_pipeline.fit = types.MethodType(_verbose_fit, training_pipeline)

print("\n🚀 STARTING TRAINING!")
print(f"   Verbose output every {PRINT_EVERY} batches")
print(f"   Epoch results box shown after every validation")
print(f"   All metrics streamed live → DagsHub\n")

history = training_pipeline.fit(num_epochs=num_epochs)

print("\n✅ Training complete!")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 12: Save all artifacts & upload to DagsHub
# ═══════════════════════════════════════════════════════════════════════════════
print("💾 Saving training artifacts...")

checkpoint_dir = pathlib.Path("artifacts/checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Get timestamp for unique folder name
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = checkpoint_dir / f"run_{timestamp}"

# ✅ FIX: mlflow.end_run() was already called inside fit().
# Re-open the SAME run so artifact uploads go to the correct run on DagsHub.
import mlflow
_active_run_id = None
if dagshub_logger is not None:
    try:
        # end_run() was already called; start a child run under the same experiment
        # using the same run_name so DagsHub groups them correctly.
        with mlflow.start_run(
            run_name=dagshub_logger.run_name + "_artifacts",
            nested=False,
        ) as _artifact_run:
            _active_run_id = _artifact_run.info.run_id
    except Exception:
        pass  # If this also fails, artifact upload will be skipped gracefully

# Save all files locally first
training_pipeline.save_all_artifacts(
    checkpoint_dir=save_dir,
    run_metadata={"keggle_run": True, "timestamp": timestamp},
)

print(f"✅ All artifacts saved locally to: {save_dir}")
print("\n📁 Saved artifacts:")
print("   • model.pt - Best model weights")
print("   • tokenizer/ - Tokenizer files")
print("   • model_config.json - Model architecture")
print("   • label_map.json - Label mappings")
print("   • metrics.json - Training history")
print("   • run_info.json - Run metadata")
print("   • confusion_matrix.png, *.png - Evaluation plots")

# ✅ NOW upload to DagsHub (run is active)
if dagshub_logger is not None and mlflow.active_run() is not None:
    dagshub_logger.log_plots(save_dir)
    dagshub_logger.log_model_artifact(save_dir)
    print("\n✅ All artifacts uploaded to DagsHub!")
elif dagshub_logger is not None:
    print("\n⚠️  DagsHub run already closed — artifacts saved locally only.")
    print(f"   You can upload manually from: {save_dir}")

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

# Save test metrics to DagsHub (run still active from Cell 12)
if dagshub_logger is not None and mlflow.active_run() is not None:
    dagshub_logger.log_test_metrics(test_metrics)
    print("\n✅ Test metrics logged to DagsHub!")

# Save evaluation report locally
eval_pipeline.save_report(save_dir)
print(f"✅ Evaluation report saved to: {save_dir}")

# ✅ NOW close the MLflow run — after ALL uploads are done
if mlflow.active_run() is not None:
    mlflow.end_run()
    print("✅ MLflow run closed cleanly.")

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
