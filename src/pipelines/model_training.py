"""Training pipeline for BERT sentiment classification.

Provides the TrainingPipeline class for end-to-end model training with
callbacks, checkpointing, and artifact management.
"""

from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING, Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.models.bert_classifier import BertSentimentClassifier
from src.models.evaluator import ModelEvaluator
from src.models.tokenizer import SentimentTokenizer
from src.pipelines.callbacks import EarlyStopping, ModelCheckpoint
from src.pipelines.evaluator import Evaluator
from src.utils.logger import DagsHubLogger, get_logger

if TYPE_CHECKING:
    from logging import Logger

    from torch.optim.lr_scheduler import LambdaLR

    from src.models.bert_classifier import BertSentimentClassifier


class TrainingPipeline:
    """End-to-end training pipeline for BERT sentiment classification.

    Orchestrates the full training workflow: training epochs, validation,
    callbacks (early stopping, checkpointing), and artifact management.
    Integrates with DagsHub for live experiment tracking.

    Attributes:
        model: BERT sentiment classifier model.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: AdamW optimizer.
        scheduler: Learning rate scheduler with warmup.
        evaluator: ModelEvaluator for validation metrics.
        callbacks: List of callbacks (EarlyStopping, ModelCheckpoint).
        device: Device to run training on.
        config: Training configuration dictionary.
        logger: Logger instance.
        dagshub_logger: DagsHubLogger for MLflow tracking.
        history: Training history dictionary.
        best_metrics: Best validation metrics seen.
    """

    def __init__(
        self,
        model: BertSentimentClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: AdamW,
        scheduler: "LambdaLR",
        evaluator: ModelEvaluator,
        callbacks: list[EarlyStopping | ModelCheckpoint],
        device: torch.device,
        config: dict[str, Any],
        dagshub_logger: DagsHubLogger | None = None,
    ) -> None:
        """Initialize the TrainingPipeline.

        Args:
            model: BERT sentiment classifier model.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            optimizer: AdamW optimizer for parameter updates.
            scheduler: Learning rate scheduler with warmup.
            evaluator: ModelEvaluator for computing validation metrics.
            callbacks: List of callback objects (EarlyStopping, ModelCheckpoint).
            device: Device to run training on (CPU or CUDA).
            config: Training configuration dictionary.
            dagshub_logger: DagsHubLogger for MLflow experiment tracking.
                If None, DagsHub logging is disabled.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.callbacks = callbacks
        self.device = device
        self.config = config
        self.logger: Logger = get_logger(__name__)
        self.dagshub_logger = dagshub_logger

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
        }
        self.best_metrics: dict[str, float] = {}
        self._criterion = torch.nn.CrossEntropyLoss()
        self._total_batches = len(train_loader)

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Train for one epoch.

        Logs batch metrics to DagsHub every batch for live chart updates.

        Args:
            epoch: Current epoch number (0-indexed).

        Returns:
            Dictionary with train_loss and train_accuracy for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            logits = self.model(inputs)
            loss = self._criterion(logits, labels)

            loss.backward()

            training_config = self.config.get("training", {})
            gradient_clip = training_config.get("gradient_clip", 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)

            self.optimizer.step()
            self.scheduler.step()

            # Calculate batch metrics
            batch_loss = loss.item()
            predictions = torch.argmax(logits, dim=-1)
            batch_correct = (predictions == labels).sum().item()
            batch_total = labels.size(0)
            batch_accuracy = batch_correct / batch_total if batch_total > 0 else 0.0

            total_loss += batch_loss
            correct += batch_correct
            total += batch_total

            # Log batch metrics to DagsHub
            if self.dagshub_logger is not None:
                self.dagshub_logger.log_batch_metrics(
                    batch=batch_idx,
                    epoch=epoch,
                    loss=batch_loss,
                    accuracy=batch_accuracy,
                    total_batches=self._total_batches,
                )

        avg_loss = (
            total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0.0
        )
        accuracy = correct / total if total > 0 else 0.0

        return {"train_loss": avg_loss, "train_accuracy": accuracy}

    def validate(
        self, epoch: int, train_metrics: dict[str, float] | None = None
    ) -> dict[str, float]:
        """Validate the model on validation set.

        Uses Evaluator.full_eval() for metric computation. Logs epoch metrics
        to DagsHub after validation completes.

        Args:
            epoch: Current epoch number (0-indexed).
            train_metrics: Training metrics from current epoch to log alongside
                validation metrics. Defaults to None.

        Returns:
            Dictionary with val_loss, val_accuracy, and val_f1.
        """
        # Use Evaluator for validation metrics
        evaluator = Evaluator(self.model, self.device)
        metrics = evaluator.full_eval(self.val_loader)

        # Compute validation loss separately
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
                logits = self.model(inputs)
                loss = self._criterion(logits, labels)
                total_loss += loss.item()

        avg_loss = (
            total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
        )

        val_metrics = {
            "val_loss": avg_loss,
            "val_accuracy": metrics["accuracy"],
            "val_f1": metrics["macro_f1"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics.get("weighted_f1", 0.0),
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
        }

        # Log epoch metrics to DagsHub
        if self.dagshub_logger is not None and train_metrics is not None:
            # Get current learning rate from scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.dagshub_logger.log_epoch_metrics(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                learning_rate=current_lr,
            )

            # Log confusion matrix if available
            if "confusion_matrix" in metrics:
                label_names = ["negative", "neutral", "positive"]
                self.dagshub_logger.log_confusion_matrix(
                    cm=metrics["confusion_matrix"],
                    label_names=label_names,
                    step=epoch,
                )

        return val_metrics

    def fit(self, num_epochs: int) -> dict[str, list[float]]:
        """Run the full training loop.

        Initializes DagsHub logging if enabled, trains for specified epochs,
        and logs final metrics.

        Args:
            num_epochs: Number of epochs to train for.

        Returns:
            Full training history dictionary.
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")

        # Start DagsHub logging if enabled
        if self.dagshub_logger is not None:
            self.dagshub_logger.start_run()
            self.dagshub_logger.log_config(self.config)
            self.logger.info("DagsHub logging started")

        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch, train_metrics=train_metrics)

            combined_metrics = {**train_metrics, **val_metrics}

            self.history["train_loss"].append(train_metrics["train_loss"])
            self.history["train_accuracy"].append(train_metrics["train_accuracy"])
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_accuracy"].append(val_metrics["val_accuracy"])
            self.history["val_f1"].append(val_metrics["val_f1"])

            self.logger.info(
                f"Train loss: {train_metrics['train_loss']:.4f}, "
                f"accuracy: {train_metrics['train_accuracy']:.4f} | "
                f"Val loss: {val_metrics['val_loss']:.4f}, "
                f"accuracy: {val_metrics['val_accuracy']:.4f}, "
                f"f1: {val_metrics['val_f1']:.4f}"
            )

            should_stop = False
            for callback in self.callbacks:
                if isinstance(callback, EarlyStopping):
                    if callback(epoch, combined_metrics):
                        should_stop = True
                        self.logger.info(
                            f"Early stopping triggered at epoch {epoch + 1}"
                        )
                        break
                elif isinstance(callback, ModelCheckpoint):
                    if callback(epoch, combined_metrics):
                        is_best = (
                            callback.should_save
                            and callback.best_metric
                            == val_metrics.get(
                                callback.monitor.replace("val_", "")
                                if "val_" in callback.monitor
                                else callback.monitor,
                                0,
                            )
                        )
                        self.save_checkpoint(epoch, combined_metrics, is_best=is_best)

            if should_stop:
                break

        self.logger.info("Training complete")

        # End DagsHub logging if enabled
        if self.dagshub_logger is not None:
            self.dagshub_logger.end_run()

        return self.history

    def save_checkpoint(
        self,
        epoch: int,
        metrics: dict[str, float],
        is_best: bool = False,
    ) -> pathlib.Path:
        """Save model checkpoint.

        Args:
            epoch: Current epoch number.
            metrics: Dictionary of current metrics.
            is_best: Whether this is the best model so far.

        Returns:
            Path to saved checkpoint directory.
        """
        training_config = self.config.get("training", {})
        checkpoint_dir = pathlib.Path(
            training_config.get("checkpoint_dir", "artifacts/checkpoints")
        )
        run_name = f"run_epoch{epoch + 1}"
        save_dir = checkpoint_dir / run_name
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = save_dir / "checkpoint.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
            },
            checkpoint_path,
        )

        if is_best:
            best_path = checkpoint_dir / "model.pt"
            torch.save(self.model.state_dict(), best_path)
            self.logger.info(f"Saved best model to {best_path}")

        registry_path = checkpoint_dir / "registry.json"
        registry: dict[str, Any] = {"checkpoints": []}
        if registry_path.exists():
            with open(registry_path, encoding="utf-8") as f:
                registry = json.load(f)

        checkpoint_entry = {
            "epoch": epoch + 1,
            "path": str(save_dir),
            "metrics": metrics,
        }
        registry["checkpoints"].append(checkpoint_entry)

        if is_best or "best_checkpoint" not in registry:
            registry["best_checkpoint"] = checkpoint_entry

        top_k = training_config.get("top_k_checkpoints", 3)
        if len(registry["checkpoints"]) > top_k:
            registry["checkpoints"] = registry["checkpoints"][-top_k:]

        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)

        self.logger.info(f"Saved checkpoint to {save_dir}")
        return save_dir

    def save_all_artifacts(
        self,
        checkpoint_dir: pathlib.Path | str,
        preprocessor_config: dict[str, Any] | None = None,
        class_weights: list[float] | None = None,
        run_metadata: dict[str, Any] | None = None,
        test_metrics: dict[str, float] | None = None,
    ) -> pathlib.Path:
        """Save all training artifacts for prediction and reproducibility.

        Saves model weights, tokenizer, configs, and metadata needed for
        inference and training reproduction.

        Args:
            checkpoint_dir: Directory to save artifacts to.
            preprocessor_config: Preprocessor settings used during training.
                Defaults to None (uses default settings).
            class_weights: Class weights for imbalanced data. Defaults to None.
            run_metadata: Additional run metadata (timestamp, seed, device, etc).
                Defaults to None.
            test_metrics: Test set metrics to log to DagsHub. Defaults to None.

        Returns:
            Path where artifacts were saved.
        """
        checkpoint_path = pathlib.Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # 1. Save model.pt - model weights
        model_path = checkpoint_path / "model.pt"
        torch.save(self.model.state_dict(), model_path)

        # 2. Save tokenizer/ directory
        tokenizer_path = checkpoint_path / "tokenizer"
        tokenizer_path.mkdir(parents=True, exist_ok=True)
        tokenizer = SentimentTokenizer(
            model_name=self.config.get("model", {}).get("name", "bert-base-uncased")
        )
        tokenizer.save(tokenizer_path)

        # 3. Save model_config.json - architecture config
        model_config = {
            "model_name": self.config.get("model", {}).get("name", "bert-base-uncased"),
            "num_labels": self.config.get("model", {}).get("num_labels", 3),
            "max_length": self.config.get("training", {}).get("max_length", 128),
            "dropout": self.config.get("model", {}).get("dropout", 0.3),
            "hidden_size": 768,  # bert-base-uncased default
        }
        model_config_path = checkpoint_path / "model_config.json"
        with open(model_config_path, "w", encoding="utf-8") as f:
            json.dump(model_config, f, indent=2)

        # 4. Save label_map.json - bidirectional mapping
        label_map = {
            "int_to_label": {"0": "negative", "1": "neutral", "2": "positive"},
            "label_to_int": {"negative": 0, "neutral": 1, "positive": 2},
        }
        label_map_path = checkpoint_path / "label_map.json"
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump(label_map, f, indent=2)

        # 5. Save preprocessor_config.json
        if preprocessor_config is None:
            preprocessor_config = {
                "remove_urls": True,
                "remove_mentions": True,
                "remove_hashtags": False,
                "lowercase": True,
                "handle_emojis": True,
            }
        preprocessor_config_path = checkpoint_path / "preprocessor_config.json"
        with open(preprocessor_config_path, "w", encoding="utf-8") as f:
            json.dump(preprocessor_config, f, indent=2)

        # 6. Save class_weights.json
        if class_weights is None:
            class_weights = [1.0, 1.0, 1.0]  # Default equal weights
        class_weights_path = checkpoint_path / "class_weights.json"
        with open(class_weights_path, "w", encoding="utf-8") as f:
            json.dump({"weights": class_weights}, f, indent=2)

        # 7. Save train_config.yaml - copy of config at training time
        import shutil
        import yaml

        train_config_path = checkpoint_path / "train_config.yaml"
        with open(train_config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False)

        # 8. Save metrics.json - per-epoch metrics
        metrics_path = checkpoint_path / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        # 9. Save run_info.json - run metadata
        from datetime import datetime

        if run_metadata is None:
            run_metadata = {}

        best_epoch = 0
        if self.history["val_accuracy"]:
            best_epoch = self.history["val_accuracy"].index(
                max(self.history["val_accuracy"])
            )

        run_info = {
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            "run_name": checkpoint_path.name,
            "seed": self.config.get("training", {}).get("seed", 42),
            "device": str(self.device),
            "train_size": len(self.train_loader.dataset) if hasattr(self.train_loader, "dataset") else 0,
            "val_size": len(self.val_loader.dataset) if hasattr(self.val_loader, "dataset") else 0,
            "best_epoch": best_epoch,
            "early_stopped": False,
            "final_val_f1": max(self.history["val_f1"]) if self.history["val_f1"] else 0.0,
        }
        run_info.update(run_metadata)
        run_info_path = checkpoint_path / "run_info.json"
        with open(run_info_path, "w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=2)

        # Save evaluation report (plots)
        self.evaluator.save_report(checkpoint_path)

        # Update registry
        self._update_registry(checkpoint_path, run_info, label_map)

        # Log artifacts to DagsHub if enabled
        if self.dagshub_logger is not None:
            self.dagshub_logger.log_plots(checkpoint_path)
            self.dagshub_logger.log_model_artifact(checkpoint_path)

            if test_metrics is not None:
                self.dagshub_logger.log_test_metrics(test_metrics)

            self.logger.info("Logged all artifacts to DagsHub")

        self.logger.info(f"Saved all artifacts to {checkpoint_path}")
        return checkpoint_path

    def _update_registry(
        self,
        checkpoint_path: pathlib.Path,
        run_info: dict[str, Any],
        label_map: dict[str, dict],
    ) -> None:
        """Update the checkpoint registry with new run information.

        Args:
            checkpoint_path: Path to the checkpoint directory.
            run_info: Run metadata dictionary.
            label_map: Label mapping dictionary.
        """
        registry_path = pathlib.Path("src/models/checkpoints/registry.json")
        registry_path.parent.mkdir(parents=True, exist_ok=True)

        registry: dict[str, Any] = {"best_checkpoint": "", "checkpoints": []}
        if registry_path.exists():
            with open(registry_path, encoding="utf-8") as f:
                registry = json.load(f)

        checkpoint_entry = {
            "run_name": checkpoint_path.name,
            "checkpoint_dir": str(checkpoint_path),
            "val_f1": run_info.get("final_val_f1", 0.0),
            "best_epoch": run_info.get("best_epoch", 0),
            "timestamp": run_info.get("timestamp", ""),
            "artifacts": [
                "model.pt",
                "tokenizer/",
                "model_config.json",
                "label_map.json",
                "preprocessor_config.json",
                "train_config.yaml",
                "metrics.json",
                "run_info.json",
            ],
        }
        registry["checkpoints"].append(checkpoint_entry)

        # Update best checkpoint if this is the best so far
        if not registry["best_checkpoint"] or checkpoint_entry["val_f1"] >= max(
            cp["val_f1"] for cp in registry["checkpoints"]
        ):
            registry["best_checkpoint"] = checkpoint_path.name

        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
