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
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from logging import Logger

    from torch.optim.lr_scheduler import LambdaLR

    from src.models.bert_classifier import BertSentimentClassifier


class TrainingPipeline:
    """End-to-end training pipeline for BERT sentiment classification.

    Orchestrates the full training workflow: training epochs, validation,
    callbacks (early stopping, checkpointing), and artifact management.

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

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
        }
        self.best_metrics: dict[str, float] = {}
        self._criterion = torch.nn.CrossEntropyLoss()

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Train for one epoch.

        Args:
            epoch: Current epoch number (0-indexed).

        Returns:
            Dictionary with train_loss and train_accuracy for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in self.train_loader:
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

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        avg_loss = (
            total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0.0
        )
        accuracy = correct / total if total > 0 else 0.0

        return {"train_loss": avg_loss, "train_accuracy": accuracy}

    def validate(self, epoch: int) -> dict[str, float]:
        """Validate the model on validation set.

        Uses Evaluator.full_eval() for metric computation.

        Args:
            epoch: Current epoch number (0-indexed).

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

        return {
            "val_loss": avg_loss,
            "val_accuracy": metrics["accuracy"],
            "val_f1": metrics["macro_f1"],
        }

    def fit(self, num_epochs: int) -> dict[str, list[float]]:
        """Run the full training loop.

        Args:
            num_epochs: Number of epochs to train for.

        Returns:
            Full training history dictionary.
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

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

    def save_all_artifacts(self, checkpoint_dir: pathlib.Path | str) -> None:
        """Save all training artifacts.

        Saves tokenizer, model_config.json, train_config.yaml, label_map.json,
        metrics.json, and run_info.json. Also calls evaluator.save_report.

        Args:
            checkpoint_dir: Directory to save artifacts to.
        """
        checkpoint_path = pathlib.Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        model_config = self.config.get("model", {})
        model_config_path = checkpoint_path / "model_config.json"
        with open(model_config_path, "w", encoding="utf-8") as f:
            json.dump(model_config, f, indent=2)

        train_config_path = checkpoint_path / "train_config.yaml"
        import yaml

        with open(train_config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False)

        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        label_map_path = checkpoint_path / "label_map.json"
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump(label_map, f, indent=2)

        metrics_path = checkpoint_path / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        run_info = {
            "num_epochs": len(self.history["train_loss"]),
            "final_train_loss": self.history["train_loss"][-1]
            if self.history["train_loss"]
            else None,
            "final_val_accuracy": self.history["val_accuracy"][-1]
            if self.history["val_accuracy"]
            else None,
            "best_val_accuracy": max(self.history["val_accuracy"])
            if self.history["val_accuracy"]
            else None,
        }
        run_info_path = checkpoint_path / "run_info.json"
        with open(run_info_path, "w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=2)

        tokenizer_path = checkpoint_path / "tokenizer"
        tokenizer = SentimentTokenizer(
            model_name=model_config.get("name", "bert-base-uncased")
        )
        tokenizer.save(tokenizer_path)

        self.evaluator.save_report(checkpoint_path)

        self.logger.info(f"Saved all artifacts to {checkpoint_path}")
