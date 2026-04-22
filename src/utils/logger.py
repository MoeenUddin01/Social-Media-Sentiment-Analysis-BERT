"""Structured logging configuration for the project.

Provides consistent logging setup across all modules with
configurable log levels and output formats.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
from datetime import datetime
from typing import TYPE_CHECKING, Any

import dagshub
import mlflow
import mlflow.pytorch
from dagshub.upload import Repo

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Sequence

    import matplotlib.figure
    import numpy as np
    import torch


def setup_logging(
    log_level: str = "INFO",
    log_file: str | None = None,
    format_string: str | None = None,
) -> None:
    """Configure logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file path for log output.
        format_string: Optional custom format string.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        log_path = pathlib.Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(level=level, format=format_string, handlers=handlers)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.

    Args:
        name: Logger name, typically __name__.

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)


class DagsHubLogger:
    """DagsHub/MLflow experiment tracking logger.

    Provides live experiment tracking during BERT sentiment classification training.
    Logs metrics at batch and epoch levels, artifacts, and model checkpoints.
    Creates live updating charts on the DagsHub dashboard.

    Attributes:
        repo_owner: DagsHub repository owner username.
        repo_name: DagsHub repository name.
        experiment_name: MLflow experiment name.
        run_name: Unique run name (model_name + timestamp).
        tracking_uri: MLflow tracking URI for DagsHub.
    """

    def __init__(
        self,
        config: dict[str, Any],
    ) -> None:
        """Initialize DagsHubLogger and set up MLflow tracking.

        Reads DagsHub configuration from config dict, initializes dagshub,
        and sets the MLflow experiment. Generates a unique run name based on
        model name and timestamp.

        Args:
            config: Full configuration dictionary containing 'dagshub' and
                'model'/'training' sections.

        Raises:
            KeyError: If required dagshub config keys are missing.
        """
        # Initialise logger FIRST so it is available in all code below
        self._logger = get_logger(__name__)

        dagshub_config = config.get("dagshub", {})

        self.repo_owner = dagshub_config.get("repo_owner", "your_dagshub_username")
        self.repo_name = dagshub_config.get("repo_name", "bert-sentiment")
        self.experiment_name = dagshub_config.get(
            "experiment_name", "bert-sentiment-analysis"
        )
        self.tracking_uri = dagshub_config.get(
            "tracking_uri",
            f"https://dagshub.com/{self.repo_owner}/{self.repo_name}.mlflow",
        )
        self.log_every_n_batches = dagshub_config.get("log_every_n_batches", 1)

        # Generate run name
        model_name = config.get("model", {}).get("name", "bert")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{model_name}_{timestamp}"

        # Set up MLflow tracking URI first
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Initialize dagshub (this sets up authentication)
        dagshub.init(
            repo_owner=self.repo_owner,
            repo_name=self.repo_name,
            mlflow=True,
        )
        
        # Create or get experiment - handle 404 by creating experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                self._logger.info(f"Created new experiment: {self.experiment_name} (ID: {experiment_id})")
            else:
                self._logger.info(f"Using existing experiment: {self.experiment_name}")
        except Exception as e:
            # If get fails, try creating directly
            try:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                self._logger.info(f"Created new experiment: {self.experiment_name}")
            except Exception:
                self._logger.warning(f"Could not create/get experiment: {e}")
        
        # Now set the experiment
        try:
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            self._logger.error(f"Failed to set experiment: {e}")

        self._logger.info(
            f"DagsHubLogger initialized for {self.repo_owner}/{self.repo_name}"
        )

    def start_run(self) -> None:
        """Start an MLflow run and log all configuration parameters.

        Starts the run with the generated run_name and logs:
        - All config.yaml values as flat key-value params
        - Model and training hyperparameters

        Raises:
            Exception: If MLflow start_run fails.
        """
        try:
            mlflow.start_run(run_name=self.run_name)
            self._logger.info(f"Started MLflow run: {self.run_name}")

            # Log configuration parameters (flattened)
            self._log_config_params()

        except Exception as e:
            self._logger.error(f"Failed to start MLflow run: {e}")
            raise

    def _log_config_params(self) -> None:
        """Log configuration parameters as MLflow params.

        Logs all relevant config values extracted from the config dict.
        """
        # These would be stored from config during init
        # For now, log basic info
        mlflow.log_param("run_name", self.run_name)
        mlflow.log_param("repo_owner", self.repo_owner)
        mlflow.log_param("repo_name", self.repo_name)

    def log_config(self, config: dict[str, Any]) -> None:
        """Log all configuration values as MLflow params.

        Args:
            config: Full configuration dictionary with 'model',
                'training', 'data' sections.
        """
        # Log model config
        model_config = config.get("model", {})
        mlflow.log_param("model_name", model_config.get("name", "bert-base-uncased"))
        mlflow.log_param("num_labels", model_config.get("num_labels", 3))
        mlflow.log_param("dropout", model_config.get("dropout", 0.3))

        # Log training config
        training_config = config.get("training", {})
        mlflow.log_param("max_length", training_config.get("max_length", 128))
        mlflow.log_param("batch_size", training_config.get("batch_size", 32))
        mlflow.log_param("learning_rate", training_config.get("learning_rate", 2.0e-5))
        mlflow.log_param("weight_decay", training_config.get("weight_decay", 0.01))
        mlflow.log_param("warmup_ratio", training_config.get("warmup_ratio", 0.1))
        mlflow.log_param("num_epochs", training_config.get("num_epochs", 10))
        mlflow.log_param("seed", training_config.get("seed", 42))

        self._logger.info("Logged configuration parameters to MLflow")

    def log_batch_metrics(
        self,
        batch: int,
        epoch: int,
        loss: float,
        accuracy: float,
        total_batches: int,
    ) -> None:
        """Log per-batch training metrics to MLflow.

        Called every batch during training to create live updating charts
        on the DagsHub dashboard.

        Args:
            batch: Current batch index (0-indexed within epoch).
            epoch: Current epoch number (0-indexed).
            loss: Batch loss value.
            accuracy: Batch accuracy value.
            total_batches: Total number of batches per epoch.

        Note:
            Step is calculated as (epoch * total_batches) + batch for
            continuous chart updates across epochs.
        """
        step = (epoch * total_batches) + batch

        mlflow.log_metric("train/batch_loss", loss, step=step)
        mlflow.log_metric("train/batch_accuracy", accuracy, step=step)

    def log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        learning_rate: float,
    ) -> None:
        """Log per-epoch metrics to MLflow.

        Called once per epoch after validation completes.

        Args:
            epoch: Current epoch number (0-indexed).
            train_metrics: Dictionary containing 'train_loss' and
                'train_accuracy'.
            val_metrics: Dictionary containing validation metrics like
                'val_loss', 'val_accuracy', 'macro_f1', 'weighted_f1',
                'precision', 'recall'.
            learning_rate: Current learning rate from scheduler.

        Note:
            Step equals the epoch number for epoch-level metrics.
        """
        step = epoch

        # Log training epoch metrics
        mlflow.log_metric(
            "train/epoch_loss", train_metrics.get("train_loss", 0.0), step=step
        )
        mlflow.log_metric(
            "train/epoch_accuracy",
            train_metrics.get("train_accuracy", 0.0),
            step=step,
        )

        # Log validation metrics
        mlflow.log_metric("val/loss", val_metrics.get("val_loss", 0.0), step=step)
        mlflow.log_metric(
            "val/accuracy", val_metrics.get("val_accuracy", 0.0), step=step
        )
        mlflow.log_metric(
            "val/f1_macro", val_metrics.get("macro_f1", val_metrics.get("val_f1", 0.0)), step=step
        )
        mlflow.log_metric(
            "val/f1_weighted", val_metrics.get("weighted_f1", 0.0), step=step
        )
        mlflow.log_metric(
            "val/precision", val_metrics.get("precision", 0.0), step=step
        )
        mlflow.log_metric("val/recall", val_metrics.get("recall", 0.0), step=step)

        # Log learning rate
        mlflow.log_metric("learning_rate", learning_rate, step=step)

        self._logger.info(
            f"Logged epoch {epoch + 1} metrics to DagsHub "
            f"(train_loss: {train_metrics.get('train_loss', 0.0):.4f}, "
            f"val_acc: {val_metrics.get('val_accuracy', 0.0):.4f})"
        )

    def log_test_metrics(self, test_metrics: dict[str, float]) -> None:
        """Log test set evaluation metrics.

        Called once at the end of training on the test set.

        Args:
            test_metrics: Dictionary containing test metrics like
                'accuracy', 'macro_f1', 'weighted_f1', 'precision', 'recall'.
        """
        mlflow.log_metric("test/accuracy", test_metrics.get("accuracy", 0.0))
        mlflow.log_metric("test/f1_macro", test_metrics.get("macro_f1", 0.0))
        mlflow.log_metric("test/f1_weighted", test_metrics.get("weighted_f1", 0.0))
        mlflow.log_metric("test/precision", test_metrics.get("precision", 0.0))
        mlflow.log_metric("test/recall", test_metrics.get("recall", 0.0))

        self._logger.info(
            f"Logged test metrics - Accuracy: {test_metrics.get('accuracy', 0.0):.4f}, "
            f"Macro F1: {test_metrics.get('macro_f1', 0.0):.4f}"
        )

    def log_plots(self, plots_dir: str | pathlib.Path) -> None:
        """Log evaluation plot artifacts to MLflow.

        Called after evaluator.save_report() to log all generated
        visualization images.

        Args:
            plots_dir: Directory containing plot files.

        Raises:
            FileNotFoundError: If plot directory or files don't exist.
        """
        plots_path = pathlib.Path(plots_dir)

        if not plots_path.exists():
            self._logger.warning(f"Plots directory not found: {plots_path}")
            return

        plot_files = [
            "confusion_matrix.png",
            "training_curves.png",
            "confidence_histogram.png",
            "per_class_f1_bar.png",
            "label_distribution.png",
        ]

        for plot_file in plot_files:
            plot_path = plots_path / plot_file
            if plot_path.exists():
                mlflow.log_artifact(str(plot_path), artifact_path="plots")
                self._logger.info(f"Logged plot artifact: {plot_file}")

    def log_model_artifact(self, checkpoint_dir: str | pathlib.Path) -> None:
        """Log model checkpoint directory as MLflow artifact.

        Logs the entire checkpoint directory including:
        - model.pt (best model weights)
        - tokenizer/ directory
        - model_config.json
        - label_map.json

        Args:
            checkpoint_dir: Directory containing model artifacts.

        Raises:
            FileNotFoundError: If checkpoint directory doesn't exist.
        """
        checkpoint_path = pathlib.Path(checkpoint_dir)

        if not checkpoint_path.exists():
            self._logger.warning(f"Checkpoint directory not found: {checkpoint_path}")
            return

        # Log entire directory as artifact
        mlflow.log_artifacts(str(checkpoint_path), artifact_path="model")
        self._logger.info(f"Logged model artifacts from {checkpoint_path}")

        # Log individual important files for easy access
        important_files = ["model_config.json", "label_map.json", "metrics.json"]
        for file_name in important_files:
            file_path = checkpoint_path / file_name
            if file_path.exists():
                mlflow.log_artifact(str(file_path), artifact_path="model/config")

    def log_confusion_matrix(
        self,
        cm: "np.ndarray",
        label_names: Sequence[str],
        step: int,
    ) -> None:
        """Log confusion matrix as an MLflow artifact image.

        Converts the confusion matrix to a matplotlib figure and logs
        it as an artifact. Called every epoch to show chart evolution.

        Args:
            cm: Confusion matrix as 2D numpy array.
            label_names: List of class label names.
            step: Step number for artifact naming (typically epoch).

        Raises:
            ImportError: If matplotlib is not available.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=label_names,
                yticklabels=label_names,
                ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"Confusion Matrix - Epoch {step + 1}")

            # Save temporary file
            temp_path = pathlib.Path(f"/tmp/confusion_matrix_step_{step}.png")
            fig.savefig(temp_path, bbox_inches="tight", dpi=150)
            plt.close(fig)

            # Log to MLflow
            mlflow.log_artifact(
                str(temp_path), artifact_path="confusion_matrices"
            )

            # Clean up
            temp_path.unlink(missing_ok=True)

            self._logger.info(f"Logged confusion matrix for step {step}")

        except ImportError:
            self._logger.warning("matplotlib or seaborn not available for confusion matrix")
        except Exception as e:
            self._logger.error(f"Failed to log confusion matrix: {e}")

    def end_run(self) -> None:
        """End the MLflow run and print DagsHub experiment URL.

        Closes the active MLflow run and prints the URL where experiment
        results can be viewed on DagsHub.
        """
        try:
            mlflow.end_run()
            experiment_url = (
                f"https://dagshub.com/{self.repo_owner}/{self.repo_name}/experiments"
            )
            self._logger.info(f"MLflow run ended")
            print(f"\n{'=' * 60}")
            print(f"DagsHub Experiment URL:")
            print(f"{experiment_url}")
            print(f"{'=' * 60}\n")
        except Exception as e:
            self._logger.error(f"Error ending MLflow run: {e}")
