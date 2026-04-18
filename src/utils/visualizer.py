"""Visualization utilities for training and evaluation.

Provides plotting functions for training curves, confusion matrices,
and performance visualizations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pathlib

    import numpy as np


class TrainingVisualizer:
    """Visualizer for training and evaluation metrics.

    Generates plots for training curves, confusion matrices,
    and performance comparisons.

    Args:
        save_dir: Directory to save visualization outputs.
        style: Matplotlib style to use for plots.

    Raises:
        ValueError: If save_dir is invalid.
    """
    pass


class ResultVisualizer:
    """Visualizer for evaluation results and metrics.

    Generates plots for confusion matrices, confidence distributions,
    and per-class performance metrics.
    """

    def __init__(self) -> None:
        """Initialize the result visualizer."""
        pass

    def plot_per_class_f1_bar(
        self,
        f1_scores: "np.ndarray",
        label_names: list[str],
        save_path: "pathlib.Path | str",
    ) -> None:
        """Plot per-class F1 scores as a bar chart.

        Args:
            f1_scores: Array of F1 scores for each class.
            label_names: List of class names.
            save_path: Path to save the plot to.
        """
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 6))
            bars = ax.bar(label_names, f1_scores, color=["#3498db", "#e74c3c", "#2ecc71"])

            ax.set_ylabel("F1 Score", fontsize=12)
            ax.set_title("Per-Class F1 Scores", fontsize=14)
            ax.set_ylim(0, 1.0)

            # Add value labels on bars
            for bar, score in zip(bars, f1_scores):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{score:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except ImportError:
            pass

    def plot_confidence_histogram(
        self,
        confidences: "np.ndarray",
        save_path: "pathlib.Path | str",
    ) -> None:
        """Plot histogram of prediction confidences.

        Args:
            confidences: Array of confidence scores for predictions.
            save_path: Path to save the plot to.
        """
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 6))

            ax.hist(confidences, bins=20, color="#3498db", edgecolor="black", alpha=0.7)
            ax.set_xlabel("Confidence Score", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.set_title("Distribution of Prediction Confidences", fontsize=14)
            ax.set_xlim(0, 1.0)

            # Add mean line
            mean_conf = confidences.mean()
            ax.axvline(mean_conf, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_conf:.3f}")
            ax.legend()

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except ImportError:
            pass

    def plot_confusion_matrix(
        self,
        cm: "np.ndarray",
        label_map: dict[int, str],
        save_path: "pathlib.Path | str",
    ) -> None:
        """Plot confusion matrix as a heatmap.

        Args:
            cm: Confusion matrix array (num_classes, num_classes).
            label_map: Mapping from class index to class name.
            save_path: Path to save the plot to.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(8, 6))

            label_names = [label_map[i] for i in range(len(cm))]
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=label_names,
                yticklabels=label_names,
                ax=ax,
            )

            ax.set_xlabel("Predicted", fontsize=12)
            ax.set_ylabel("True", fontsize=12)
            ax.set_title("Confusion Matrix", fontsize=14)

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except ImportError:
            pass
