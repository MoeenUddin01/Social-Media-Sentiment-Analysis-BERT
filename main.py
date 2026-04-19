"""Main entry point for BERT sentiment analysis CLI.

Provides commands for training, evaluation, and inference.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch
from torch.utils.data import DataLoader

import yaml

from src.data.dataset import SentimentDataset
from src.models.tokenizer import SentimentTokenizer
from src.pipelines.data_preprocessin import DataPipeline
from src.pipelines.model_evaluation import EvaluationPipeline


def main() -> int:
    """Main entry point for CLI.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(description="BERT Sentiment Analysis CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train a sentiment model")
    train_parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file",
    )
    train_parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="Name of raw data file in dataset/raw/",
    )

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory or file",
    )
    eval_parser.add_argument(
        "--test-data",
        type=str,
        default="dataset/processed/test_cleaned.csv",
        help="Path to test data CSV file",
    )
    eval_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation reports (defaults to checkpoint dir)",
    )

    args = parser.parse_args()

    if args.command == "train":
        return run_train(args)
    elif args.command == "evaluate":
        return run_evaluate(args)
    else:
        parser.print_help()
        return 1


def run_train(args: argparse.Namespace) -> int:
    """Run the train subcommand.

    Args:
        args: Parsed command line arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        config_path = pathlib.Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            return 1

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        tokenizer = SentimentTokenizer(
            model_name=config.get("model", {}).get("name", "bert-base-uncased")
        )
        print("Tokenizer initialized")

        pipeline = DataPipeline(config, tokenizer)
        print(f"Starting data pipeline with {args.data_file}...")

        train_loader, val_loader, test_loader = pipeline.run(args.data_file)
        print(
            f"Data pipeline complete: "
            f"train={len(train_loader.dataset)}, "
            f"val={len(val_loader.dataset)}, "
            f"test={len(test_loader.dataset)}"
        )

        print("\nReady for training - pass loaders to TrainingPipeline")

        return 0

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return 1
    except KeyError as e:
        print(f"Error: Missing configuration key - {e}")
        return 1
    except Exception as e:
        print(f"Error during training setup: {e}")
        return 1


def run_evaluate(args: argparse.Namespace) -> int:
    """Run the evaluate subcommand.

    Args:
        args: Parsed command line arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Resolve checkpoint path
        checkpoint_path = pathlib.Path(args.checkpoint)
        if checkpoint_path.is_dir():
            checkpoints_dir = checkpoint_path
        else:
            checkpoints_dir = checkpoint_path.parent

        # Initialize EvaluationPipeline
        print(f"Loading model from {checkpoints_dir}...")
        pipeline = EvaluationPipeline(checkpoint_dir=checkpoints_dir, device=device)
        print("EvaluationPipeline initialized")

        # Create test dataloader
        print(f"Loading test data from {args.test_data}...")
        tokenizer = SentimentTokenizer()
        test_dataset = SentimentDataset(
            data=args.test_data,
            tokenizer=tokenizer.tokenizer,
            max_length=128,
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        print(f"Loaded {len(test_dataset)} test samples")

        # Run evaluation
        print("\nRunning evaluation...")
        metrics = pipeline.run(test_loader)

        print("\nEvaluation Results:")
        print(f"  Accuracy:      {metrics['accuracy']:.4f}")
        print(f"  Macro F1:      {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1:   {metrics['weighted_f1']:.4f}")
        print(f"  Precision:     {metrics['precision']:.4f}")
        print(f"  Recall:        {metrics['recall']:.4f}")

        # Save report
        output_dir = args.output_dir if args.output_dir else checkpoints_dir
        pipeline.save_report(output_dir)
        print(f"\nEvaluation report saved to: {output_dir}")

        return 0

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return 1
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
