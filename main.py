"""Main entry point for BERT sentiment analysis CLI.

Provides commands for training, evaluation, and inference.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch
from torch.utils.data import DataLoader

from src.data.dataset import SentimentDataset
from src.models.bert_classifier import BertSentimentClassifier
from src.models.evaluator import ModelEvaluator
from src.models.tokenizer import SentimentTokenizer


def main() -> int:
    """Main entry point for CLI.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(
        description="BERT Sentiment Analysis CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

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

    if args.command == "evaluate":
        return run_evaluate(args)
    else:
        parser.print_help()
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
            # Find checkpoint file in directory
            checkpoint_files = list(checkpoint_path.glob("*.pt"))
            if not checkpoint_files:
                print(f"Error: No checkpoint files found in {checkpoint_path}")
                return 1
            checkpoint_file = checkpoint_files[0]
            checkpoints_dir = checkpoint_path
        else:
            checkpoint_file = checkpoint_path
            checkpoints_dir = checkpoint_path.parent

        # Load model via BertSentimentClassifier.from_pretrained
        print(f"Loading model from {checkpoint_file}...")
        model = BertSentimentClassifier.from_pretrained(
            checkpoint_name=checkpoint_file.name,
            checkpoints_dir=checkpoints_dir,
        )
        model.to(device)

        # Initialize tokenizer
        tokenizer = SentimentTokenizer()
        print("Tokenizer initialized")

        # Create test dataloader
        print(f"Loading test data from {args.test_data}...")
        test_dataset = SentimentDataset(
            data=args.test_data,
            tokenizer=tokenizer.tokenizer,
            max_length=128,
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        print(f"Loaded {len(test_dataset)} test samples")

        # Instantiate ModelEvaluator
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        evaluator = ModelEvaluator(model, device, label_map)
        print("ModelEvaluator initialized")

        # Run evaluation
        print("\nRunning evaluation...")
        metrics = evaluator.evaluate(test_loader)

        print(f"\nEvaluation Results:")
        print(f"  Accuracy:      {metrics['accuracy']:.4f}")
        print(f"  Macro F1:      {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1:   {metrics['weighted_f1']:.4f}")
        print(f"  Precision:     {metrics['precision']:.4f}")
        print(f"  Recall:        {metrics['recall']:.4f}")

        # Save report
        output_dir = args.output_dir if args.output_dir else checkpoints_dir
        evaluator.save_report(output_dir)
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
