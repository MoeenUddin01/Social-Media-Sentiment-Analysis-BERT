"""Training entry point for BERT sentiment classification.

Loads configuration, initializes model and tokenizer, builds DataLoaders,
and runs the training pipeline.
"""

from __future__ import annotations

import pathlib
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import SentimentDataset
from src.models.bert_classifier import BertSentimentClassifier
from src.models.fine_tuner import FineTuner
from src.models.tokenizer import SentimentTokenizer


def load_config(config_path: pathlib.Path | str | None = None) -> dict[str, Any]:
    """Load training configuration from YAML file.

    Args:
        config_path: Path to config YAML file. Defaults to config.ymal
            in project root.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    if config_path is None:
        project_root = pathlib.Path(__file__).parent.parent.parent
        config_path = project_root / "config.ymal"
    else:
        config_path = pathlib.Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Handle empty config file
    if config is None:
        config = {}

    return config


def get_default_config() -> dict[str, Any]:
    """Get default training configuration.

    Returns:
        Default configuration dictionary with training hyperparameters.
    """
    return {
        "model": {
            "name": "bert-base-uncased",
            "num_labels": 3,
            "dropout": 0.3,
        },
        "training": {
            "batch_size": 32,
            "max_length": 128,
            "epochs": 3,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_steps": 500,
            "gradual_unfreeze": True,
            "freeze_epochs": 1,
        },
        "data": {
            "train_file": "train_cleaned.csv",
            "val_file": "val_cleaned.csv",
            "text_column": "cleaned_text",
            "label_column": "label",
        },
        "checkpoints": {
            "save_dir": "artifacts/checkpoints",
            "save_best": True,
            "save_every_epoch": False,
        },
    }


def create_dataloaders(
    tokenizer: SentimentTokenizer,
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders.

    Args:
        tokenizer: Initialized tokenizer for encoding text.
        config: Training configuration dictionary.

    Returns:
        Tuple of (train_dataloader, val_dataloader).
    """
    data_config = config.get("data", {})
    train_file = data_config.get("train_file", "train_cleaned.csv")
    val_file = data_config.get("val_file", "val_cleaned.csv")
    max_length = config.get("training", {}).get("max_length", 128)
    batch_size = config.get("training", {}).get("batch_size", 32)
    text_column = data_config.get("text_column", "cleaned_text")
    label_column = data_config.get("label_column", "label")

    # Create datasets
    train_dataset = SentimentDataset(
        data=train_file,
        tokenizer=tokenizer.tokenizer,
        max_length=max_length,
        text_column=text_column,
        label_column=label_column,
    )

    val_dataset = SentimentDataset(
        data=val_file,
        tokenizer=tokenizer.tokenizer,
        max_length=max_length,
        text_column=text_column,
        label_column=label_column,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader


def train_epoch(
    model: BertSentimentClassifier,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train for one epoch.

    Args:
        model: The BERT classifier model.
        train_loader: DataLoader for training data.
        optimizer: Optimizer for parameter updates.
        device: Device to run training on.

    Returns:
        Dictionary with training metrics.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    criterion = torch.nn.CrossEntropyLoss()

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Create inputs dict for model
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        logits = model(inputs)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return {"loss": avg_loss, "accuracy": accuracy}


def validate(
    model: BertSentimentClassifier,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Validate the model.

    Args:
        model: The BERT classifier model.
        val_loader: DataLoader for validation data.
        device: Device to run validation on.

    Returns:
        Dictionary with validation metrics.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Create inputs dict for model
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

            logits = model(inputs)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return {"loss": avg_loss, "accuracy": accuracy}


def train(config: dict[str, Any] | None = None) -> None:
    """Run the training pipeline.

    Loads config, initializes model and tokenizer, builds DataLoaders,
    and trains the model with optional gradual unfreezing.

    Args:
        config: Training configuration dictionary. If None, loads from
            config file or uses defaults.
    """
    # Load or use provided config
    if config is None:
        try:
            config = load_config()
        except FileNotFoundError:
            config = get_default_config()

    # Merge with defaults for any missing keys
    defaults = get_default_config()
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
        elif isinstance(value, dict):
            config[key] = {**value, **config.get(key, {})}

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize tokenizer
    tokenizer = SentimentTokenizer()
    print("Tokenizer initialized: bert-base-uncased")

    # Initialize model
    num_labels = config.get("model", {}).get("num_labels", 3)
    model = BertSentimentClassifier(num_labels=num_labels)
    model.to(device)
    print(f"Model initialized with {num_labels} labels")

    # Create DataLoaders
    train_loader, val_loader = create_dataloaders(tokenizer, config)
    print(f"DataLoaders created: {len(train_loader)} train batches")

    # Initialize fine-tuner for layer management
    fine_tuner = FineTuner(model)

    # Get training config
    training_config = config.get("training", {})
    epochs = training_config.get("epochs", 3)
    base_lr = training_config.get("learning_rate", 2e-5)
    weight_decay = training_config.get("weight_decay", 0.01)
    gradual_unfreeze = training_config.get("gradual_unfreeze", True)
    freeze_epochs = training_config.get("freeze_epochs", 1)

    # Determine which layers to unfreeze based on epoch
    num_bert_layers = len(fine_tuner.bert_encoder)

    best_val_accuracy = 0.0
    best_checkpoint_path: pathlib.Path | None = None

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Apply gradual unfreezing strategy
        if gradual_unfreeze:
            if epoch < freeze_epochs:
                # Freeze all BERT layers, train only classifier
                fine_tuner.freeze_base_layers()
                print("  Strategy: Frozen base layers (classifier only)")
            else:
                # Gradually unfreeze more layers
                progress = (epoch - freeze_epochs) / max(1, epochs - freeze_epochs)
                layers_to_unfreeze = max(1, int(num_bert_layers * progress))
                fine_tuner.gradual_unfreeze(layer_num=layers_to_unfreeze)
                print(f"  Strategy: Unfrozen top {layers_to_unfreeze} layers")
        else:
            # Unfreeze all layers from start
            fine_tuner.unfreeze_all()
            print("  Strategy: All layers unfrozen")

        # Get optimizer with layer-wise learning rate decay
        param_groups = fine_tuner.get_parameter_groups(
            base_lr=base_lr, decay_rate=0.95
        )
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, " f"Accuracy: {train_metrics['accuracy']:.4f}")

        # Validate
        val_metrics = validate(model, val_loader, device)
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, " f"Accuracy: {val_metrics['accuracy']:.4f}")

        # Save checkpoint
        save_best = config.get("checkpoints", {}).get("save_best", True)
        if save_best and val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            checkpoint_name = f"best_model_epoch{epoch + 1}.pt"
            best_checkpoint_path = model.save(checkpoint_name)
            print(f"  Saved best checkpoint: {best_checkpoint_path}")

    print("\nTraining complete!")
    if best_checkpoint_path:
        print(f"Best checkpoint: {best_checkpoint_path} " f"(accuracy: {best_val_accuracy:.4f})")

    # Save final checkpoint
    final_checkpoint = model.save("final_model.pt")
    print(f"Final checkpoint saved: {final_checkpoint}")

    # Save tokenizer
    tokenizer_path = tokenizer.save()
    print(f"Tokenizer saved: {tokenizer_path}")

    # Final evaluation with ModelEvaluator
    from src.models.evaluator import ModelEvaluator

    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    evaluator = ModelEvaluator(model, device, label_map)

    print("\nRunning final evaluation on validation set...")
    eval_metrics = evaluator.evaluate(val_loader)
    print(f"Final eval - Accuracy: {eval_metrics['accuracy']:.4f}, " f"Macro F1: {eval_metrics['macro_f1']:.4f}")

    # Save evaluation report
    checkpoint_dir = final_checkpoint.parent
    evaluator.save_report(checkpoint_dir)
    print(f"Evaluation report saved to: {checkpoint_dir}")


if __name__ == "__main__":
    train()
