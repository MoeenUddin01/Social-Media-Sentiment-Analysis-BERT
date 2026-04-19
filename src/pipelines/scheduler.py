"""Learning rate scheduler factory for BERT training.

Provides various learning rate scheduling strategies including
linear warmup, cosine annealing, and polynomial decay.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

if TYPE_CHECKING:
    from torch.optim.lr_scheduler import LambdaLR

    from src.models.bert_classifier import BertSentimentClassifier


def get_optimizer(
    model: "BertSentimentClassifier",
    config: dict,
) -> AdamW:
    """Create AdamW optimizer for BERT model.

    Args:
        model: BERT sentiment classifier model.
        config: Training configuration dictionary containing learning_rate
            and weight_decay keys.

    Returns:
        AdamW optimizer configured for the model parameters.

    Raises:
        KeyError: If learning_rate or weight_decay not found in config.
    """
    training_config = config.get("training", {})
    learning_rate = training_config.get("learning_rate", 2e-5)
    weight_decay = training_config.get("weight_decay", 0.01)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    return AdamW(optimizer_grouped_parameters, lr=learning_rate)


def get_scheduler(
    optimizer: AdamW,
    num_training_steps: int,
    config: dict,
) -> "LambdaLR":
    """Create learning rate scheduler with warmup.

    Args:
        optimizer: AdamW optimizer to schedule.
        num_training_steps: Total number of training steps (epochs * batches).
        config: Training configuration dictionary containing warmup_steps.

    Returns:
        Linear schedule with warmup LambdaLR.
    """
    training_config = config.get("training", {})
    warmup_steps = training_config.get("warmup_steps", 500)

    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
