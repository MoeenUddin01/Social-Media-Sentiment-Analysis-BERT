"""HuggingFace Trainer wrapper for BERT fine-tuning.

Provides a high-level interface for training BERT models using
the HuggingFace transformers library.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    import torch.nn as nn


class FineTuner:
    """Fine-tuning utilities for BERT models.

    Provides methods for layer freezing, gradual unfreezing, and
    generating parameter groups with layer-wise learning rate decay
    for AdamW optimizer.

    Attributes:
        model: The BERT model to fine-tune.
        bert_encoder: Reference to the BERT encoder layers.
    """

    def __init__(self, model: nn.Module) -> None:
        """Initialize the FineTuner.

        Args:
            model: A BERT-based model with a bert attribute containing
                encoder layers.
        """
        self.model = model
        self.bert_encoder = model.bert.encoder.layer

    def freeze_base_layers(self) -> None:
        """Freeze all BERT base layers (encoder).

        Only the classification head remains trainable.
        """
        for param in self.model.bert.parameters():
            param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = True

    def gradual_unfreeze(self, layer_num: int) -> None:
        """Unfreeze layers starting from the top (layer-wise).

        Unfreezes the last `layer_num` transformer layers plus the
        pooler and classification head.

        Args:
            layer_num: Number of layers to unfreeze from the top.

        Raises:
            ValueError: If layer_num is negative or exceeds layer count.
        """
        num_layers = len(self.bert_encoder)
        if layer_num < 0:
            raise ValueError(f"layer_num must be non-negative, got {layer_num}")
        if layer_num > num_layers:
            raise ValueError(
                f"layer_num ({layer_num}) exceeds total layers ({num_layers})"
            )

        # First, freeze all bert layers
        self.freeze_base_layers()

        # Unfreeze the last `layer_num` layers
        start_idx = num_layers - layer_num
        for i in range(start_idx, num_layers):
            for param in self.bert_encoder[i].parameters():
                param.requires_grad = True

        # Always unfreeze pooler and classifier
        if hasattr(self.model.bert, "pooler"):
            for param in self.model.bert.pooler.parameters():
                param.requires_grad = True

        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def get_parameter_groups(
        self, base_lr: float, decay_rate: float = 0.95
    ) -> list[dict]:
        """Generate parameter groups with layer-wise learning rate decay.

        Creates parameter groups for AdamW optimizer where each lower
        layer receives a slightly reduced learning rate.

        Args:
            base_lr: Base learning rate for the top layer.
            decay_rate: Multiplicative decay factor per layer.
                Default is 0.95 (5% reduction per layer).

        Returns:
            List of parameter group dictionaries with 'params' and 'lr' keys,
            suitable for torch.optim.AdamW.
        """
        num_layers = len(self.bert_encoder)
        parameter_groups: list[dict] = []

        # Group parameters by layer
        for layer_idx in range(num_layers):
            layer_params: list[nn.Parameter] = []
            for param in self.bert_encoder[layer_idx].parameters():
                if param.requires_grad:
                    layer_params.append(param)

            if layer_params:
                # Lower layers get lower learning rates
                layer_lr = base_lr * (decay_rate ** (num_layers - 1 - layer_idx))
                parameter_groups.append({"params": layer_params, "lr": layer_lr})

        # Pooler parameters
        if hasattr(self.model.bert, "pooler"):
            pooler_params: list[nn.Parameter] = [
                p for p in self.model.bert.pooler.parameters() if p.requires_grad
            ]
            if pooler_params:
                # Pooler uses base learning rate
                parameter_groups.append({"params": pooler_params, "lr": base_lr})

        # Classifier parameters (highest learning rate)
        classifier_params: list[nn.Parameter] = [
            p for p in self.model.classifier.parameters() if p.requires_grad
        ]
        if classifier_params:
            # Classification head gets 2x base learning rate
            parameter_groups.append(
                {"params": classifier_params, "lr": base_lr * 2.0}
            )

        return parameter_groups
