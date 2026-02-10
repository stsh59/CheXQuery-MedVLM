"""
Auxiliary Classification Head for CheXbert Labels.

This module provides multi-label classification for the 14 CheXbert
conditions, serving as an auxiliary training signal.

Supports:
- pos_weight-based BCE (correct class-imbalance handling)
- Focal loss (down-weights easy negatives, focuses on hard positives)
"""
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    alpha: Optional[float] = None,
) -> torch.Tensor:
    """
    Focal loss for multi-label binary classification.

    Focal loss down-weights easy (well-classified) examples so that training
    concentrates on hard, misclassified examples. This is especially effective
    for extreme class imbalance where most negatives are trivially correct.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        logits: Raw predictions [batch, num_classes]
        targets: Binary targets [batch, num_classes]
        pos_weight: Per-class weight for positive examples [num_classes]
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Global balancing factor for positives vs negatives

    Returns:
        Scalar focal loss
    """
    # Compute standard BCE per element (no reduction)
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )

    # p_t: probability of the correct class
    probs = torch.sigmoid(logits)
    pt = torch.where(targets == 1, probs, 1 - probs)

    # Focal modulating factor: (1 - p_t)^gamma
    focal_weight = (1 - pt) ** gamma

    loss = focal_weight * bce

    # Optional alpha balancing (alpha for positives, 1-alpha for negatives)
    if alpha is not None:
        alpha_t = torch.where(targets == 1, alpha, 1 - alpha)
        loss = alpha_t * loss

    return loss


class AuxiliaryClassificationHead(nn.Module):
    """
    Multi-label classification head for CheXbert conditions.

    Takes condition query embeddings and predicts presence/absence
    of each CheXbert condition.

    Args:
        hidden_dim: Input hidden dimension
        num_classes: Number of classes (14 for CheXbert)
        dropout: Dropout rate
        use_focal_loss: Whether to use focal loss instead of plain BCE
        focal_gamma: Focusing parameter for focal loss
        focal_alpha: Balancing factor for focal loss (None to disable)
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_classes: int = 14,
        dropout: float = 0.2,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        focal_alpha: Optional[float] = None,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        loss_type = "Focal" if use_focal_loss else "BCE"
        logger.info(
            f"Initialized AuxiliaryClassificationHead for {num_classes} classes "
            f"(loss={loss_type}, gamma={focal_gamma})"
        )

    def forward(
        self,
        condition_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            condition_embeddings: Condition query embeddings [batch, 14, hidden]

        Returns:
            logits: Classification logits [batch, 14]
        """
        # Apply classifier to each condition embedding
        logits = self.classifier(condition_embeddings)  # [batch, 14, 1]
        logits = logits.squeeze(-1)  # [batch, 14]

        return logits

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Compute classification loss with proper class-imbalance handling.

        Uses pos_weight (not weight) so that only positive-class loss is
        scaled up. This corrects for rare conditions being drowned out by
        the overwhelming negative majority. Optionally uses focal loss for
        additional hard-example mining.

        Args:
            logits: Predicted logits [batch, 14]
            targets: Target labels [batch, 14]
            class_weights: Per-label pos_weight for imbalanced data [14]
            mask: Sample-level mask [batch] or [batch, 14]

        Returns:
            loss: Scalar loss value, or None if no valid samples
        """
        # --- Compute per-element loss (keep 2D structure for pos_weight) ---
        pos_weight = None
        if class_weights is not None:
            if class_weights.dim() == 1:
                pos_weight = class_weights  # [14]

        if self.use_focal_loss:
            loss = focal_bce_with_logits(
                logits,
                targets,
                pos_weight=pos_weight,
                gamma=self.focal_gamma,
                alpha=self.focal_alpha,
            )  # [batch, 14]
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight, reduction="none"
            )  # [batch, 14]

        # --- Apply sample-level mask (preserving 2D structure) ---
        if mask is not None:
            if mask.dim() == 1:
                mask_2d = mask.unsqueeze(-1)  # [batch, 1]
            else:
                mask_2d = mask
            mask_float = (mask_2d > 0).float()
            num_valid = mask_float.sum().clamp(min=1)

            if mask_float.sum() == 0:
                return None

            loss = loss * mask_float
            return loss.sum() / (num_valid * logits.shape[-1])

        return loss.mean()

    def predict(
        self,
        condition_embeddings: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict class labels.

        Args:
            condition_embeddings: Condition query embeddings
            threshold: Classification threshold

        Returns:
            predictions: Binary predictions [batch, 14]
            probabilities: Prediction probabilities [batch, 14]
        """
        logits = self.forward(condition_embeddings)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= threshold).float()

        return predictions, probabilities
