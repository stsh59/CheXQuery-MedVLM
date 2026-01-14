"""
Auxiliary Classification Head for CheXbert Labels.

This module provides multi-label classification for the 14 CheXbert
conditions, serving as an auxiliary training signal.
"""
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AuxiliaryClassificationHead(nn.Module):
    """
    Multi-label classification head for CheXbert conditions.
    
    Takes condition query embeddings and predicts presence/absence
    of each CheXbert condition.
    
    Args:
        hidden_dim: Input hidden dimension
        num_classes: Number of classes (14 for CheXbert)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_classes: int = 14,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        logger.info(f"Initialized AuxiliaryClassificationHead for {num_classes} classes")
    
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
    ) -> torch.Tensor:
        """
        Compute binary cross-entropy loss.
        
        Args:
            logits: Predicted logits [batch, 14]
            targets: Target labels [batch, 14]
            class_weights: Optional class weights for imbalanced data
            
        Returns:
            loss: Scalar loss value
        """
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(-1).expand_as(logits)
            mask = mask > 0
            if mask.sum() == 0:
                return None
            logits = logits[mask]
            targets = targets[mask]
        
        if class_weights is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits, targets, weight=class_weights
            )
        else:
            loss = F.binary_cross_entropy_with_logits(logits, targets)
        
        return loss
    
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
