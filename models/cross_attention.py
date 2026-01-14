"""
Multi-Head Cross-Attention Module.

This module implements cross-attention where queries (condition + anatomical)
attend to patch tokens from the vision encoder.
"""
import logging
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CrossAttentionLayer(nn.Module):
    """
    Single cross-attention layer with feed-forward network.
    
    Args:
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        ffn_dim: Feed-forward network dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 8,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        queries: torch.Tensor,
        key_value: torch.Tensor,
        key_value_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            queries: Query tensor [batch, num_queries, hidden_dim]
            key_value: Key/Value tensor [batch, num_patches, hidden_dim]
            key_value_mask: Optional mask for key/value
            
        Returns:
            output: Attended queries [batch, num_queries, hidden_dim]
            attn_weights: Attention weights [batch, num_queries, num_patches]
        """
        # Cross-attention with residual
        attn_output, attn_weights = self.cross_attn(
            query=queries,
            key=key_value,
            value=key_value,
            key_padding_mask=key_value_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        queries = self.norm1(queries + attn_output)
        
        # Feed-forward with residual
        ffn_output = self.ffn(queries)
        queries = self.norm2(queries + ffn_output)
        
        return queries, attn_weights


class CrossAttentionModule(nn.Module):
    """
    Multi-layer cross-attention module.
    
    Implements multiple cross-attention layers where combined queries
    (condition + anatomical) attend to visual patch tokens.
    
    Args:
        num_layers: Number of cross-attention layers
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        ffn_dim: Feed-forward network dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        num_layers: int = 2,
        hidden_dim: int = 768,
        num_heads: int = 8,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Stack of cross-attention layers
        self.layers = nn.ModuleList([
            CrossAttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        logger.info(f"Initialized CrossAttentionModule with {num_layers} layers")
    
    def forward(
        self,
        queries: torch.Tensor,
        patch_tokens: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through all cross-attention layers.
        
        Args:
            queries: Combined queries [batch, num_queries, hidden_dim]
            patch_tokens: Patch tokens [batch, num_patches, hidden_dim]
            patch_mask: Optional patch mask
            return_attention: Whether to return attention weights
            
        Returns:
            output: Attended queries [batch, num_queries, hidden_dim]
            attn_weights: Optional attention weights from last layer
        """
        all_attn_weights = []
        
        for layer in self.layers:
            queries, attn_weights = layer(
                queries=queries,
                key_value=patch_tokens,
                key_value_mask=patch_mask,
            )
            if return_attention:
                all_attn_weights.append(attn_weights)
        
        if return_attention:
            # Return attention from last layer
            return queries, all_attn_weights[-1]
        
        return queries, None
