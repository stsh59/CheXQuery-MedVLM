"""
Gated Fusion Module with Learned Query Pooling.

This module combines CLS token with query embeddings using adaptive gating,
then pools the queries down to a fixed number of visual tokens.
"""
import logging
from typing import Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GatedFusionModule(nn.Module):
    """
    Gated fusion with learned query pooling.
    
    This module:
    1. Computes an adaptive gate to balance global (CLS) and local (queries) info
    2. Uses learned pooling queries to compress N queries to M tokens
    3. Produces final visual tokens for the text decoder
    
    Args:
        hidden_dim: Hidden dimension
        num_condition_queries: Number of condition queries
        num_anatomical_queries: Number of anatomical queries
        num_pool_queries: Number of output pooled tokens
        gate_hidden_dim: Hidden dimension for gate MLP
        num_heads: Number of attention heads for pooling
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_condition_queries: int = 14,
        num_anatomical_queries: int = 6,
        num_pool_queries: int = 10,
        gate_hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_input_queries = num_condition_queries + num_anatomical_queries
        self.num_pool_queries = num_pool_queries
        
        # Gate network: determines balance between CLS and queries
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        # Learnable pooling queries
        self.pool_queries = nn.Parameter(
            torch.randn(num_pool_queries, hidden_dim) * 0.02
        )
        
        # Pooling cross-attention
        self.pool_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer norms
        self.norm_pool = nn.LayerNorm(hidden_dim)
        self.norm_output = nn.LayerNorm(hidden_dim)
        
        # Final projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
        logger.info(f"Initialized GatedFusionModule: {self.num_input_queries} queries -> {num_pool_queries} pooled + 1 CLS")
    
    def forward(
        self,
        cls_token: torch.Tensor,
        query_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            cls_token: CLS token [batch, hidden_dim]
            query_embeddings: Query embeddings [batch, num_queries, hidden_dim]
            
        Returns:
            visual_tokens: Fused visual tokens [batch, num_pool_queries + 1, hidden_dim]
            gate_values: Gate values [batch, 1]
        """
        batch_size = cls_token.shape[0]
        
        # Compute gate value
        query_mean = query_embeddings.mean(dim=1)  # [batch, hidden]
        gate_input = torch.cat([cls_token, query_mean], dim=-1)  # [batch, hidden*2]
        gate = self.gate_network(gate_input)  # [batch, 1]
        
        # Apply gating to CLS token
        cls_gated = gate * cls_token  # [batch, hidden]
        
        # Apply inverse gating to queries (weighted sum effect)
        queries_weighted = (1 - gate).unsqueeze(-1) * query_embeddings  # [batch, num_q, hidden]
        
        # Learned query pooling
        pool_queries = self.pool_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Cross-attention: pool queries attend to weighted query embeddings
        pooled_output, _ = self.pool_attention(
            query=pool_queries,
            key=queries_weighted,
            value=queries_weighted,
        )
        pooled_output = self.norm_pool(pool_queries + pooled_output)
        
        # Combine CLS token with pooled queries
        cls_expanded = cls_gated.unsqueeze(1)  # [batch, 1, hidden]
        visual_tokens = torch.cat([cls_expanded, pooled_output], dim=1)  # [batch, num_pool+1, hidden]
        
        # Final projection and normalization
        visual_tokens = self.output_projection(visual_tokens)
        visual_tokens = self.norm_output(visual_tokens)
        
        return visual_tokens, gate


class QuerySplitter(nn.Module):
    """
    Utility to split combined queries back into condition and anatomical.
    """
    
    def __init__(
        self,
        num_condition_queries: int = 14,
        num_anatomical_queries: int = 6,
    ):
        super().__init__()
        self.num_condition = num_condition_queries
        self.num_anatomical = num_anatomical_queries
    
    def forward(
        self,
        combined_queries: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split combined queries.
        
        Args:
            combined_queries: [batch, num_condition + num_anatomical, hidden]
            
        Returns:
            condition_queries: [batch, num_condition, hidden]
            anatomical_queries: [batch, num_anatomical, hidden]
        """
        condition = combined_queries[:, :self.num_condition, :]
        anatomical = combined_queries[:, self.num_condition:, :]
        return condition, anatomical
