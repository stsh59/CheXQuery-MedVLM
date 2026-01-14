"""
Anatomical Region Query Module.

This module implements learnable queries for anatomical regions of chest X-rays.
Combined with condition queries, this provides spatial grounding for the model.
"""
import logging
from typing import List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Anatomical regions for chest X-ray
ANATOMICAL_REGIONS = [
    "cardiac",      # Heart and cardiac silhouette
    "left_lung",    # Left lung field
    "right_lung",   # Right lung field
    "mediastinum",  # Mediastinal structures
    "diaphragm",    # Diaphragm and costophrenic angles
    "spine",        # Spine and skeletal structures
]


class AnatomicalQueryModule(nn.Module):
    """
    Anatomical region queries.
    
    Implements 6 learnable query vectors representing different anatomical
    regions of chest X-rays. These queries provide spatial grounding,
    allowing the model to attend to specific anatomical areas.
    
    Args:
        num_queries: Number of anatomical queries (default: 6)
        hidden_dim: Query dimension
        regions: List of region names
        init_std: Standard deviation for initialization
    """
    
    def __init__(
        self,
        num_queries: int = 6,
        hidden_dim: int = 768,
        regions: Optional[List[str]] = None,
        init_std: float = 0.02,
    ):
        super().__init__()
        
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.regions = regions or ANATOMICAL_REGIONS
        
        # Initialize queries with Xavier uniform
        self.queries = nn.Parameter(torch.empty(num_queries, hidden_dim))
        nn.init.xavier_uniform_(self.queries)
        
        logger.info(f"Initialized {num_queries} anatomical region queries")
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Get anatomical queries expanded for batch.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Queries expanded for batch [batch, num_queries, hidden_dim]
        """
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        return queries
    
    def get_region_names(self) -> List[str]:
        """Get list of region names."""
        return self.regions.copy()
