"""
CheXbert-Initialized Condition Query Module.

This module implements learnable queries initialized from BioBERT embeddings
of CheXbert condition labels. This is a novel contribution that provides
condition-aware visual attention.
"""
import logging
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

# CheXbert condition labels (14 conditions)
CHEXBERT_CONDITIONS = [
    "no finding",
    "enlarged cardiomediastinum",
    "cardiomegaly",
    "lung opacity",
    "lung lesion",
    "edema",
    "consolidation",
    "pneumonia",
    "atelectasis",
    "pneumothorax",
    "pleural effusion",
    "pleural other",
    "fracture",
    "support devices",
]


class ConditionQueryModule(nn.Module):
    """
    CheXbert-initialized condition queries.
    
    Initializes 14 learnable query vectors from BioBERT embeddings of
    CheXbert condition labels. During training, these queries learn to
    attend to patches that are relevant for each condition.
    
    Args:
        num_queries: Number of condition queries (default: 14)
        hidden_dim: Query dimension
        biobert_model: BioBERT model for initialization
        conditions: List of condition names
    """
    
    def __init__(
        self,
        num_queries: int = 14,
        hidden_dim: int = 768,
        biobert_model: str = "dmis-lab/biobert-base-cased-v1.2",
        conditions: Optional[List[str]] = None,
    ):
        super().__init__()
        
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.conditions = conditions or CHEXBERT_CONDITIONS
        
        # Initialize queries from BioBERT
        initial_queries = self._initialize_from_biobert(biobert_model)
        
        # Make queries learnable parameters
        self.queries = nn.Parameter(initial_queries)
        
        logger.info(f"Initialized {num_queries} condition queries from BioBERT")
    
    def _initialize_from_biobert(self, model_name: str) -> torch.Tensor:
        """
        Initialize query vectors from BioBERT embeddings.
        
        Args:
            model_name: BioBERT model name
            
        Returns:
            Tensor of query embeddings [num_queries, hidden_dim]
        """
        logger.info(f"Loading BioBERT for query initialization: {model_name}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.eval()
            
            embeddings = []
            with torch.no_grad():
                for condition in self.conditions:
                    # Tokenize condition name
                    inputs = tokenizer(
                        condition,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=32,
                    )
                    
                    # Get CLS token embedding
                    outputs = model(**inputs)
                    cls_embed = outputs.last_hidden_state[:, 0, :]  # [1, hidden]
                    embeddings.append(cls_embed)
            
            # Stack all embeddings
            query_embeds = torch.cat(embeddings, dim=0)  # [14, hidden]
            
            # Project if BioBERT dim differs from target dim
            if query_embeds.shape[1] != self.hidden_dim:
                projection = nn.Linear(query_embeds.shape[1], self.hidden_dim)
                query_embeds = projection(query_embeds)
            
            logger.info(f"Initialized condition queries with shape: {query_embeds.shape}")
            
            # Clean up
            del model, tokenizer
            
            return query_embeds
            
        except Exception as e:
            logger.warning(f"Failed to load BioBERT: {e}. Using random initialization.")
            return torch.randn(self.num_queries, self.hidden_dim) * 0.02
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Get condition queries expanded for batch.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Queries expanded for batch [batch, num_queries, hidden_dim]
        """
        # Expand queries for batch
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        return queries
    
    def get_condition_names(self) -> List[str]:
        """Get list of condition names."""
        return self.conditions.copy()
