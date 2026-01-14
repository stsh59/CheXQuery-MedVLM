"""
Linear projection layer to map SigLIP embeddings to BioGPT hidden size.
"""
import torch
import torch.nn as nn


class ProjectionLayer(nn.Module):
    """
    Linear projection from SigLIP embeddings to BioGPT hidden size.
    
    Args:
        input_dim: SigLIP embedding dimension (default: 768)
        output_dim: BioGPT hidden dimension (default: 1024)
        dropout: Dropout rate
    """
    
    def __init__(self, input_dim: int = 768, output_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """
        Project image embeddings.
        
        Args:
            image_embeds: SigLIP image embeddings [batch_size, input_dim]
        
        Returns:
            Projected embeddings [batch_size, output_dim]
        """
        return self.projection(image_embeds)

