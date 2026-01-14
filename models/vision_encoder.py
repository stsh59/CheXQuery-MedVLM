"""
SigLIP Vision Encoder with Patch Token Extraction.
"""
import logging
from typing import Tuple, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor
from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)


class VisionEncoder(nn.Module):
    """
    SigLIP-based vision encoder that extracts both CLS and patch tokens.
    
    Args:
        model_name: HuggingFace model name
        hidden_dim: Output hidden dimension
        use_lora: Whether to apply LoRA
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        freeze_embeddings: Whether to freeze embedding layers
    """
    
    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-384",
        hidden_dim: int = 768,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        freeze_embeddings: bool = True,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.use_lora = use_lora
        
        # Load SigLIP model
        logger.info(f"Loading vision encoder: {model_name}")
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.embeddings = self._resolve_embeddings()
        
        # Get model's hidden dimension
        model_hidden_dim = self.model.config.vision_config.hidden_size
        
        # Projection if dimensions don't match
        if model_hidden_dim != hidden_dim:
            self.projection = nn.Linear(model_hidden_dim, hidden_dim)
            logger.info(f"Added projection: {model_hidden_dim} -> {hidden_dim}")
        else:
            self.projection = nn.Identity()
        
        # Apply LoRA
        if use_lora:
            self._apply_lora(lora_rank, lora_alpha, lora_dropout)
        
        # Freeze embeddings if specified
        if freeze_embeddings:
            self._freeze_embeddings()
        
        # Log parameters
        self._log_parameters()
    
    def _apply_lora(self, rank: int, alpha: int, dropout: float):
        """Apply LoRA to vision model."""
        logger.info(f"Applying LoRA (rank={rank}, alpha={alpha})")
        
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        # Refresh embeddings reference after PEFT wrapping
        self.embeddings = self._resolve_embeddings()

    def _resolve_vision_model(self):
        """Resolve the underlying vision model regardless of PEFT wrapping."""
        if hasattr(self.model, 'base_model'):
            base_model = self.model.base_model
        else:
            base_model = self.model
        if hasattr(base_model, 'vision_model'):
            return base_model.vision_model
        if hasattr(base_model, 'model') and hasattr(base_model.model, 'vision_model'):
            return base_model.model.vision_model
        return None

    def _resolve_embeddings(self):
        """Resolve the vision embeddings module."""
        vision_model = self._resolve_vision_model()
        if vision_model is not None and hasattr(vision_model, 'embeddings'):
            return vision_model.embeddings
        return None
    
    def _freeze_embeddings(self):
        """Freeze embedding layers."""
        # Freeze patch embedding and position embedding
        self.embeddings = self._resolve_embeddings()
        if self.embeddings is not None:
            for param in self.embeddings.parameters():
                param.requires_grad = False
            logger.info("Froze vision embeddings")
    
    def _log_parameters(self):
        """Log trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Vision Encoder - Total: {total_params:,}, Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass extracting CLS and patch tokens.
        
        Args:
            pixel_values: Input images [batch, 3, H, W]
            
        Returns:
            cls_token: CLS token embeddings [batch, hidden_dim]
            patch_tokens: Patch token embeddings [batch, num_patches, hidden_dim]
        """
        # Get vision model outputs
        vision_model = self._resolve_vision_model()
        if vision_model is None:
            raise RuntimeError("Failed to resolve vision model for forward pass")
        
        # Forward through vision encoder
        outputs = vision_model(pixel_values=pixel_values)
        hidden_states = outputs.last_hidden_state  # [batch, num_tokens, hidden]
        
        # SigLIP: first token is CLS, rest are patches
        cls_token = hidden_states[:, 0, :]  # [batch, hidden]
        patch_tokens = hidden_states[:, 1:, :]  # [batch, num_patches, hidden]
        
        # Project if needed
        cls_token = self.projection(cls_token)
        patch_tokens = self.projection(patch_tokens)
        
        return cls_token, patch_tokens
    
    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Get pooled image features (CLS token only).
        
        Args:
            pixel_values: Input images [batch, 3, H, W]
            
        Returns:
            Image features [batch, hidden_dim]
        """
        cls_token, _ = self.forward(pixel_values)
        return cls_token
    
    @property
    def num_patches(self) -> int:
        """Get number of patch tokens."""
        # For 384x384 image with patch size 16: (384/16)^2 = 576
        image_size = self.model.config.vision_config.image_size
        patch_size = self.model.config.vision_config.patch_size
        return (image_size // patch_size) ** 2
