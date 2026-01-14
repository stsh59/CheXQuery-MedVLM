"""
CheXQuery-MedVLM: Main Model Integration.

This is the main model class that integrates all components:
- SigLIP Vision Encoder
- CheXbert-Initialized Condition Queries
- Anatomical Region Queries
- Cross-Attention Module
- Gated Fusion Module
- Flan-T5 Text Decoder
- Auxiliary Classification Head
"""
import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from transformers.modeling_outputs import Seq2SeqLMOutput

from models.vision_encoder import VisionEncoder
from models.condition_queries import ConditionQueryModule
from models.anatomical_queries import AnatomicalQueryModule
from models.cross_attention import CrossAttentionModule
from models.gated_fusion import GatedFusionModule, QuerySplitter
from models.text_decoder import TextDecoder
from models.auxiliary_head import AuxiliaryClassificationHead
from data.preprocessing import get_prompt_template

logger = logging.getLogger(__name__)


class CheXQueryMedVLM(nn.Module):
    """
    CheXQuery-MedVLM: Anatomical Region-Guided Medical Vision-Language Model.
    
    A novel architecture for chest X-ray report generation featuring:
    1. CheXbert-initialized condition queries for pathology detection
    2. Anatomical region queries for spatial grounding
    3. Gated fusion for adaptive global/local balance
    4. Multi-task learning with auxiliary classification
    
    Args:
        vision_config: Vision encoder configuration
        condition_config: Condition query configuration
        anatomical_config: Anatomical query configuration
        cross_attention_config: Cross-attention configuration
        fusion_config: Gated fusion configuration
        decoder_config: Text decoder configuration
        auxiliary_config: Auxiliary head configuration
    """
    
    def __init__(
        self,
        # Vision encoder
        vision_model_name: str = "google/siglip-base-patch16-384",
        vision_hidden_dim: int = 768,
        vision_use_lora: bool = True,
        vision_lora_rank: int = 8,
        # Condition queries
        num_condition_queries: int = 14,
        # Anatomical queries
        num_anatomical_queries: int = 6,
        # Cross-attention
        num_cross_attn_layers: int = 2,
        num_attn_heads: int = 8,
        ffn_dim: int = 3072,
        # Gated fusion
        num_pool_queries: int = 10,
        # Text decoder
        decoder_model_name: str = "google/flan-t5-base",
        decoder_use_lora: bool = True,
        decoder_lora_rank: int = 16,
        max_length: int = 512,
        prompt_template: Optional[str] = None,
        # General
        hidden_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_condition_queries = num_condition_queries
        self.num_anatomical_queries = num_anatomical_queries
        self.num_total_queries = num_condition_queries + num_anatomical_queries
        # Prompt to guide the decoder output format
        self.prompt_template = prompt_template or get_prompt_template()
        
        # ============ Vision Encoder ============
        self.vision_encoder = VisionEncoder(
            model_name=vision_model_name,
            hidden_dim=hidden_dim,
            use_lora=vision_use_lora,
            lora_rank=vision_lora_rank,
        )
        
        # ============ Query Modules ============
        self.condition_queries = ConditionQueryModule(
            num_queries=num_condition_queries,
            hidden_dim=hidden_dim,
        )
        
        self.anatomical_queries = AnatomicalQueryModule(
            num_queries=num_anatomical_queries,
            hidden_dim=hidden_dim,
        )
        
        # ============ Cross-Attention ============
        self.cross_attention = CrossAttentionModule(
            num_layers=num_cross_attn_layers,
            hidden_dim=hidden_dim,
            num_heads=num_attn_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        
        # ============ Query Splitter ============
        self.query_splitter = QuerySplitter(
            num_condition_queries=num_condition_queries,
            num_anatomical_queries=num_anatomical_queries,
        )
        
        # ============ Gated Fusion ============
        self.gated_fusion = GatedFusionModule(
            hidden_dim=hidden_dim,
            num_condition_queries=num_condition_queries,
            num_anatomical_queries=num_anatomical_queries,
            num_pool_queries=num_pool_queries,
            dropout=dropout,
        )
        
        # ============ Text Decoder ============
        self.text_decoder = TextDecoder(
            model_name=decoder_model_name,
            hidden_dim=hidden_dim,
            max_length=max_length,
            use_lora=decoder_use_lora,
            lora_rank=decoder_lora_rank,
        )
        
        # ============ Auxiliary Head ============
        self.auxiliary_head = AuxiliaryClassificationHead(
            hidden_dim=hidden_dim,
            num_classes=num_condition_queries,
            dropout=dropout,
        )
        
        # Store tokenizer reference
        self.tokenizer = self.text_decoder.tokenizer
        
        logger.info("CheXQuery-MedVLM initialized successfully")
        self._log_total_parameters()
    
    def _strip_prompt_prefix(self, text: str) -> str:
        """Remove prompt template from generated text if present."""
        if not text:
            return text
        prompt = (self.prompt_template or "").strip()
        if not prompt:
            return text.strip()
        if text.lstrip().startswith(prompt):
            stripped = text.lstrip()[len(prompt):]
            return stripped.lstrip()
        return text.strip()
    
    def _log_total_parameters(self):
        """Log total model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total:,}")
        logger.info(f"Trainable parameters: {trainable:,} ({100*trainable/total:.2f}%)")
    
    def encode_image(
        self,
        pixel_values: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Encode image and apply query-based attention.
        
        Args:
            pixel_values: Input images [batch, 3, H, W]
            return_attention: Whether to return attention weights
            
        Returns:
            visual_tokens: Final visual tokens for decoder
            attn_weights: Optional attention weights
            condition_embeds: Condition query embeddings (for auxiliary head)
            gate_values: Fusion gate values
        """
        batch_size = pixel_values.shape[0]
        
        # Get CLS and patch tokens from vision encoder
        cls_token, patch_tokens = self.vision_encoder(pixel_values)
        
        # Get queries
        condition_queries = self.condition_queries(batch_size)
        anatomical_queries = self.anatomical_queries(batch_size)
        
        # Combine queries
        combined_queries = torch.cat(
            [condition_queries, anatomical_queries],
            dim=1
        )  # [batch, 20, hidden]
        
        # Cross-attention: queries attend to patch tokens
        attended_queries, attn_weights = self.cross_attention(
            queries=combined_queries,
            patch_tokens=patch_tokens,
            return_attention=return_attention,
        )
        
        # Split back to get condition embeddings for auxiliary head
        condition_embeds, anatomical_embeds = self.query_splitter(attended_queries)
        
        # Gated fusion
        visual_tokens, gate_values = self.gated_fusion(
            cls_token=cls_token,
            query_embeddings=attended_queries,
        )
        
        return visual_tokens, attn_weights, condition_embeds, gate_values
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        texts: Optional[List[str]] = None,
        labels: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        chexbert_labels: Optional[torch.Tensor] = None,
        chexbert_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass for training.
        
        Args:
            pixel_values: Input images [batch, 3, H, W]
            texts: Target text strings (used to create labels if not provided)
            labels: Pre-tokenized target labels
            decoder_input_ids: Decoder input IDs
            decoder_attention_mask: Decoder attention mask
            chexbert_labels: CheXbert binary labels [batch, 14]
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with loss, logits, and optional attention weights
        """
        # Encode image
        visual_tokens, attn_weights, condition_embeds, gate_values = self.encode_image(
            pixel_values,
            return_attention=return_attention,
        )
        
        # Prepare labels if texts provided
        if labels is None and texts is not None:
            # Prepend prompt to targets so the decoder learns to follow the format
            prompted_texts = [f"{self.prompt_template}\n{text}" for text in texts]
            label_dict = self.text_decoder.prepare_labels(prompted_texts)
            labels = label_dict["labels"].to(pixel_values.device)
            decoder_input_ids = label_dict["decoder_input_ids"].to(pixel_values.device)
            decoder_attention_mask = label_dict["decoder_attention_mask"].to(pixel_values.device)
        
        # Text decoder forward
        decoder_outputs = self.text_decoder(
            visual_tokens=visual_tokens,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )
        
        generation_loss = decoder_outputs.loss
        
        # Auxiliary classification
        auxiliary_loss = None
        auxiliary_logits = None
        if chexbert_labels is not None:
            auxiliary_logits = self.auxiliary_head(condition_embeds)
            auxiliary_loss = self.auxiliary_head.compute_loss(
                auxiliary_logits, chexbert_labels, mask=chexbert_mask
            )
        
        return {
            "loss": generation_loss,
            "generation_loss": generation_loss,
            "auxiliary_loss": auxiliary_loss,
            "logits": decoder_outputs.logits,
            "auxiliary_logits": auxiliary_logits,
            "attention_weights": attn_weights,
            "gate_values": gate_values,
        }
    
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        max_length: int = 512,
        num_beams: int = 4,
        **kwargs,
    ) -> List[str]:
        """
        Generate reports from images.
        
        Args:
            pixel_values: Input images [batch, 3, H, W]
            max_length: Maximum generation length
            num_beams: Number of beams
            **kwargs: Additional generation arguments
            
        Returns:
            List of generated report strings
        """
        # Encode image
        visual_tokens, _, _, _ = self.encode_image(pixel_values)
        batch_size = visual_tokens.size(0)
        
        # Build decoder prompt ids/mask to enforce output format
        prompt_ids = self.text_decoder.tokenizer(
            self.prompt_template,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(visual_tokens.device)
        # Expand prompt to batch
        prompt_ids = prompt_ids.expand(batch_size, -1)
        prompt_attn = torch.ones_like(prompt_ids, device=visual_tokens.device)

        # Generate text
        generated_texts = self.text_decoder.generate(
            visual_tokens=visual_tokens,
            max_length=max_length,
            num_beams=num_beams,
            decoder_input_ids=prompt_ids,
            decoder_attention_mask=prompt_attn,
            **kwargs,
        )
        # Strip prompt template from outputs if it appears in the decoded text
        return [self._strip_prompt_prefix(t) for t in generated_texts]
    
    @torch.no_grad()
    def generate_with_attention(
        self,
        pixel_values: torch.Tensor,
        max_length: int = 512,
        num_beams: int = 4,
        **kwargs,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """
        Generate reports with attention visualization.
        
        Args:
            pixel_values: Input images
            max_length: Maximum generation length
            num_beams: Number of beams
            
        Returns:
            generated_texts: List of generated texts
            attention_weights: Attention weights [batch, num_queries, num_patches]
            gate_values: Gate values [batch, 1]
        """
        # Encode with attention
        visual_tokens, attn_weights, _, gate_values = self.encode_image(
            pixel_values,
            return_attention=True,
        )
        
        # Generate
        generated_texts = self.text_decoder.generate(
            visual_tokens=visual_tokens,
            max_length=max_length,
            num_beams=num_beams,
            **kwargs,
        )
        
        return generated_texts, attn_weights, gate_values
    
    def get_condition_names(self) -> List[str]:
        """Get condition query names."""
        return self.condition_queries.get_condition_names()
    
    def get_region_names(self) -> List[str]:
        """Get anatomical region names."""
        return self.anatomical_queries.get_region_names()
