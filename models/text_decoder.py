"""
Flan-T5 Text Decoder with LoRA.

This module wraps Flan-T5 for text generation, using visual tokens
as encoder hidden states for cross-attention.
"""
import logging
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput
from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)


class TextDecoder(nn.Module):
    """
    Flan-T5 based text decoder with LoRA.
    
    Uses visual tokens as encoder hidden states, enabling cross-attention
    between generated text and visual features.
    
    Args:
        model_name: HuggingFace model name
        hidden_dim: Expected hidden dimension (should match visual tokens)
        max_length: Maximum generation length
        use_lora: Whether to apply LoRA
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
    """
    
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        hidden_dim: int = 768,
        max_length: int = 512,
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.use_lora = use_lora
        
        # Load model and tokenizer
        logger.info(f"Loading text decoder: {model_name}")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # Verify dimensions match
        model_hidden = self.model.config.d_model
        if model_hidden != hidden_dim:
            logger.warning(
                f"Hidden dim mismatch: visual={hidden_dim}, T5={model_hidden}. "
                f"Adding projection layer."
            )
            self.visual_projection = nn.Linear(hidden_dim, model_hidden)
        else:
            self.visual_projection = nn.Identity()
        
        # Apply LoRA
        if use_lora:
            self._apply_lora(lora_rank, lora_alpha, lora_dropout)
        
        # Log parameters
        self._log_parameters()
    
    def _apply_lora(self, rank: int, alpha: int, dropout: float):
        """Apply LoRA to T5 model."""
        logger.info(f"Applying LoRA to T5 (rank={rank}, alpha={alpha})")
        
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["q", "k", "v", "o"],
            bias="none",
        )
        
        self.model = get_peft_model(self.model, lora_config)
    
    def _log_parameters(self):
        """Log trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Text Decoder - Total: {total_params:,}, Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    def forward(
        self,
        visual_tokens: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Seq2SeqLMOutput:
        """
        Forward pass for training.
        
        Args:
            visual_tokens: Visual tokens from fusion module [batch, num_tokens, hidden]
            input_ids: Not used (we use visual tokens as encoder output)
            attention_mask: Attention mask for visual tokens
            decoder_input_ids: Decoder input token IDs
            decoder_attention_mask: Decoder attention mask
            labels: Target labels for loss computation
            
        Returns:
            Seq2SeqLMOutput with loss and logits
        """
        batch_size, num_visual_tokens, _ = visual_tokens.shape
        
        # Project visual tokens if needed
        visual_tokens = self.visual_projection(visual_tokens)
        
        # Create encoder outputs structure
        # T5 expects encoder_outputs as tuple: (hidden_states, )
        encoder_outputs = (visual_tokens,)
        
        # Create attention mask for visual tokens if not provided
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, num_visual_tokens,
                dtype=torch.long,
                device=visual_tokens.device
            )
        
        # Forward through T5
        outputs = self.model(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )
        
        return outputs
    
    @torch.no_grad()
    def generate(
        self,
        visual_tokens: torch.Tensor,
        max_length: Optional[int] = None,
        min_length: int = 20,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,
        early_stopping: bool = True,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> List[str]:
        """
        Generate text from visual tokens.
        
        Args:
            visual_tokens: Visual tokens [batch, num_tokens, hidden]
            max_length: Maximum generation length
            min_length: Minimum generation length
            num_beams: Number of beams for beam search
            length_penalty: Length penalty
            no_repeat_ngram_size: No repeat n-gram size
            early_stopping: Whether to stop early
            
        Returns:
            List of generated text strings
        """
        batch_size, num_visual_tokens, _ = visual_tokens.shape
        max_length = max_length or self.max_length
        
        # Project visual tokens
        visual_tokens = self.visual_projection(visual_tokens)
        
        # Create encoder outputs
        encoder_outputs = (visual_tokens,)
        
        # Create attention mask
        attention_mask = torch.ones(
            batch_size, num_visual_tokens,
            dtype=torch.long,
            device=visual_tokens.device
        )
        
        # If a prompt is provided, use it as decoder input
        gen_kwargs = dict(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
        )
        if decoder_input_ids is not None:
            gen_kwargs["decoder_input_ids"] = decoder_input_ids
        if decoder_attention_mask is not None:
            gen_kwargs["decoder_attention_mask"] = decoder_attention_mask
        
        # Generate
        generated_ids = self.model.generate(**gen_kwargs, **kwargs)
        
        # Decode
        generated_texts = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        
        return generated_texts
    
    def prepare_labels(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare labels for training.
        
        Args:
            texts: List of target texts
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        max_length = max_length or self.max_length
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        
        # Create labels (pad tokens set to -100 for loss masking)
        labels = encoded["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Shift decoder inputs to prevent target leakage
        decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels)
        
        return {
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": encoded["attention_mask"],
            "labels": labels,
        }
