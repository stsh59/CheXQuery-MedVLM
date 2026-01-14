"""
BioGPT integration for medical report generation.
"""
import torch
import torch.nn as nn
from transformers import BioGptForCausalLM, BioGptTokenizer
from typing import Optional

from utils.config import BIOGPT_MODEL_NAME, MAX_GENERATION_LENGTH


class BioGPTGenerator(nn.Module):
    """
    BioGPT model for generating medical impressions.
    
    Args:
        model_name: HuggingFace model name
        freeze_encoder: Whether to freeze the encoder layers
    """
    
    def __init__(
        self,
        model_name: str = BIOGPT_MODEL_NAME,
        freeze_encoder: bool = False
    ):
        super().__init__()
        
        self.tokenizer = BioGptTokenizer.from_pretrained(model_name)
        self.model = BioGptForCausalLM.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        
        self.hidden_size = self.model.config.hidden_size
        
        if freeze_encoder:
            for name, param in self.model.named_parameters():
                if 'lm_head' not in name:
                    param.requires_grad = False
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        """
        Forward pass with optional image conditioning.
        
        Args:
            input_ids: Text token IDs
            attention_mask: Attention mask
            image_embeds: Optional projected image embeddings [batch_size, hidden_size]
            labels: Optional labels for training
        
        Returns:
            Model outputs
        """
        if image_embeds is not None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
            
            image_embeds = image_embeds.unsqueeze(1)
            
            inputs_embeds = torch.cat([image_embeds, inputs_embeds], dim=1)
            
            image_attention = torch.ones(
                (attention_mask.shape[0], 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            attention_mask = torch.cat([image_attention, attention_mask], dim=1)
            
            if labels is not None:
                image_labels = torch.full(
                    (labels.shape[0], 1),
                    fill_value=-100,
                    dtype=labels.dtype,
                    device=labels.device
                )
                labels = torch.cat([image_labels, labels], dim=1)
            
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels
            )
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        
        return outputs
    
    @torch.no_grad()
    def generate(
        self,
        image_embeds: torch.Tensor,
        max_length: int = MAX_GENERATION_LENGTH,
        num_beams: int = 4,
        temperature: float = 1.0,
        top_p: float = 0.9
    ):
        """
        Generate text from image embeddings.
        
        Args:
            image_embeds: Projected image embeddings
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
        
        Returns:
            Generated text
        """
        batch_size = image_embeds.shape[0]
        
        start_tokens = torch.full(
            (batch_size, 1),
            fill_value=self.tokenizer.bos_token_id if self.tokenizer.bos_token_id else self.tokenizer.eos_token_id,
            dtype=torch.long,
            device=image_embeds.device
        )
        
        start_embeds = self.model.get_input_embeddings()(start_tokens)
        
        image_embeds = image_embeds.unsqueeze(1)
        inputs_embeds = torch.cat([image_embeds, start_embeds], dim=1)
        
        attention_mask = torch.ones(
            (batch_size, inputs_embeds.shape[1]),
            dtype=torch.long,
            device=image_embeds.device
        )
        
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            do_sample=True if temperature > 0 else False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return generated_texts

