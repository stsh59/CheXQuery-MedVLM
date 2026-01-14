"""
Full pipeline (SigLIP + Projection + BioGPT) as PyTorch Lightning Module.
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup
from typing import Optional, List

from models.medical_siglip_lightning import MedicalSigLIPLightning
from models.projection import ProjectionLayer
from models.biogpt_generator import BioGPTGenerator
from utils.config import LEARNING_RATE, WEIGHT_DECAY, WARMUP_STEPS, MAX_GENERATION_LENGTH


class FullPipelineLightning(pl.LightningModule):
    """
    Full pipeline Lightning Module: SigLIP encoder + projection + BioGPT.
    
    Args:
        siglip_checkpoint: Path to pretrained SigLIP checkpoint
        freeze_siglip: Whether to freeze SigLIP encoder
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_steps: Warmup steps
    """
    
    def __init__(
        self,
        siglip_checkpoint: Optional[str] = None,
        freeze_siglip: bool = True,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        warmup_steps: int = WARMUP_STEPS
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        
        if siglip_checkpoint:
            # Load pretrained SigLIP using Lightning's method
            self.siglip = MedicalSigLIPLightning.load_from_checkpoint(siglip_checkpoint)
        else:
            self.siglip = MedicalSigLIPLightning(peft_method='lora')
        
        if freeze_siglip:
            for param in self.siglip.parameters():
                param.requires_grad = False
        
        self.projection = ProjectionLayer(input_dim=768, output_dim=1024)
        
        self.biogpt = BioGPTGenerator(freeze_encoder=False)
        
        if hasattr(self.biogpt.model, "gradient_checkpointing_enable"):
            self.biogpt.model.gradient_checkpointing_enable()
        
        self.tokenizer = self.biogpt.tokenizer
    
    def forward(self, images, input_ids, attention_mask, labels=None):
        """Forward pass."""
        image_embeds = self.siglip.encode_image(images)
        
        projected_embeds = self.projection(image_embeds)
        
        outputs = self.biogpt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_embeds=projected_embeds,
            labels=labels
        )
        
        return outputs
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        images, texts, metadata = batch
        images = images.to(self.device)
        
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        outputs = self(images, input_ids, attention_mask, labels=labels)
        loss = outputs.loss
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images, texts, metadata = batch
        images = images.to(self.device)
        
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        outputs = self(images, input_ids, attention_mask, labels=labels)
        loss = outputs.loss
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
    
    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        max_length: int = MAX_GENERATION_LENGTH,
        num_beams: int = 4
    ) -> List[str]:
        """
        Generate text from images.
        
        Args:
            images: Image tensors
            max_length: Maximum generation length
            num_beams: Number of beams
        
        Returns:
            List of generated texts
        """
        image_embeds = self.siglip.encode_image(images)
        projected_embeds = self.projection(image_embeds)
        
        generated = self.biogpt.generate(
            projected_embeds,
            max_length=max_length,
            num_beams=num_beams
        )
        
        return generated

