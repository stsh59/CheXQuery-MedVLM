"""
Medical-SigLIP as PyTorch Lightning Module.
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel, AutoProcessor, get_cosine_schedule_with_warmup
from typing import Optional

from utils.config import SIGLIP_MODEL_NAME, LEARNING_RATE, WEIGHT_DECAY, WARMUP_STEPS
from models.peft_config import (
    get_lora_config, get_qlora_config, apply_lora, apply_qlora, count_trainable_parameters
)


class ContrastiveLoss(nn.Module):
    """Contrastive loss for image-text alignment."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor):
        """Compute contrastive loss."""
        logits = torch.matmul(image_embeds, text_embeds.t()) / self.temperature
        
        batch_size = logits.shape[0]
        labels = torch.arange(batch_size, device=logits.device)
        
        loss_i2t = self.cross_entropy(logits, labels)
        loss_t2i = self.cross_entropy(logits.t(), labels)
        
        loss = (loss_i2t + loss_t2i) / 2
        
        return loss


class MedicalSigLIPLightning(pl.LightningModule):
    """
    Medical-SigLIP Lightning Module for image-text alignment.
    
    Args:
        model_name: HuggingFace model name
        peft_method: 'lora' or 'qlora'
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_steps: Warmup steps for scheduler
        temperature: Contrastive loss temperature
    """
    
    def __init__(
        self,
        model_name: str = SIGLIP_MODEL_NAME,
        peft_method: str = 'lora',
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        warmup_steps: int = WARMUP_STEPS,
        temperature: float = 0.07
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.peft_method = peft_method
        
        if peft_method == 'qlora':
            lora_config, bnb_config = get_qlora_config()
            self.model = AutoModel.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
            self.model = apply_qlora(self.model, lora_config)
        elif peft_method == 'lora':
            self.model = AutoModel.from_pretrained(model_name)
            lora_config = get_lora_config()
            self.model = apply_lora(self.model, lora_config)
        else:
            self.model = AutoModel.from_pretrained(model_name)
            
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
            
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        self.criterion = ContrastiveLoss(temperature=temperature)
        
        self.embed_dim = self.model.config.hidden_size if hasattr(self.model.config, 'hidden_size') else 768
        
        param_info = count_trainable_parameters(self.model)
        print(f"Trainable: {param_info['trainable_params']:,} / {param_info['all_params']:,} ({param_info['trainable_percentage']:.2f}%)")
    
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images to embeddings."""
        outputs = self.model.get_image_features(pixel_values=pixel_values)
        return outputs
    
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text to embeddings."""
        outputs = self.model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs
    
    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Forward pass."""
        image_embeds = self.encode_image(pixel_values)
        text_embeds = self.encode_text(input_ids, attention_mask)
        
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        
        return image_embeds, text_embeds
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        images = batch['image']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        image_embeds, text_embeds = self(images, input_ids, attention_mask)
        
        loss = self.criterion(image_embeds, text_embeds)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images = batch['image']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        image_embeds, text_embeds = self(images, input_ids, attention_mask)
        
        loss = self.criterion(image_embeds, text_embeds)
        
        similarities = (image_embeds * text_embeds).sum(dim=-1).mean()
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_similarity', similarities, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = get_cosine_schedule_with_warmup(
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

