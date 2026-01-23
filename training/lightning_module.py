"""
PyTorch Lightning Module for CheXQuery-MedVLM.
"""
import logging
from typing import Any, Dict, List, Optional

import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup

from models.chexquery_medvlm import CheXQueryMedVLM

logger = logging.getLogger(__name__)


class CheXQueryLightningModule(pl.LightningModule):
    """
    PyTorch Lightning Module for CheXQuery-MedVLM training.
    
    Handles:
    - Three-phase training strategy
    - Combined generation + auxiliary loss
    - Freezing/unfreezing components per phase
    
    Args:
        model_config: Model configuration dictionary
        training_config: Training configuration dictionary
        phase: Current training phase (1, 2, or 3)
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        phase: int = 1,
        prompt_template: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_config = model_config
        self.training_config = training_config
        self.current_phase = phase
        self.prompt_template = prompt_template
        
        # Get phase-specific config
        self.phase_config = training_config.get(f"phase{phase}", {})
        
        # Build model
        self.model = CheXQueryMedVLM(
            vision_model_name=model_config.get("vision_encoder", {}).get("model_name", "google/siglip-base-patch16-384"),
            vision_hidden_dim=model_config.get("vision_encoder", {}).get("hidden_dim", 768),
            vision_use_lora=model_config.get("vision_encoder", {}).get("lora", {}).get("enabled", True),
            vision_lora_rank=model_config.get("vision_encoder", {}).get("lora", {}).get("rank", 8),
            num_condition_queries=model_config.get("condition_queries", {}).get("num_queries", 14),
            num_anatomical_queries=model_config.get("anatomical_queries", {}).get("num_queries", 6),
            num_cross_attn_layers=model_config.get("cross_attention", {}).get("num_layers", 2),
            num_attn_heads=model_config.get("cross_attention", {}).get("num_heads", 8),
            ffn_dim=model_config.get("cross_attention", {}).get("ffn_dim", 3072),
            num_pool_queries=model_config.get("gated_fusion", {}).get("num_pool_queries", 10),
            multi_view_enabled=model_config.get("multi_view", {}).get("enabled", False),
            decoder_model_name=model_config.get("text_decoder", {}).get("model_name", "google/flan-t5-base"),
            decoder_use_lora=model_config.get("text_decoder", {}).get("lora", {}).get("enabled", True),
            decoder_lora_rank=model_config.get("text_decoder", {}).get("lora", {}).get("rank", 16),
            max_length=model_config.get("text_decoder", {}).get("max_length", 512),
            prompt_template=prompt_template,
            hidden_dim=model_config.get("vision_encoder", {}).get("hidden_dim", 768),
            dropout=model_config.get("cross_attention", {}).get("dropout", 0.1),
        )
        
        # Label-specific auxiliary weights
        self.auxiliary_label_weights = self._build_auxiliary_label_weights()
        
        # Loss weights
        loss_config = self.phase_config.get("loss", {})
        self.generation_weight = loss_config.get("generation_weight", 1.0)
        self.auxiliary_weight = loss_config.get("auxiliary_weight", 0.3)
        self.label_smoothing = loss_config.get("label_smoothing", 0.0)
        
        # Apply phase-specific freezing
        self._apply_phase_freezing()
    
    def _apply_phase_freezing(self):
        """Apply freezing based on current phase."""
        freeze_modules = self.phase_config.get("freeze", [])
        
        for module_name in freeze_modules:
            module = self._get_module_by_name(module_name)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = False
                logger.info(f"Phase {self.current_phase}: Froze {module_name}")
    
    def _get_module_by_name(self, name: str):
        """Get module by name string."""
        parts = name.split(".")
        module = self.model
        
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        
        return module

    def _normalize_label_key(self, name: str) -> str:
        return "".join(ch.lower() for ch in name if ch.isalnum())

    def _build_auxiliary_label_weights(self) -> Optional[torch.Tensor]:
        weights_config = (
            self.training_config.get("training", {}).get("auxiliary_label_weights")
            or {}
        )
        if not weights_config:
            return None
        # Normalize keys for case/spacing differences
        normalized = {
            self._normalize_label_key(k): float(v) for k, v in weights_config.items()
        }
        label_names = self.model.get_condition_names()
        weights = []
        for name in label_names:
            key = self._normalize_label_key(name)
            weights.append(normalized.get(key, 1.0))
        return torch.tensor(weights, dtype=torch.float32)
    
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass."""
        pixel_values_lateral = batch.get("images_lateral")
        outputs = self.model(
            pixel_values=batch.get("images", batch.get("images_frontal")),
            pixel_values_lateral=pixel_values_lateral,
            texts=batch["texts"],
            chexbert_labels=batch.get("chexbert_labels"),
            chexbert_mask=batch.get("chexbert_mask"),
            auxiliary_label_weights=self.auxiliary_label_weights.to(self.device)
            if self.auxiliary_label_weights is not None
            else None,
        )
        return outputs
    
    def _compute_total_loss(
        self,
        generation_loss: torch.Tensor,
        auxiliary_loss: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute weighted total loss."""
        total_loss = self.generation_weight * generation_loss
        
        if auxiliary_loss is not None and self.auxiliary_weight > 0:
            total_loss = total_loss + self.auxiliary_weight * auxiliary_loss
        
        return total_loss
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        outputs = self.forward(batch)
        
        # Compute total loss
        total_loss = self._compute_total_loss(
            outputs["generation_loss"],
            outputs["auxiliary_loss"],
        )
        
        # Logging
        self.log("train/loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("train/gen_loss", outputs["generation_loss"], sync_dist=True)
        
        if outputs["auxiliary_loss"] is not None:
            self.log("train/aux_loss", outputs["auxiliary_loss"], sync_dist=True)
        
        if outputs["gate_values"] is not None:
            gate_mean = outputs["gate_values"].mean()
            self.log("train/gate_value", gate_mean, sync_dist=True)
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """Validation step."""
        outputs = self.forward(batch)
        
        total_loss = self._compute_total_loss(
            outputs["generation_loss"],
            outputs["auxiliary_loss"],
        )
        
        # Logging
        self.log("val/loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("val/gen_loss", outputs["generation_loss"], sync_dist=True)
        
        if outputs["auxiliary_loss"] is not None:
            self.log("val/aux_loss", outputs["auxiliary_loss"], sync_dist=True)
        
        # Auxiliary label precision/recall (per label)
        if outputs.get("auxiliary_logits") is not None and batch.get("chexbert_labels") is not None:
            logits = outputs["auxiliary_logits"]
            targets = batch["chexbert_labels"].to(logits.device)
            mask = batch.get("chexbert_mask")
            if mask is not None:
                mask = mask.to(logits.device).float()
            preds = (torch.sigmoid(logits) >= 0.5).float()
            label_names = self.model.get_condition_names()
            for idx, name in enumerate(label_names):
                if mask is not None:
                    valid = mask > 0
                    pred_col = preds[:, idx][valid]
                    tgt_col = targets[:, idx][valid]
                else:
                    pred_col = preds[:, idx]
                    tgt_col = targets[:, idx]
                tp = ((pred_col == 1) & (tgt_col == 1)).sum().float()
                fp = ((pred_col == 1) & (tgt_col == 0)).sum().float()
                fn = ((pred_col == 0) & (tgt_col == 1)).sum().float()
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                key = "".join(ch.lower() for ch in name if ch.isalnum())
                self.log(f"val/aux_precision_{key}", precision, sync_dist=True)
                self.log(f"val/aux_recall_{key}", recall, sync_dist=True)
        
        return {"val_loss": total_loss}
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        optimizer_config = self.phase_config.get("optimizer", {})
        scheduler_config = self.phase_config.get("scheduler", {})
        
        # Get trainable parameters
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        
        # Optimizer
        optimizer = AdamW(
            trainable_params,
            lr=float(optimizer_config.get("lr", 1e-4)),
            weight_decay=float(optimizer_config.get("weight_decay", 0.01)),
            betas=tuple(optimizer_config.get("betas", [0.9, 0.999])),
            eps=float(optimizer_config.get("eps", 1e-8)),
        )
        
        # Scheduler
        scheduler_type = scheduler_config.get("type", "cosine_with_warmup")
        
        if scheduler_type == "cosine_with_warmup":
            warmup_steps = scheduler_config.get("warmup_steps", 500)
            total_steps = self.trainer.estimated_stepping_batches
            
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        else:
            # Cosine annealing without warmup
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=float(scheduler_config.get("min_lr", 1e-7)),
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
    
    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        images_lateral: Optional[torch.Tensor] = None,
        max_length: int = 512,
        num_beams: int = 4,
        **kwargs,
    ) -> List[str]:
        """Generate reports from images."""
        self.model.eval()
        return self.model.generate(
            pixel_values=images,
            pixel_values_lateral=images_lateral,
            max_length=max_length,
            num_beams=num_beams,
            **kwargs,
        )
