"""
Training orchestration for CheXQuery-MedVLM.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from data.datamodule import ChestXrayDataModule
from training.lightning_module import CheXQueryLightningModule

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_model(
    model_config_path: str = "configs/model_config.yaml",
    train_config_path: str = "configs/train_config.yaml",
    data_config_path: str = "configs/data_config.yaml",
    phase: int = 1,
    checkpoint_path: Optional[str] = None,
    resume_from: Optional[str] = None,
) -> pl.Trainer:
    """
    Train CheXQuery-MedVLM model.
    
    Args:
        model_config_path: Path to model configuration
        train_config_path: Path to training configuration
        data_config_path: Path to data configuration
        phase: Training phase (1, 2, or 3)
        checkpoint_path: Path to save checkpoints
        resume_from: Path to checkpoint to resume from
        
    Returns:
        Trained Lightning Trainer
    """
    # Load configs
    model_config = load_config(model_config_path)
    train_config = load_config(train_config_path)
    data_config = load_config(data_config_path)
    
    phase_config = train_config.get(f"phase{phase}", {})
    training_settings = train_config.get("training", {})
    prompt_template = data_config.get("text", {}).get("prompt_template")
    
    logger.info(f"Starting Phase {phase} training: {phase_config.get('name', 'unknown')}")
    
    # Set seed
    seed = train_config.get("seed", 42)
    pl.seed_everything(seed, workers=True)
    
    # Data module
    datamodule = ChestXrayDataModule(
        batch_size=phase_config.get("batch_size", 8),
        num_workers=training_settings.get("num_workers", 4),
        image_size=data_config.get("image", {}).get("size", 384),
        projection_type=data_config.get("filtering", {}).get("projection_type", "Frontal"),
        splits_file=data_config.get("splits", {}).get("split_file"),
        text_output_template=data_config.get("text", {}).get("output_template"),
        text_max_length=data_config.get("text", {}).get("max_length", 512),
        image_mean=data_config.get("image", {}).get("mean"),
        image_std=data_config.get("image", {}).get("std"),
        augmentation_config=data_config.get("augmentation", {}),
        use_siglip_processor=data_config.get("image", {}).get("use_siglip_processor", False),
        processor_model=data_config.get("image", {}).get("processor_model"),
    )
    
    # Model
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        model = CheXQueryLightningModule.load_from_checkpoint(
            resume_from,
            model_config=model_config,
            training_config=train_config,
            phase=phase,
            prompt_template=prompt_template,
        )
    else:
        model = CheXQueryLightningModule(
            model_config=model_config,
            training_config=train_config,
            phase=phase,
            prompt_template=prompt_template,
        )
    
    # Callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_dir = checkpoint_path or train_config.get("checkpoint_dir", "outputs/checkpoints")
    checkpoint_dir = Path(checkpoint_dir) / f"phase{phase}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch_{epoch:02d}-val_loss_{val/loss:.4f}",
        monitor=training_settings.get("monitor_metric", "val/loss"),
        mode=training_settings.get("monitor_mode", "min"),
        save_top_k=training_settings.get("save_top_k", 3),
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    # Early stopping
    early_stop_config = training_settings.get("early_stopping", {})
    if early_stop_config.get("enabled", True):
        early_stopping = EarlyStopping(
            monitor=training_settings.get("monitor_metric", "val/loss"),
            patience=early_stop_config.get("patience", 5),
            min_delta=early_stop_config.get("min_delta", 0.001),
            mode=training_settings.get("monitor_mode", "min"),
        )
        callbacks.append(early_stopping)
    
    # Logger
    log_dir = train_config.get("log_dir", "outputs/logs")
    
    if training_settings.get("use_wandb", False):
        tb_logger = WandbLogger(
            project=training_settings.get("wandb_project", "chexquery-medvlm"),
            name=f"phase{phase}",
            save_dir=log_dir,
        )
    else:
        tb_logger = TensorBoardLogger(
            save_dir=log_dir,
            name=f"phase{phase}",
        )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=phase_config.get("epochs", 10),
        accelerator="auto",
        devices="auto",
        precision=training_settings.get("precision", "bf16-mixed"),
        accumulate_grad_batches=phase_config.get("gradient_accumulation_steps", 1),
        gradient_clip_val=training_settings.get("gradient_clip_val", 1.0),
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=training_settings.get("log_every_n_steps", 10),
        val_check_interval=1.0,
        deterministic=True,
        enable_checkpointing=True,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.fit(model, datamodule)
    
    logger.info(f"Training complete. Best model: {checkpoint_callback.best_model_path}")
    
    return trainer


def train_all_phases(
    model_config_path: str = "configs/model_config.yaml",
    train_config_path: str = "configs/train_config.yaml",
    data_config_path: str = "configs/data_config.yaml",
):
    """
    Train all phases sequentially.
    
    Args:
        model_config_path: Path to model configuration
        train_config_path: Path to training configuration
        data_config_path: Path to data configuration
    """
    train_config = load_config(train_config_path)
    
    # Phase 1: Query Alignment
    logger.info("=" * 60)
    logger.info("PHASE 1: Query Alignment")
    logger.info("=" * 60)
    trainer1 = train_model(
        model_config_path=model_config_path,
        train_config_path=train_config_path,
        data_config_path=data_config_path,
        phase=1,
    )
    phase1_checkpoint = trainer1.checkpoint_callback.best_model_path
    
    # Phase 2: End-to-End Fine-tuning
    logger.info("=" * 60)
    logger.info("PHASE 2: End-to-End Fine-tuning")
    logger.info("=" * 60)
    trainer2 = train_model(
        model_config_path=model_config_path,
        train_config_path=train_config_path,
        data_config_path=data_config_path,
        phase=2,
        resume_from=phase1_checkpoint,
    )
    phase2_checkpoint = trainer2.checkpoint_callback.best_model_path
    
    # Phase 3: Generation Fine-tuning (optional)
    phase3_config = train_config.get("phase3", {})
    if phase3_config.get("enabled", False):
        logger.info("=" * 60)
        logger.info("PHASE 3: Generation Fine-tuning")
        logger.info("=" * 60)
        trainer3 = train_model(
            model_config_path=model_config_path,
            train_config_path=train_config_path,
            data_config_path=data_config_path,
            phase=3,
            resume_from=phase2_checkpoint,
        )
    
    logger.info("=" * 60)
    logger.info("All training phases complete!")
    logger.info("=" * 60)
