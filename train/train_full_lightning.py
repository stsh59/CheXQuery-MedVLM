"""
Train full pipeline (SigLIP + Projection + BioGPT) with PyTorch Lightning.
"""
import argparse
import yaml
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from data.datamodule import IUXrayGenerationDataModule
from models.full_pipeline_lightning import FullPipelineLightning
from utils.config import OUTPUT_DIR, CHECKPOINT_DIR, RANDOM_SEED
from utils.logger import setup_logger

logger = setup_logger(__name__)


def train_full_pipeline(args):
    """
    Train full pipeline using PyTorch Lightning.
    
    Args:
        args: Command line arguments
    """
    config_path = Path(args.config) if args.config else Path(__file__).parent / 'train_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    exp_name = "full_pipeline"
    logger.info(f"Training full pipeline (SigLIP + Projection + BioGPT)")
    
    pl.seed_everything(config.get('seed', RANDOM_SEED))
    
    datamodule = IUXrayGenerationDataModule(
        batch_size=config.get('batch_size', 8),
        num_workers=config.get('num_workers', 4),
        projection_type=config.get('projection_type')
    )
    
    model = FullPipelineLightning(
        siglip_checkpoint=args.siglip_checkpoint,
        freeze_siglip=args.freeze_siglip,
        learning_rate=float(config.get('learning_rate', 1e-4)),
        weight_decay=float(config.get('weight_decay', 0.01)),
        warmup_steps=int(config.get('warmup_steps', 100))
    )
    
    checkpoint_dir = CHECKPOINT_DIR / exp_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='epoch_{epoch:02d}-val_loss_{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    tb_logger = TensorBoardLogger(
        save_dir=OUTPUT_DIR / 'lightning_logs',
        name=exp_name
    )
    
    num_epochs = getattr(args, 'num_epochs', None)
    if num_epochs is None:
        num_epochs = config.get('num_epochs', 10)
        
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator='auto',
        devices='auto',
        precision='16-mixed' if config.get('use_amp', True) else 32,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=tb_logger,
        log_every_n_steps=config.get('log_every_n_steps', 10),
        gradient_clip_val=config.get('max_grad_norm', 1.0),
        deterministic=True,
        num_sanity_val_steps=0
    )
    
    logger.info("Starting training with PyTorch Lightning Trainer...")
    try:
        trainer.fit(model, datamodule)
    except Exception as e:
        logger.error(f"Training interrupted: {e}")
    
    # Manual save to ensure checkpoint exists even if teardown crashes
    manual_ckpt_path = checkpoint_dir / "manual_save.ckpt"
    trainer.save_checkpoint(manual_ckpt_path)
    logger.info(f"Manually saved checkpoint to: {manual_ckpt_path}")
    
    logger.info(f"Training completed! Best model saved to: {checkpoint_callback.best_model_path}")
    
    return trainer, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train full pipeline with Lightning")
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--siglip_checkpoint', type=str, help='Path to pretrained SigLIP checkpoint')
    parser.add_argument('--freeze_siglip', action='store_true', default=True, help='Freeze SigLIP encoder')
    
    args = parser.parse_args()
    train_full_pipeline(args)

