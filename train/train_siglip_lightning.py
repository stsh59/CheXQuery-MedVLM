"""
Train Medical-SigLIP with PyTorch Lightning.
"""
import argparse
import yaml
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from data.datamodule import IUXrayDataModule
from models.medical_siglip_lightning import MedicalSigLIPLightning
from utils.config import OUTPUT_DIR, CHECKPOINT_DIR, RANDOM_SEED
from utils.logger import setup_logger

logger = setup_logger(__name__)


def train_siglip(args):
    """
    Train Medical-SigLIP using PyTorch Lightning.
    
    Args:
        args: Command line arguments
    """
    config_path = Path(args.config) if args.config else Path(__file__).parent / 'train_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.peft_method:
        config['peft_method'] = args.peft_method
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.num_epochs:
        config['num_epochs'] = args.num_epochs
    
    exp_name = f"siglip_{config['peft_method']}"
    logger.info(f"Training Medical-SigLIP with {config['peft_method'].upper()}")
    
    pl.seed_everything(config.get('seed', RANDOM_SEED))
    
    datamodule = IUXrayDataModule(
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 4),
        use_contrastive=True,
        projection_type=config.get('projection_type'),
        max_text_length=config.get('max_text_length', 128)
    )
    
    model = MedicalSigLIPLightning(
        peft_method=config['peft_method'],
        learning_rate=float(config['learning_rate']),
        weight_decay=float(config.get('weight_decay', 0.01)),
        warmup_steps=int(config.get('warmup_steps', 500)),
        temperature=float(config.get('temperature', 0.07))
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
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min',
        verbose=True
    )
    
    tb_logger = TensorBoardLogger(
        save_dir=OUTPUT_DIR / 'lightning_logs',
        name=exp_name
    )
    
    trainer = pl.Trainer(
        max_epochs=config['num_epochs'],
        accelerator='auto',
        devices='auto',
        precision='16-mixed' if config.get('use_amp', True) else 32,
        callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
        logger=tb_logger,
        log_every_n_steps=config.get('log_every_n_steps', 10),
        gradient_clip_val=config.get('max_grad_norm', 1.0),
        accumulate_grad_batches=config.get('gradient_accumulation_steps', 1),
        deterministic=True
    )
    
    logger.info("Starting training with PyTorch Lightning Trainer...")
    trainer.fit(model, datamodule)
    
    logger.info(f"Training completed! Best model saved to: {checkpoint_callback.best_model_path}")
    
    return trainer, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Medical-SigLIP with Lightning")
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--peft_method', type=str, choices=['lora', 'qlora'], help='PEFT method')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    
    args = parser.parse_args()
    train_siglip(args)

