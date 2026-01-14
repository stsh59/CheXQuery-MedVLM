"""
Compare LoRA vs QLoRA using PyTorch Lightning with automatic profiling.
"""
import time
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.profilers import SimpleProfiler
from pathlib import Path
import yaml
import json
import matplotlib.pyplot as plt
import pandas as pd

from data.datamodule import IUXrayDataModule
from models.medical_siglip_lightning import MedicalSigLIPLightning
from utils.config import OUTPUT_DIR, CHECKPOINT_DIR, RANDOM_SEED
from utils.logger import setup_logger

logger = setup_logger(__name__)


def train_and_measure(peft_method: str, config_path: str = None):
    """
    Train using Lightning and measure performance.
    
    Args:
        peft_method: 'lora' or 'qlora'
        config_path: Path to config file
    
    Returns:
        Dictionary with training stats
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training with {peft_method.upper()}")
    logger.info(f"{'='*60}\n")
    
    config_path = Path(config_path) if config_path else Path(__file__).parent.parent / 'train' / 'train_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['peft_method'] = peft_method
    
    pl.seed_everything(config.get('seed', RANDOM_SEED))
    
    datamodule = IUXrayDataModule(
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 4),
        use_contrastive=True,
        projection_type=config.get('projection_type'),
        max_text_length=config.get('max_text_length', 128)
    )
    
    model = MedicalSigLIPLightning(
        peft_method=peft_method,
        learning_rate=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.01),
        warmup_steps=config.get('warmup_steps', 500),
        temperature=config.get('temperature', 0.07)
    )
    
    exp_name = f"siglip_{peft_method}"
    checkpoint_dir = CHECKPOINT_DIR / exp_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    profiler = SimpleProfiler(
        dirpath=OUTPUT_DIR / 'profiling',
        filename=f'{peft_method}_profile'
    )
    
    trainer = pl.Trainer(
        max_epochs=config['num_epochs'],
        accelerator='auto',
        devices='auto',
        precision='16-mixed' if config.get('use_amp', True) else 32,
        callbacks=[checkpoint_callback, lr_monitor],
        profiler=profiler,
        log_every_n_steps=config.get('log_every_n_steps', 10),
        gradient_clip_val=config.get('max_grad_norm', 1.0),
        deterministic=True,
        enable_progress_bar=True
    )
    
    start_time = time.time()
    
    trainer.fit(model, datamodule)
    
    training_time = time.time() - start_time
    
    peak_gpu_memory = 0
    if torch.cuda.is_available():
        peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024 ** 3
        torch.cuda.reset_peak_memory_stats()
    
    checkpoint_data = torch.load(checkpoint_callback.best_model_path, map_location='cpu')
    best_val_loss = trainer.callback_metrics.get('val_loss', float('inf'))
    
    stats = {
        'peft_method': peft_method,
        'training_time_minutes': training_time / 60,
        'peak_gpu_gb': peak_gpu_memory,
        'best_val_loss': float(best_val_loss),
        'total_epochs': trainer.current_epoch + 1,
        'checkpoint_path': str(checkpoint_callback.best_model_path)
    }
    
    logger.info(f"\n{peft_method.upper()} Training Stats:")
    logger.info(f"  Training Time: {stats['training_time_minutes']:.2f} minutes")
    logger.info(f"  Peak GPU Memory: {stats['peak_gpu_gb']:.2f} GB")
    logger.info(f"  Best Val Loss: {stats['best_val_loss']:.4f}")
    logger.info(f"  Total Epochs: {stats['total_epochs']}")
    
    return stats


def compare_methods(config_path: str = None):
    """
    Compare LoRA and QLoRA methods using Lightning.
    
    Args:
        config_path: Path to config file
    """
    output_dir = OUTPUT_DIR / "comparison_lightning"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for method in ['lora', 'qlora']:
        try:
            stats = train_and_measure(method, config_path)
            results[method] = stats
        except Exception as e:
            logger.error(f"Failed to train with {method}: {e}")
            results[method] = None
    
    # Filter successful methods
    successful_methods = [m for m, r in results.items() if r is not None]
    
    if not successful_methods:
        logger.error("No methods trained successfully.")
        return results
        
    comparison_df = pd.DataFrame([results[m] for m in successful_methods])
    comparison_csv = output_dir / "comparison_results.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    logger.info(f"\nSaved comparison results to {comparison_csv}")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('LoRA vs QLoRA Comparison (PyTorch Lightning)', fontsize=14, fontweight='bold')
    
    training_times = [results[m]['training_time_minutes'] for m in successful_methods]
    peak_gpus = [results[m]['peak_gpu_gb'] for m in successful_methods]
    val_losses = [results[m]['best_val_loss'] for m in successful_methods]
    epochs = [results[m]['total_epochs'] for m in successful_methods]
    
    colors = ['#3498db', '#e74c3c'][:len(successful_methods)]
    
    axes[0, 0].bar(successful_methods, training_times, color=colors)
    axes[0, 0].set_title('Training Time')
    axes[0, 0].set_ylabel('Minutes')
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(training_times):
        axes[0, 0].text(i, v, f'{v:.1f}m', ha='center', va='bottom')
    
    axes[0, 1].bar(successful_methods, peak_gpus, color=colors)
    axes[0, 1].set_title('Peak GPU Memory')
    axes[0, 1].set_ylabel('GB')
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(peak_gpus):
        axes[0, 1].text(i, v, f'{v:.2f}GB', ha='center', va='bottom')
    
    axes[1, 0].bar(successful_methods, val_losses, color=colors)
    axes[1, 0].set_title('Best Validation Loss')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(val_losses):
        axes[1, 0].text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    axes[1, 1].bar(successful_methods, epochs, color=colors)
    axes[1, 1].set_title('Epochs Trained')
    axes[1, 1].set_ylabel('Epochs')
    axes[1, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(epochs):
        axes[1, 1].text(i, v, f'{v}', ha='center', va='bottom')
    
    plt.tight_layout()
    plot_path = output_dir / "comparison_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved comparison plots to {plot_path}")
    
    with open(output_dir / "comparison_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n" + "="*60)
    logger.info("COMPARISON SUMMARY (PyTorch Lightning)")
    logger.info("="*60)
    
    if 'lora' in successful_methods and 'qlora' in successful_methods:
        if results['lora']['training_time_minutes'] > 0:
            time_reduction = ((results['lora']['training_time_minutes'] - results['qlora']['training_time_minutes']) / results['lora']['training_time_minutes'] * 100)
            logger.info(f"LoRA Training Time:   {results['lora']['training_time_minutes']:.2f} min")
            logger.info(f"QLoRA Training Time:  {results['qlora']['training_time_minutes']:.2f} min")
            logger.info(f"Time Difference:      {time_reduction:+.1f}%")
        
        if results['lora']['peak_gpu_gb'] > 0:
            memory_reduction = ((results['lora']['peak_gpu_gb'] - results['qlora']['peak_gpu_gb']) / results['lora']['peak_gpu_gb'] * 100)
            logger.info(f"\nLoRA GPU Memory:      {results['lora']['peak_gpu_gb']:.2f} GB")
            logger.info(f"QLoRA GPU Memory:     {results['qlora']['peak_gpu_gb']:.2f} GB")
            logger.info(f"Memory Reduction:     {memory_reduction:.1f}%")
        
        logger.info(f"\nLoRA Best Val Loss:   {results['lora']['best_val_loss']:.4f}")
        logger.info(f"QLoRA Best Val Loss:  {results['qlora']['best_val_loss']:.4f}")
    else:
        for method in successful_methods:
            logger.info(f"{method.upper()} Training Time:   {results[method]['training_time_minutes']:.2f} min")
            logger.info(f"{method.upper()} GPU Memory:      {results[method]['peak_gpu_gb']:.2f} GB")
            logger.info(f"{method.upper()} Best Val Loss:   {results[method]['best_val_loss']:.4f}")
            
    logger.info("="*60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare LoRA vs QLoRA with Lightning")
    parser.add_argument('--config', type=str, help='Path to training config file')
    
    args = parser.parse_args()
    compare_methods(args.config)

