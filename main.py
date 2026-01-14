"""
Main CLI for running different stages of the Medical-SigLIP pipeline.
Built with PyTorch Lightning.
"""
import argparse
import sys
from pathlib import Path

from data.split import create_data_splits
from utils.logger import setup_logger

logger = setup_logger(__name__)


def prepare_data(args):
    """Prepare and split the data."""
    logger.info("Preparing data splits...")
    splits = create_data_splits()
    logger.info(f"Data preparation completed:")
    logger.info(f"  Train: {len(splits['train'])} patients")
    logger.info(f"  Val: {len(splits['val'])} patients")
    logger.info(f"  Test: {len(splits['test'])} patients")


def train_siglip(args):
    """Train SigLIP with LoRA or QLoRA using Lightning."""
    from train.train_siglip_lightning import train_siglip as train_fn
    
    logger.info(f"Training SigLIP with {args.peft_method} using PyTorch Lightning...")
    
    class TrainArgs:
        def __init__(self):
            self.config = args.config
            self.peft_method = args.peft_method
            self.batch_size = args.batch_size
            self.num_epochs = args.num_epochs
    
    train_args = TrainArgs()
    train_fn(train_args)


def train_full(args):
    """Train full pipeline: SigLIP + projection + BioGPT using Lightning."""
    from train.train_full_lightning import train_full_pipeline
    
    logger.info("Training full pipeline with PyTorch Lightning...")
    
    class TrainArgs:
        def __init__(self):
            self.config = args.config
            self.siglip_checkpoint = args.siglip_checkpoint
            self.freeze_siglip = args.freeze_siglip
            self.num_epochs = args.num_epochs
    
    train_args = TrainArgs()
    train_full_pipeline(train_args)


def evaluate(args):
    """Evaluate a trained model."""
    from eval.evaluate_lightning import evaluate_model
    
    logger.info(f"Evaluating model from {args.checkpoint}...")
    
    evaluate_model(
        checkpoint_path=args.checkpoint,
        split=args.split,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        limit_batches=args.limit_batches
    )


def qualitative(args):
    """Run qualitative analysis."""
    from eval.qualitative_analysis_lightning import visualize_samples
    
    logger.info("Running qualitative analysis with PyTorch Lightning...")
    
    visualize_samples(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples
    )


def compare(args):
    """Compare LoRA vs QLoRA."""
    from experiments.compare_lora_qlora_lightning import compare_methods
    
    logger.info("Comparing LoRA vs QLoRA with PyTorch Lightning...")
    
    compare_methods(config_path=args.config)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Medical-SigLIP: Fine-tuning with PEFT using PyTorch Lightning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare data
  python main.py prepare_data
  
  # Train SigLIP with LoRA
  python main.py train_siglip --peft_method lora
  
  # Train SigLIP with QLoRA
  python main.py train_siglip --peft_method qlora --num_epochs 10
  
  # Train full pipeline
  python main.py train_full --siglip_checkpoint checkpoints/siglip_lora/last.ckpt
  
  # Evaluate model
  python main.py evaluate --checkpoint checkpoints/full_pipeline/last.ckpt
  
  # Qualitative analysis
  python main.py qualitative --checkpoint checkpoints/full_pipeline/last.ckpt --num_samples 10
  
  # Compare LoRA vs QLoRA
  python main.py compare
  
Note: All training uses PyTorch Lightning Trainer
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    prepare_parser = subparsers.add_parser('prepare_data', help='Prepare and split the dataset')
    
    train_siglip_parser = subparsers.add_parser('train_siglip', help='Train SigLIP encoder with Lightning')
    train_siglip_parser.add_argument('--peft_method', type=str, default='lora', choices=['lora', 'qlora'], help='PEFT method to use')
    train_siglip_parser.add_argument('--config', type=str, help='Path to config file')
    train_siglip_parser.add_argument('--batch_size', type=int, help='Batch size')
    train_siglip_parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    
    train_full_parser = subparsers.add_parser('train_full', help='Train full pipeline with Lightning')
    train_full_parser.add_argument('--siglip_checkpoint', type=str, help='Path to pretrained SigLIP checkpoint')
    train_full_parser.add_argument('--config', type=str, help='Path to config file')
    train_full_parser.add_argument('--freeze_siglip', action='store_true', default=True, help='Freeze SigLIP encoder')
    train_full_parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    eval_parser.add_argument('--split', type=str, default='test', choices=['val', 'test'], help='Data split to evaluate')
    eval_parser.add_argument('--output_dir', type=str, help='Output directory for results')
    eval_parser.add_argument('--limit_batches', type=int, help='Limit number of batches for testing')
    
    qual_parser = subparsers.add_parser('qualitative', help='Run qualitative analysis')
    qual_parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    qual_parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    
    compare_parser = subparsers.add_parser('compare', help='Compare LoRA vs QLoRA')
    compare_parser.add_argument('--config', type=str, help='Path to config file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    logger.info("="*60)
    logger.info(f"Medical-SigLIP Pipeline (PyTorch Lightning) - Command: {args.command}")
    logger.info("="*60)
    
    if args.command == 'prepare_data':
        prepare_data(args)
    elif args.command == 'train_siglip':
        train_siglip(args)
    elif args.command == 'train_full':
        train_full(args)
    elif args.command == 'evaluate':
        evaluate(args)
    elif args.command == 'qualitative':
        qualitative(args)
    elif args.command == 'compare':
        compare(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)
    
    logger.info("="*60)
    logger.info("Completed successfully!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
