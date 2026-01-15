"""
CheXQuery-MedVLM: Main Entry Point

A novel vision-language model for chest X-ray report generation
featuring CheXbert-initialized condition queries and anatomical region queries.
"""
import argparse
import logging
import sys
from pathlib import Path

import yaml
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_data(args):
    """Prepare data splits and download dataset."""
    logger.info("Preparing data...")
    
    from data.datamodule import ChestXrayDataModule
    data_config = load_config(args.data_config)
    
    datamodule = ChestXrayDataModule(
        batch_size=args.batch_size,
        splits_file=args.splits_file,
        text_output_template=data_config.get("text", {}).get("output_template"),
        text_max_length=data_config.get("text", {}).get("max_length", 512),
        image_mean=data_config.get("image", {}).get("mean"),
        image_std=data_config.get("image", {}).get("std"),
        augmentation_config=data_config.get("augmentation", {}),
        use_siglip_processor=data_config.get("image", {}).get("use_siglip_processor", False),
        processor_model=data_config.get("image", {}).get("processor_model"),
        sampling_config=data_config.get("sampling", {}),
    )
    
    # This will download data and create splits
    datamodule.prepare_data()
    datamodule.setup()
    
    logger.info(f"Train samples: {len(datamodule.train_dataset)}")
    logger.info(f"Val samples: {len(datamodule.val_dataset)}")
    logger.info(f"Test samples: {len(datamodule.test_dataset)}")
    logger.info("Data preparation complete!")


def train(args):
    """Train the model."""
    logger.info("Starting training...")
    
    from training.trainer import train_model, train_all_phases
    
    if args.all_phases:
        train_all_phases(
            model_config_path=args.model_config,
            train_config_path=args.train_config,
            data_config_path=args.data_config,
        )
    else:
        train_model(
            model_config_path=args.model_config,
            train_config_path=args.train_config,
            data_config_path=args.data_config,
            phase=args.phase,
            checkpoint_path=args.checkpoint_dir,
            resume_from=args.resume,
        )


def evaluate(args):
    """Evaluate the model."""
    logger.info("Starting evaluation...")
    
    from evaluation.evaluate import evaluate_from_configs
    
    metrics = evaluate_from_configs(
        checkpoint_path=args.checkpoint,
        model_config_path=args.model_config,
        train_config_path=args.train_config,
        data_config_path=args.data_config,
        eval_config_path=args.eval_config,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_beams=args.num_beams,
        split=args.split,
    )
    
    logger.info("Evaluation complete!")
    return metrics


def generate(args):
    """Generate reports for images."""
    logger.info("Generating reports...")
    
    from PIL import Image
    
    from data.augmentations import get_inference_transforms
    from data.preprocessing import postprocess_generated_report
    from training.lightning_module import CheXQueryLightningModule
    
    # Load configs
    model_config = load_config(args.model_config)
    train_config = load_config(args.train_config)
    data_config = load_config(args.data_config)
    eval_config = load_config(args.eval_config)
    generation_config = eval_config.get("generation", {})
    postprocess_config = eval_config.get("postprocess", {})
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = CheXQueryLightningModule.load_from_checkpoint(
        args.checkpoint,
        model_config=model_config,
        training_config=train_config,
        phase=2,
        prompt_template=data_config.get("text", {}).get("prompt_template"),
    )
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Image transform
    transform = get_inference_transforms(
        image_size=data_config.get("image", {}).get("size", 384),
        mean=data_config.get("image", {}).get("mean", [0.5, 0.5, 0.5]),
        std=data_config.get("image", {}).get("std", [0.5, 0.5, 0.5]),
    )
    
    # Generate for each image
    image_paths = args.images
    if len(image_paths) == 1 and Path(image_paths[0]).is_dir():
        # Directory provided
        image_dir = Path(image_paths[0])
        image_paths = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
    
    for image_path in image_paths:
        logger.info(f"Processing: {image_path}")
        
        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Generate
        with torch.no_grad():
            report = model.generate(
                images=image_tensor,
                max_length=generation_config.get("max_length", args.max_length),
                min_length=generation_config.get("min_length", 20),
                num_beams=generation_config.get("num_beams", args.num_beams),
                length_penalty=generation_config.get("length_penalty", 1.0),
                no_repeat_ngram_size=generation_config.get("no_repeat_ngram_size", 3),
                early_stopping=generation_config.get("early_stopping", True),
                do_sample=generation_config.get("do_sample", False),
                temperature=generation_config.get("temperature", 1.0),
                top_p=generation_config.get("top_p", 1.0),
                top_k=generation_config.get("top_k", 50),
            )[0]
        report = postprocess_generated_report(
            report,
            prompt_template=data_config.get("text", {}).get("prompt_template", ""),
            apply_prompt_strip=postprocess_config.get("strip_prompt", True),
            apply_impression_consistency=postprocess_config.get("impression_consistency", False),
        )
        
        print(f"\n{'='*60}")
        print(f"Image: {image_path}")
        print(f"{'='*60}")
        print(f"Report: {report}")
        print(f"{'='*60}\n")


def visualize(args):
    """Generate attention visualizations."""
    logger.info("Generating visualizations...")
    
    from PIL import Image
    
    from data.augmentations import get_inference_transforms
    from training.lightning_module import CheXQueryLightningModule
    from visualization.attention_viz import save_all_visualizations
    
    # Load configs
    model_config = load_config(args.model_config)
    train_config = load_config(args.train_config)
    data_config = load_config(args.data_config)
    eval_config = load_config(args.eval_config)
    generation_config = eval_config.get("generation", {})
    
    # Load model
    model = CheXQueryLightningModule.load_from_checkpoint(
        args.checkpoint,
        model_config=model_config,
        training_config=train_config,
        phase=2,
        prompt_template=data_config.get("text", {}).get("prompt_template"),
    )
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Image transform
    transform = get_inference_transforms(
        image_size=data_config.get("image", {}).get("size", 384),
        mean=data_config.get("image", {}).get("mean", [0.5, 0.5, 0.5]),
        std=data_config.get("image", {}).get("std", [0.5, 0.5, 0.5]),
    )
    
    # Process images
    for image_path in args.images:
        logger.info(f"Processing: {image_path}")
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Generate with attention
        with torch.no_grad():
            generated, attn_weights, gate_values = model.model.generate_with_attention(
                pixel_values=image_tensor,
                max_length=generation_config.get("max_length", args.max_length),
                num_beams=generation_config.get("num_beams", args.num_beams),
            )
        
        # Get query names
        condition_names = model.model.get_condition_names()
        region_names = model.model.get_region_names()
        
        # Split attention weights
        num_condition = len(condition_names)
        condition_attn = attn_weights[0, :num_condition, :]
        anatomical_attn = attn_weights[0, num_condition:, :]
        
        # Save visualizations
        sample_id = Path(image_path).stem
        save_all_visualizations(
            image=image_tensor[0],
            condition_attention=condition_attn,
            anatomical_attention=anatomical_attn,
            condition_names=condition_names,
            region_names=region_names,
            output_dir=args.output_dir,
            sample_id=sample_id,
        )
        
        logger.info(f"Generated report: {generated[0]}")


def compute_chexbert_labels(args):
    """Compute and save CheXbert labels for all reports."""
    logger.info("Computing CheXbert labels...")
    from data.chexbert_labels import compute_chexbert_labels_from_config
    data_config = load_config(args.data_config)
    compute_chexbert_labels_from_config(
        data_config=data_config,
        output_path=args.output_path,
        batch_size=args.batch_size,
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CheXQuery-MedVLM: Chest X-ray Report Generation"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--model-config",
        default="configs/model_config.yaml",
        help="Path to model config"
    )
    common_parser.add_argument(
        "--train-config",
        default="configs/train_config.yaml",
        help="Path to training config"
    )
    common_parser.add_argument(
        "--data-config",
        default="configs/data_config.yaml",
        help="Path to data config"
    )
    common_parser.add_argument(
        "--eval-config",
        default="configs/eval_config.yaml",
        help="Path to eval config"
    )
    
    # Prepare data
    prepare_parser = subparsers.add_parser(
        "prepare",
        parents=[common_parser],
        help="Prepare data splits"
    )
    prepare_parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size"
    )
    prepare_parser.add_argument(
        "--splits-file",
        default="outputs/splits/data_splits.json",
        help="Path to save splits"
    )
    
    # Train
    train_parser = subparsers.add_parser(
        "train",
        parents=[common_parser],
        help="Train the model"
    )
    train_parser.add_argument(
        "--phase",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Training phase"
    )
    train_parser.add_argument(
        "--all-phases",
        action="store_true",
        help="Run all training phases"
    )
    train_parser.add_argument(
        "--checkpoint-dir",
        default="outputs/checkpoints",
        help="Checkpoint directory"
    )
    train_parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    # Evaluate
    eval_parser = subparsers.add_parser(
        "evaluate",
        parents=[common_parser],
        help="Evaluate the model"
    )
    eval_parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint"
    )
    eval_parser.add_argument(
        "--output-dir",
        default="outputs/evaluation",
        help="Output directory"
    )
    eval_parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size"
    )
    eval_parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max generation length"
    )
    eval_parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Number of beams"
    )
    eval_parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split"
    )
    
    # Generate
    gen_parser = subparsers.add_parser(
        "generate",
        parents=[common_parser],
        help="Generate reports"
    )
    gen_parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint"
    )
    gen_parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Image paths or directory"
    )
    gen_parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max generation length"
    )
    gen_parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Number of beams"
    )
    
    # Visualize
    viz_parser = subparsers.add_parser(
        "visualize",
        parents=[common_parser],
        help="Generate attention visualizations"
    )
    viz_parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint"
    )
    viz_parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Image paths"
    )
    viz_parser.add_argument(
        "--output-dir",
        default="outputs/visualizations",
        help="Output directory"
    )
    viz_parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max generation length"
    )
    viz_parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Number of beams"
    )

    # CheXbert labels
    chex_parser = subparsers.add_parser(
        "chexbert_labels",
        parents=[common_parser],
        help="Compute CheXbert labels for all reports"
    )
    chex_parser.add_argument(
        "--output-path",
        default="outputs/chexbert_labels.json",
        help="Path to save CheXbert labels"
    )
    chex_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for CheXbert labeling"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Execute command
    logger.info("=" * 60)
    logger.info("CheXQuery-MedVLM")
    logger.info("=" * 60)
    
    if args.command == "prepare":
        prepare_data(args)
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "generate":
        generate(args)
    elif args.command == "visualize":
        visualize(args)
    elif args.command == "chexbert_labels":
        compute_chexbert_labels(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
