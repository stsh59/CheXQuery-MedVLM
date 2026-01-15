"""
Model evaluation script for CheXQuery-MedVLM.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import pandas as pd
from tqdm import tqdm

from data.datamodule import ChestXrayDataModule
from data.preprocessing import postprocess_generated_report
from training.lightning_module import CheXQueryLightningModule
from evaluation.metrics import MedicalReportMetrics

logger = logging.getLogger(__name__)


def evaluate_model(
    checkpoint_path: str,
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    data_config: Dict[str, Any],
    eval_config: Optional[Dict[str, Any]] = None,
    output_dir: str = "outputs/evaluation",
    batch_size: int = 8,
    max_length: int = 512,
    num_beams: int = 4,
    split: str = "test",
    save_predictions: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a trained model on the test set.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_config: Model configuration
        training_config: Training configuration
        data_config: Data configuration
        output_dir: Output directory for results
        batch_size: Evaluation batch size
        max_length: Maximum generation length
        num_beams: Number of beams for generation
        split: Dataset split to evaluate on
        save_predictions: Whether to save predictions to file
        
    Returns:
        Dictionary of evaluation metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generation_config = (eval_config or {}).get("generation", {})
    normalization_config = (eval_config or {}).get("normalization", {})
    postprocess_config = (eval_config or {}).get("postprocess", {})
    
    # Load model
    logger.info(f"Loading model from {checkpoint_path}")
    prompt_template = data_config.get("text", {}).get("prompt_template")
    model = CheXQueryLightningModule.load_from_checkpoint(
        checkpoint_path,
        model_config=model_config,
        training_config=training_config,
        phase=2,  # Use phase 2 config for evaluation
        prompt_template=prompt_template,
    )
    model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Data
    datamodule = ChestXrayDataModule(
        batch_size=batch_size,
        num_workers=4,
        image_size=data_config.get("image", {}).get("size", 384),
        projection_type=data_config.get("filtering", {}).get("projection_type", "Frontal"),
        splits_file=data_config.get("splits", {}).get("split_file"),
        text_output_template=data_config.get("text", {}).get("output_template"),
        text_max_length=data_config.get("text", {}).get("max_length", 512),
        image_mean=data_config.get("image", {}).get("mean"),
        image_std=data_config.get("image", {}).get("std"),
        use_siglip_processor=data_config.get("image", {}).get("use_siglip_processor", False),
        processor_model=data_config.get("image", {}).get("processor_model"),
        sampling_config=data_config.get("sampling", {}),
    )
    datamodule.setup()
    
    if split == "test":
        dataloader = datamodule.test_dataloader()
    elif split == "val":
        dataloader = datamodule.val_dataloader()
    else:
        dataloader = datamodule.train_dataloader()
    
    # Generate predictions
    logger.info(f"Generating predictions on {split} set...")
    all_predictions = []
    all_references = []
    all_metadata = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["images"].to(device)
            texts = batch["texts"]
            metadata = batch["metadata"]
            
            # Generate
            predictions = model.generate(
                images=images,
                max_length=generation_config.get("max_length", max_length),
                min_length=generation_config.get("min_length", 20),
                num_beams=generation_config.get("num_beams", num_beams),
                length_penalty=generation_config.get("length_penalty", 1.0),
                no_repeat_ngram_size=generation_config.get("no_repeat_ngram_size", 3),
                early_stopping=generation_config.get("early_stopping", True),
                do_sample=generation_config.get("do_sample", False),
                temperature=generation_config.get("temperature", 1.0),
                top_p=generation_config.get("top_p", 1.0),
                top_k=generation_config.get("top_k", 50),
            )
            # Post-process predictions (prompt strip, impression consistency)
            prompt_template = data_config.get("text", {}).get("prompt_template", "")
            processed = [
                postprocess_generated_report(
                    p,
                    prompt_template=prompt_template,
                    apply_prompt_strip=postprocess_config.get("strip_prompt", True),
                    apply_impression_consistency=postprocess_config.get("impression_consistency", False),
                )
                for p in predictions
            ]
            all_predictions.extend(processed)
            all_references.extend(texts)
            all_metadata.extend(metadata)
    
    # Compute metrics
    logger.info("Computing metrics...")
    label_names = data_config.get("chexbert", {}).get("labels")
    metrics_calculator = MedicalReportMetrics(
        normalization_config=normalization_config,
        label_names=label_names,
    )
    metrics = metrics_calculator.compute_all_metrics(
        references=all_references,
        hypotheses=all_predictions,
    )
    
    # Log metrics
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    for metric_name, value in sorted(metrics.items()):
        logger.info(f"{metric_name}: {value:.4f}")
    logger.info("=" * 60)
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Save predictions
    if save_predictions:
        predictions_df = pd.DataFrame({
            "uid": [m["uid"] for m in all_metadata],
            "filename": [m["filename"] for m in all_metadata],
            "reference": all_references,
            "prediction": all_predictions,
        })
        predictions_path = output_dir / "predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"Saved predictions to {predictions_path}")
    
    return metrics


def evaluate_from_configs(
    checkpoint_path: str,
    model_config_path: str = "configs/model_config.yaml",
    train_config_path: str = "configs/train_config.yaml",
    data_config_path: str = "configs/data_config.yaml",
    eval_config_path: str = "configs/eval_config.yaml",
    output_dir: str = "outputs/evaluation",
    **kwargs,
) -> Dict[str, float]:
    """
    Evaluate model using config file paths.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_config_path: Path to model config
        train_config_path: Path to training config
        data_config_path: Path to data config
        output_dir: Output directory
        **kwargs: Additional arguments for evaluate_model
        
    Returns:
        Dictionary of evaluation metrics
    """
    import yaml
    
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    with open(train_config_path, 'r') as f:
        training_config = yaml.safe_load(f)
    
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    with open(eval_config_path, 'r') as f:
        eval_config = yaml.safe_load(f)
    
    return evaluate_model(
        checkpoint_path=checkpoint_path,
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        eval_config=eval_config,
        output_dir=output_dir,
        **kwargs,
    )
