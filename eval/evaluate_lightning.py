"""
Evaluate models using PyTorch Lightning.
"""
import torch
import pytorch_lightning as pl
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json

from data.datamodule import IUXrayGenerationDataModule
from models.full_pipeline_lightning import FullPipelineLightning
from eval.metrics import MedicalReportMetrics, format_metrics_report
from utils.config import OUTPUT_DIR
from utils.logger import setup_logger

logger = setup_logger(__name__)


@torch.no_grad()
def generate_reports(model, datamodule, split='test', limit_batches=None):
    """
    Generate reports using Lightning model.
    
    Args:
        model: Trained Lightning model
        datamodule: Lightning DataModule
        split: Data split to use
        limit_batches: Optional limit on number of batches
    
    Returns:
        List of results
    """
    model.eval()
    
    if split == 'test':
        dataloader = datamodule.test_dataloader()
    elif split == 'val':
        dataloader = datamodule.val_dataloader()
    else:
        raise ValueError(f"Invalid split: {split}")
    
    results = []
    
    for i, (images, texts, metadata) in enumerate(tqdm(dataloader, desc="Generating reports")):
        if limit_batches and i >= limit_batches:
            break
            
        images = images.to(model.device)
        
        generated_texts = model.generate(images, max_length=256, num_beams=4)
        
        for i in range(len(generated_texts)):
            results.append({
                'uid': metadata[i]['uid'],
                'filename': metadata[i]['filename'],
                'projection': metadata[i]['projection'],
                'generated': generated_texts[i],
                'reference': texts[i]
            })
    
    return results


def evaluate_model(checkpoint_path: str, split: str = 'test', output_dir: Path = None, limit_batches: int = None):
    """
    Evaluate a Lightning model checkpoint.
    
    Args:
        checkpoint_path: Path to Lightning checkpoint
        split: Data split to evaluate on
        output_dir: Directory to save results
        limit_batches: Optional limit on number of batches
    """
    checkpoint_path = Path(checkpoint_path)
    logger.info(f"Loading Lightning checkpoint from {checkpoint_path}")
    
    model = FullPipelineLightning.load_from_checkpoint(checkpoint_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    datamodule = IUXrayGenerationDataModule(
        batch_size=8,
        num_workers=4
    )
    datamodule.setup(stage='test' if split == 'test' else 'fit')
    
    logger.info(f"Generating reports on {split} set...")
    results = generate_reports(model, datamodule, split=split, limit_batches=limit_batches)
    
    if output_dir is None:
        output_dir = OUTPUT_DIR / "evaluation" / checkpoint_path.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame(results)
    results_csv = output_dir / f"generated_reports_{split}.csv"
    results_df.to_csv(results_csv, index=False)
    logger.info(f"Saved generated reports to {results_csv}")
    
    logger.info("Computing evaluation metrics...")
    metric_calculator = MedicalReportMetrics()
    
    references = [r['reference'] for r in results]
    generated = [r['generated'] for r in results]
    
    metrics = metric_calculator.compute_all_metrics(references, generated)
    
    logger.info("\n" + format_metrics_report(metrics))
    
    metrics_file = output_dir / f"metrics_{split}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_file}")
    
    return metrics, results

