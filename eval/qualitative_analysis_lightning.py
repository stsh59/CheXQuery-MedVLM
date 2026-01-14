"""
Qualitative analysis using PyTorch Lightning models.
"""
import torch
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import random

from data.datamodule import IUXrayGenerationDataModule
from models.full_pipeline_lightning import FullPipelineLightning
from utils.config import OUTPUT_DIR, IMAGES_DIR
from utils.logger import setup_logger

logger = setup_logger(__name__)


@torch.no_grad()
def visualize_samples(checkpoint_path: str, num_samples: int = 5):
    """
    Visualize predictions using Lightning model.
    
    Args:
        checkpoint_path: Path to Lightning checkpoint
        num_samples: Number of samples to visualize
    """
    checkpoint_path = Path(checkpoint_path)
    logger.info(f"Loading Lightning checkpoint from {checkpoint_path}")
    
    model = FullPipelineLightning.load_from_checkpoint(checkpoint_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    datamodule = IUXrayGenerationDataModule(batch_size=1, num_workers=0)
    datamodule.setup(stage='test')
    
    dataset = datamodule.test_dataset
    
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for idx, sample_idx in enumerate(indices):
        image_tensor, reference_text, metadata = dataset[sample_idx]
        
        image_path = IMAGES_DIR / metadata['filename']
        original_image = Image.open(image_path).convert('RGB')
        
        image_tensor = image_tensor.unsqueeze(0).to(device)
        generated_text = model.generate(image_tensor, max_length=256, num_beams=4)[0]
        
        ax = axes[idx]
        ax.axis('off')
        
        ax.text(
            0, 0.9,
            f"UID: {metadata['uid']} | Projection: {metadata['projection']}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontweight='bold'
        )
        
        ax.text(
            0, 0.75,
            f"Generated: {generated_text}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            color='blue',
            wrap=True
        )
        
        ax.text(
            0, 0.55,
            f"Reference: {reference_text}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            color='green',
            wrap=True
        )
        
        img_ax = fig.add_axes([0.05, idx/num_samples + 0.02, 0.2, 1/num_samples - 0.04])
        img_ax.imshow(original_image, cmap='gray')
        img_ax.axis('off')
    
    plt.tight_layout()
    
    output_dir = OUTPUT_DIR / "qualitative" / checkpoint_path.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "qualitative_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization to {save_path}")
    
    plt.show()

