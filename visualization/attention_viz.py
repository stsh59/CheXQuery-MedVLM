"""
Attention visualization for CheXQuery-MedVLM.

Provides interpretability by visualizing which image regions
each query attends to.
"""
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

logger = logging.getLogger(__name__)


def visualize_attention(
    image: torch.Tensor,
    attention_weights: torch.Tensor,
    query_names: List[str],
    output_path: str,
    image_size: int = 384,
    patch_size: int = 16,
    selected_queries: Optional[List[int]] = None,
    cmap: str = "jet",
    alpha: float = 0.5,
    title: Optional[str] = None,
) -> None:
    """
    Visualize attention weights as heatmaps over the image.
    
    Args:
        image: Input image tensor [3, H, W] or [H, W, 3]
        attention_weights: Attention weights [num_queries, num_patches]
        query_names: List of query names
        output_path: Path to save visualization
        image_size: Image size
        patch_size: Patch size
        selected_queries: Which queries to visualize (None = all)
        cmap: Colormap for heatmap
        alpha: Transparency of heatmap overlay
        title: Optional title for the figure
    """
    # Convert image to numpy
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:
            # [3, H, W] -> [H, W, 3]
            image = image.permute(1, 2, 0)
        image = image.cpu().numpy()
    
    # Denormalize image if needed
    if image.min() < 0:
        image = (image + 1) / 2  # Assuming [-1, 1] normalization
    image = np.clip(image, 0, 1)
    
    # Convert attention to numpy
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().numpy()
    
    # Determine which queries to visualize
    num_queries = attention_weights.shape[0]
    if selected_queries is None:
        selected_queries = list(range(min(num_queries, 20)))  # Max 20
    
    # Calculate grid size
    n_selected = len(selected_queries)
    n_cols = min(5, n_selected)
    n_rows = (n_selected + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Number of patches per side
    patches_per_side = image_size // patch_size
    
    for idx, query_idx in enumerate(selected_queries):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Get attention for this query
        attn = attention_weights[query_idx]  # [num_patches]
        
        # Reshape to 2D
        attn_2d = attn.reshape(patches_per_side, patches_per_side)
        
        # Resize to image size
        attn_resized = np.array(
            Image.fromarray(attn_2d).resize(
                (image_size, image_size),
                Image.BILINEAR
            )
        )
        
        # Normalize attention for visualization
        attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
        
        # Plot image
        ax.imshow(image)
        
        # Overlay attention heatmap
        ax.imshow(attn_norm, cmap=cmap, alpha=alpha)
        
        # Title
        query_name = query_names[query_idx] if query_idx < len(query_names) else f"Query {query_idx}"
        ax.set_title(query_name, fontsize=10)
        ax.axis('off')
    
    # Hide unused axes
    for idx in range(len(selected_queries), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    # Add main title
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved attention visualization to {output_path}")


def visualize_condition_attention(
    image: torch.Tensor,
    attention_weights: torch.Tensor,
    condition_names: List[str],
    output_path: str,
    top_k: int = 5,
    **kwargs,
) -> None:
    """
    Visualize attention for top-k condition queries.
    
    Args:
        image: Input image
        attention_weights: Attention weights [num_queries, num_patches]
        condition_names: List of condition names
        output_path: Output path
        top_k: Number of top conditions to visualize
        **kwargs: Additional arguments for visualize_attention
    """
    # Get attention strength per query
    if isinstance(attention_weights, torch.Tensor):
        attn_np = attention_weights.cpu().numpy()
    else:
        attn_np = attention_weights
    
    # Compute attention strength (max attention per query)
    attn_strength = attn_np.max(axis=1)
    
    # Get top-k queries
    top_indices = np.argsort(attn_strength)[-top_k:][::-1]
    
    visualize_attention(
        image=image,
        attention_weights=attention_weights,
        query_names=condition_names,
        output_path=output_path,
        selected_queries=list(top_indices),
        title="Top Condition Query Attention",
        **kwargs,
    )


def create_anatomical_attention_map(
    image: torch.Tensor,
    attention_weights: torch.Tensor,
    region_names: List[str],
    output_path: str,
    **kwargs,
) -> None:
    """
    Create anatomical region attention visualization.
    
    Shows attention for each anatomical region in separate panels.
    
    Args:
        image: Input image
        attention_weights: Attention for anatomical queries [6, num_patches]
        region_names: Anatomical region names
        output_path: Output path
        **kwargs: Additional arguments
    """
    visualize_attention(
        image=image,
        attention_weights=attention_weights,
        query_names=region_names,
        output_path=output_path,
        title="Anatomical Region Attention",
        **kwargs,
    )


def save_all_visualizations(
    image: torch.Tensor,
    condition_attention: torch.Tensor,
    anatomical_attention: torch.Tensor,
    condition_names: List[str],
    region_names: List[str],
    output_dir: str,
    sample_id: str,
) -> None:
    """
    Save all attention visualizations for a sample.
    
    Args:
        image: Input image
        condition_attention: Attention for condition queries
        anatomical_attention: Attention for anatomical queries
        condition_names: Condition names
        region_names: Region names
        output_dir: Output directory
        sample_id: Sample identifier
    """
    output_dir = Path(output_dir) / sample_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Condition attention (top 5)
    visualize_condition_attention(
        image=image,
        attention_weights=condition_attention,
        condition_names=condition_names,
        output_path=str(output_dir / "condition_attention.png"),
        top_k=5,
    )
    
    # Anatomical attention
    create_anatomical_attention_map(
        image=image,
        attention_weights=anatomical_attention,
        region_names=region_names,
        output_path=str(output_dir / "anatomical_attention.png"),
    )
    
    # Combined (all queries)
    all_attention = torch.cat([condition_attention, anatomical_attention], dim=0)
    all_names = condition_names + region_names
    
    visualize_attention(
        image=image,
        attention_weights=all_attention,
        query_names=all_names,
        output_path=str(output_dir / "all_attention.png"),
        title="All Query Attention",
    )
    
    logger.info(f"Saved all visualizations to {output_dir}")
