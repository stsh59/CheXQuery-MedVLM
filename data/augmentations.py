"""
Image augmentation utilities for chest X-ray images.

IMPORTANT: Medical image augmentations must preserve anatomical validity.
- NO horizontal flip: Heart is on LEFT side; flipping creates invalid anatomy
- Minimal rotation: Patients are positioned consistently
- Conservative intensity changes: Preserve diagnostic information
"""
import torch
from torchvision import transforms
from typing import Tuple, List, Optional


def get_train_transforms(
    image_size: int = 384,
    mean: List[float] = [0.5, 0.5, 0.5],
    std: List[float] = [0.5, 0.5, 0.5],
    rotation_degrees: float = 5.0,
    translate_percent: float = 0.03,
    scale_range: Tuple[float, float] = (0.97, 1.03),
    brightness_jitter: float = 0.05,
    contrast_jitter: float = 0.05,
) -> transforms.Compose:
    """
    Get training image transforms with MEDICALLY-VALID augmentation.
    
    Medical imaging constraints:
    - NO horizontal flip: Chest anatomy is asymmetric (heart on left)
    - Limited rotation: Real patient positioning variance is small
    - Conservative scaling: Preserve relative organ sizes
    - Mild intensity changes: Maintain diagnostic quality
    
    Args:
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        # NOTE: NO RandomHorizontalFlip - would put heart on wrong side!
        transforms.RandomRotation(degrees=rotation_degrees),  # Mild rotation (patient positioning)
        transforms.RandomAffine(
            degrees=0,
            translate=(translate_percent, translate_percent),  # Small translation (centering variance)
            scale=scale_range,        # Minimal scaling (distance variance)
        ),
        transforms.ColorJitter(
            brightness=brightness_jitter,  # Conservative brightness
            contrast=contrast_jitter,      # Conservative contrast
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_val_transforms(
    image_size: int = 384,
    mean: List[float] = [0.5, 0.5, 0.5],
    std: List[float] = [0.5, 0.5, 0.5],
) -> transforms.Compose:
    """
    Get validation/test image transforms (no augmentation).
    
    Args:
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_inference_transforms(
    image_size: int = 384,
    mean: List[float] = [0.5, 0.5, 0.5],
    std: List[float] = [0.5, 0.5, 0.5],
) -> transforms.Compose:
    """
    Get inference transforms (same as validation).
    
    Args:
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Composed transforms
    """
    return get_val_transforms(image_size, mean, std)
