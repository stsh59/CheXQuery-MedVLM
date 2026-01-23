"""
PyTorch Dataset for Chest X-ray Report Generation.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from data.preprocessing import TextPreprocessor
from data.augmentations import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)


class ChestXrayDataset(Dataset):
    """
    Dataset for chest X-ray report generation.
    
    Loads image-report pairs and provides CheXbert labels for auxiliary training.
    
    Args:
        data_root: Root directory containing images and CSVs
        uids: List of patient UIDs to include
        split: Dataset split ('train', 'val', 'test')
        image_size: Target image size
        transform: Optional custom transform
        projection_type: Filter by projection type ('Frontal', 'Lateral', or None)
        chexbert_labels: Optional pre-computed CheXbert labels dict
    """
    
    def __init__(
        self,
        data_root: str,
        uids: List[int],
        split: str = "train",
        image_size: int = 384,
        transform: Optional[transforms.Compose] = None,
        projection_type: Optional[str] = "Frontal",
        projection_types: Optional[List[str]] = None,
        require_both_views: bool = False,
        chexbert_labels: Optional[Dict[str, List[int]]] = None,
        text_output_template: Optional[str] = None,
        text_max_length: int = 512,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        augmentation_config: Optional[Dict[str, float]] = None,
    ):
        self.data_root = Path(data_root)
        self.uids = set(uids)
        self.split = split
        self.image_size = image_size
        self.projection_type = projection_type
        self.projection_types = projection_types
        self.require_both_views = require_both_views
        if self.require_both_views and (not self.projection_types or len(self.projection_types) < 2):
            raise ValueError("require_both_views requires at least two projection_types.")
        self.chexbert_labels = chexbert_labels or {}
        
        # Text preprocessor
        self.text_preprocessor = TextPreprocessor(
            output_template=text_output_template or "Findings: {findings} | Impression: {impression}",
            max_length=text_max_length,
        )
        
        # Set up transforms
        if transform is not None:
            self.transform = transform
        elif split == "train":
            aug = augmentation_config or {}
            self.transform = get_train_transforms(
                image_size=image_size,
                mean=image_mean or [0.5, 0.5, 0.5],
                std=image_std or [0.5, 0.5, 0.5],
                rotation_degrees=aug.get("rotation_degrees", 5.0),
                translate_percent=aug.get("translate_percent", 0.03),
                scale_range=tuple(aug.get("scale_range", [0.97, 1.03])),
                brightness_jitter=aug.get("brightness_jitter", 0.05),
                contrast_jitter=aug.get("contrast_jitter", 0.05),
            )
        else:
            self.transform = get_val_transforms(
                image_size=image_size,
                mean=image_mean or [0.5, 0.5, 0.5],
                std=image_std or [0.5, 0.5, 0.5],
            )
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load and merge reports with projections data."""
        logger.info(f"Loading {self.split} dataset...")
        
        # Load CSVs
        reports_path = self.data_root / "indiana_reports.csv"
        projections_path = self.data_root / "indiana_projections.csv"
        
        reports_df = pd.read_csv(reports_path)
        projections_df = pd.read_csv(projections_path)
        
        # Filter by UIDs
        reports_df = reports_df[reports_df['uid'].isin(self.uids)]
        
        # Merge
        merged_df = projections_df.merge(reports_df, on='uid', how='inner')
        
        if self.require_both_views and self.projection_types:
            self.samples = self._build_paired_samples(merged_df)
        else:
            # Filter by projection type (single-view)
            if self.projection_type:
                merged_df = merged_df[merged_df['projection'] == self.projection_type]
            self.samples = self._build_single_view_samples(merged_df)

        logger.info(f"Loaded {len(self.samples)} {self.split} samples")

    def _is_valid_image(self, image_path: Path) -> bool:
        if not image_path.exists():
            return False
        try:
            img = Image.open(image_path)
            img.verify()
        except Exception:
            return False
        return True

    def _build_single_view_samples(self, merged_df: pd.DataFrame) -> List[Dict[str, str]]:
        """Build single-view samples."""
        valid_samples = []
        images_dir = self.data_root / "images" / "images_normalized"

        for _, row in merged_df.iterrows():
            image_path = images_dir / row["filename"]
            if not self._is_valid_image(image_path):
                continue

            findings = row.get("findings", "")
            impression = row.get("impression", "")
            if pd.isna(impression) or str(impression).strip() == "":
                continue

            valid_samples.append(
                {
                    "uid": row["uid"],
                    "filename": row["filename"],
                    "image_path": str(image_path),
                    "projection": row["projection"],
                    "findings": str(findings) if pd.notna(findings) else "",
                    "impression": str(impression),
                }
            )
        return valid_samples

    def _build_paired_samples(self, merged_df: pd.DataFrame) -> List[Dict[str, str]]:
        """Build paired samples with required projections."""
        images_dir = self.data_root / "images" / "images_normalized"
        projection_types = [p for p in self.projection_types if p]
        required = set(projection_types)
        paired_samples = []
        dropped_missing = 0

        for uid, group in merged_df.groupby("uid"):
            group = group[group["projection"].isin(required)]
            if group.empty:
                dropped_missing += 1
                continue

            findings = group["findings"].iloc[0] if "findings" in group else ""
            impression = group["impression"].iloc[0] if "impression" in group else ""
            if pd.isna(impression) or str(impression).strip() == "":
                continue

            view_paths = {}
            view_filenames = {}
            for projection in projection_types:
                proj_rows = group[group["projection"] == projection]
                image_path = None
                filename = None
                for _, row in proj_rows.iterrows():
                    candidate = images_dir / row["filename"]
                    if self._is_valid_image(candidate):
                        image_path = str(candidate)
                        filename = row["filename"]
                        break
                if image_path is None:
                    break
                view_paths[projection] = image_path
                view_filenames[projection] = filename

            if set(view_paths.keys()) != required:
                dropped_missing += 1
                continue

            paired_samples.append(
                {
                    "uid": uid,
                    "view_paths": view_paths,
                    "view_filenames": view_filenames,
                    "findings": str(findings) if pd.notna(findings) else "",
                    "impression": str(impression),
                }
            )

        logger.info(
            f"Paired samples built: {len(paired_samples)}; dropped (missing views): {dropped_missing}"
        )
        return paired_samples
    
    def __len__(self) -> int:
        return len(self.samples)

    def get_sampling_weights(self, target_abnormal_ratio: float = 0.5) -> Optional[List[float]]:
        """
        Compute sampling weights to oversample abnormal cases.
        """
        if not self.samples:
            return None
        abnormal_flags = []
        for sample in self.samples:
            uid_key = str(sample["uid"])
            labels = self.chexbert_labels.get(uid_key)
            if labels is None:
                abnormal_flags.append(False)
                continue
            # Treat any positive label except "No Finding" as abnormal
            abnormal_flags.append(any(labels[1:]))
        total = len(abnormal_flags)
        abnormal_count = sum(abnormal_flags)
        if abnormal_count == 0 or abnormal_count == total:
            return None
        abnormal_ratio = abnormal_count / total
        target = min(max(target_abnormal_ratio, 0.01), 0.99)
        abnormal_weight = target / abnormal_ratio
        normal_weight = (1 - target) / (1 - abnormal_ratio)
        weights = [abnormal_weight if flag else normal_weight for flag in abnormal_flags]
        return weights
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample.
        
        Returns:
            Dictionary containing:
                - image: Tensor [3, H, W]
                - text: Structured report string
                - chexbert_labels: Tensor [14] binary labels
                - metadata: Dict with uid, filename, etc.
        """
        sample = self.samples[idx]
        
        if self.require_both_views and self.projection_types:
            image_frontal = Image.open(sample["view_paths"][self.projection_types[0]]).convert("RGB")
            image_lateral = Image.open(sample["view_paths"][self.projection_types[1]]).convert("RGB")
            if self.transform:
                image_frontal = self.transform(image_frontal)
                image_lateral = self.transform(image_lateral)
        else:
            image = Image.open(sample["image_path"]).convert("RGB")
            if self.transform:
                image = self.transform(image)
        
        # Format text
        text = self.text_preprocessor.format_structured_output(
            findings=sample['findings'],
            impression=sample['impression'],
        )
        
        # Get CheXbert labels
        uid_key = str(sample["uid"])
        if uid_key in self.chexbert_labels:
            chexbert_labels = torch.tensor(
                self.chexbert_labels[uid_key],
                dtype=torch.float32
            )
            chexbert_mask = torch.tensor(1.0, dtype=torch.float32)
        else:
            # No labels available for this sample
            chexbert_labels = torch.zeros(14, dtype=torch.float32)
            chexbert_mask = torch.tensor(0.0, dtype=torch.float32)
        
        # Metadata
        if self.require_both_views and self.projection_types:
            metadata = {
                "uid": sample["uid"],
                "view_filenames": sample["view_filenames"],
                "projections": self.projection_types,
            }
            return {
                "image_frontal": image_frontal,
                "image_lateral": image_lateral,
                "text": text,
                "chexbert_labels": chexbert_labels,
                "chexbert_mask": chexbert_mask,
                "metadata": metadata,
            }
        metadata = {
            "uid": sample["uid"],
            "filename": sample["filename"],
            "projection": sample["projection"],
        }
        return {
            "image": image,
            "text": text,
            "chexbert_labels": chexbert_labels,
            "chexbert_mask": chexbert_mask,
            "metadata": metadata,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for DataLoader.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched dictionary
    """
    if "image_frontal" in batch[0]:
        images_frontal = torch.stack([item["image_frontal"] for item in batch])
        images_lateral = torch.stack([item["image_lateral"] for item in batch])
    else:
        images = torch.stack([item["image"] for item in batch])
    texts = [item["text"] for item in batch]
    chexbert_labels = torch.stack([item["chexbert_labels"] for item in batch])
    chexbert_mask = torch.stack([item["chexbert_mask"] for item in batch])
    metadata = [item["metadata"] for item in batch]

    if "image_frontal" in batch[0]:
        return {
            "images_frontal": images_frontal,
            "images_lateral": images_lateral,
            "texts": texts,
            "chexbert_labels": chexbert_labels,
            "chexbert_mask": chexbert_mask,
            "metadata": metadata,
        }
    return {
        "images": images,
        "texts": texts,
        "chexbert_labels": chexbert_labels,
        "chexbert_mask": chexbert_mask,
        "metadata": metadata,
    }
