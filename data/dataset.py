"""
PyTorch Dataset for Chest X-ray Report Generation (MIMIC-CXR).
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

# ViewPosition -> canonical category mapping
VIEW_POSITION_MAP: Dict[str, str] = {
    "PA": "Frontal",
    "AP": "Frontal",
    "LATERAL": "Lateral",
    "LL": "Lateral",
}


class ChestXrayDataset(Dataset):
    """
    Dataset for chest X-ray report generation using MIMIC-CXR.

    Loads image-report pairs from a pre-built balanced CSV and provides
    CheXbert labels for auxiliary training.

    The balanced CSV is produced by ``data.create_balanced_sample`` and
    contains one row per study with columns:
        study_id, subject_id, dicom_id, view_position, view_category,
        image_path, report_path, findings, impression, keyword_labels

    Args:
        data_root: Root directory of the MIMIC-CXR dataset.
        balanced_csv: Path to the balanced subset CSV.
        study_ids: List of study IDs to include (for train/val/test split).
        split: Dataset split ('train', 'val', 'test').
        image_size: Target image size.
        transform: Optional custom transform.
        projection_type: Canonical projection filter ('Frontal', 'Lateral', or None).
        projection_types: List of projection types for multi-view.
        require_both_views: If True, require both Frontal and Lateral views.
        chexbert_labels: Optional pre-computed CheXbert labels dict.
        text_output_template: Template for structured report output.
        text_max_length: Maximum text length.
        image_mean: Image normalization mean.
        image_std: Image normalization std.
        augmentation_config: Augmentation hyperparameters.
    """

    def __init__(
        self,
        data_root: str,
        balanced_csv: str,
        study_ids: List[int],
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
        self.balanced_csv = Path(balanced_csv)
        self.study_ids = set(int(s) for s in study_ids)
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
        """Load balanced CSV and build sample list for the current split."""
        logger.info(f"Loading {self.split} dataset from {self.balanced_csv} ...")

        df = pd.read_csv(self.balanced_csv)

        # Filter to studies in this split
        df = df[df["study_id"].isin(self.study_ids)]

        if self.require_both_views and self.projection_types:
            self.samples = self._build_paired_samples(df)
        else:
            self.samples = self._build_single_view_samples(df)

        logger.info(f"Loaded {len(self.samples)} {self.split} samples")

    def _is_valid_image(self, image_path: Path) -> bool:
        """Check whether an image file exists and is readable."""
        if not image_path.exists():
            return False
        try:
            img = Image.open(image_path)
            img.verify()
        except Exception:
            return False
        return True

    def _build_single_view_samples(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Build single-view samples from the balanced CSV."""
        valid_samples = []

        # The balanced CSV already has the best frontal image selected,
        # but we still validate each image.
        for _, row in df.iterrows():
            image_path = Path(row["image_path"])
            if not self._is_valid_image(image_path):
                continue

            findings = str(row.get("findings", "")) if pd.notna(row.get("findings")) else ""
            impression = str(row.get("impression", "")) if pd.notna(row.get("impression")) else ""

            if not impression.strip():
                continue

            valid_samples.append({
                "study_id": int(row["study_id"]),
                "subject_id": int(row["subject_id"]),
                "dicom_id": row["dicom_id"],
                "image_path": str(image_path),
                "view_position": row.get("view_position", ""),
                "findings": findings,
                "impression": impression,
            })

        return valid_samples

    def _build_paired_samples(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """
        Build paired (frontal + lateral) samples.

        For multi-view training we need the full metadata (not just the
        balanced CSV which contains one row per study).  We read metadata.csv
        to find lateral counterparts for each frontal study.
        """
        # Load full metadata for lateral-view lookup
        metadata_path = self.data_root / "metadata.csv"
        if not metadata_path.exists():
            logger.warning("metadata.csv not found; cannot build paired samples. Falling back to single-view.")
            return self._build_single_view_samples(df)

        full_meta = pd.read_csv(metadata_path)
        full_meta["view_category"] = full_meta["ViewPosition"].map(VIEW_POSITION_MAP).fillna("Other")
        full_meta["prefix"] = "p" + full_meta["subject_id"].astype(str).str[:2]

        projection_types = [p for p in self.projection_types if p]
        required = set(projection_types)
        paired_samples = []
        dropped = 0

        for _, row in df.iterrows():
            study_id = int(row["study_id"])
            subject_id = int(row["subject_id"])
            findings = str(row.get("findings", "")) if pd.notna(row.get("findings")) else ""
            impression = str(row.get("impression", "")) if pd.notna(row.get("impression")) else ""

            if not impression.strip():
                continue

            # Get all images for this study from full metadata
            study_meta = full_meta[full_meta["study_id"] == study_id]
            if study_meta.empty:
                dropped += 1
                continue

            prefix = study_meta.iloc[0]["prefix"]
            view_paths = {}
            view_filenames = {}
            for proj in projection_types:
                proj_rows = study_meta[study_meta["view_category"] == proj]
                found = False
                for _, img_row in proj_rows.iterrows():
                    dicom_id = img_row["dicom_id"]
                    img_path = (
                        self.data_root
                        / "official_data_iccv_final" / "files"
                        / prefix
                        / f"p{subject_id}"
                        / f"s{study_id}"
                        / f"{dicom_id}.jpg"
                    )
                    if self._is_valid_image(img_path):
                        view_paths[proj] = str(img_path)
                        view_filenames[proj] = f"{dicom_id}.jpg"
                        found = True
                        break
                if not found:
                    break

            if set(view_paths.keys()) != required:
                dropped += 1
                continue

            paired_samples.append({
                "study_id": study_id,
                "subject_id": subject_id,
                "view_paths": view_paths,
                "view_filenames": view_filenames,
                "findings": findings,
                "impression": impression,
            })

        logger.info(
            f"Paired samples built: {len(paired_samples)}; dropped (missing views): {dropped}"
        )
        return paired_samples

    def __len__(self) -> int:
        return len(self.samples)

    def get_sampling_weights(
        self,
        target_abnormal_ratio: float = 0.5,
        strategy: str = "condition_aware",
    ) -> Optional[List[float]]:
        """
        Compute sampling weights for the training dataloader.

        Strategies:
          - "binary": Original binary normal/abnormal oversampling.
          - "condition_aware": Weight each sample by the rarity of its
            rarest positive condition.

        Args:
            target_abnormal_ratio: Target fraction for binary strategy.
            strategy: Sampling strategy ("binary" or "condition_aware").

        Returns:
            List of per-sample weights, or None if not applicable.
        """
        if not self.samples:
            return None

        num_samples = len(self.samples)

        if strategy == "condition_aware":
            return self._condition_aware_weights(num_samples)
        else:
            return self._binary_weights(num_samples, target_abnormal_ratio)

    def _binary_weights(
        self,
        num_samples: int,
        target_abnormal_ratio: float,
    ) -> Optional[List[float]]:
        """Original binary normal/abnormal oversampling."""
        abnormal_flags = []
        for sample in self.samples:
            uid_key = str(sample["study_id"])
            labels = self.chexbert_labels.get(uid_key)
            if labels is None:
                abnormal_flags.append(False)
                continue
            abnormal_flags.append(any(labels[1:]))
        abnormal_count = sum(abnormal_flags)
        if abnormal_count == 0 or abnormal_count == num_samples:
            return None
        abnormal_ratio = abnormal_count / num_samples
        target = min(max(target_abnormal_ratio, 0.01), 0.99)
        abnormal_weight = target / abnormal_ratio
        normal_weight = (1 - target) / (1 - abnormal_ratio)
        return [abnormal_weight if f else normal_weight for f in abnormal_flags]

    def _condition_aware_weights(self, num_samples: int) -> Optional[List[float]]:
        """
        Condition-aware sampling: weight by inverse frequency of rarest
        positive condition per sample.
        """
        num_labels = 14
        label_counts = [0] * num_labels
        sample_labels = []

        for sample in self.samples:
            uid_key = str(sample["study_id"])
            labels = self.chexbert_labels.get(uid_key)
            if labels is None:
                sample_labels.append(None)
                continue
            sample_labels.append(labels)
            for i in range(num_labels):
                if labels[i]:
                    label_counts[i] += 1

        # Per-label inverse-frequency weight
        label_weights = []
        for count in label_counts:
            if count > 0:
                label_weights.append(num_samples / count)
            else:
                label_weights.append(1.0)

        # Cap to prevent extreme outliers (max 50x)
        max_weight = 50.0
        label_weights = [min(w, max_weight) for w in label_weights]

        # Per-sample weight = max label_weight across positive disease labels
        baseline = 1.0
        disease_indices = list(range(1, 13))

        weights = []
        for labels in sample_labels:
            if labels is None:
                weights.append(baseline)
                continue
            disease_weights = [
                label_weights[i] for i in disease_indices if labels[i]
            ]
            if disease_weights:
                weights.append(max(disease_weights))
            else:
                weights.append(baseline)

        # Normalize so mean weight = 1
        mean_w = sum(weights) / len(weights)
        if mean_w > 0:
            weights = [w / mean_w for w in weights]

        return weights

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample.

        Returns:
            Dictionary containing:
                - image: Tensor [3, H, W]
                - text: Structured report string
                - chexbert_labels: Tensor [14] binary labels
                - chexbert_mask: Tensor scalar (1.0 if labels available)
                - metadata: Dict with study_id, dicom_id, etc.
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
            findings=sample["findings"],
            impression=sample["impression"],
        )

        # Get CheXbert labels (keyed by study_id)
        uid_key = str(sample["study_id"])
        if uid_key in self.chexbert_labels:
            chexbert_labels = torch.tensor(
                self.chexbert_labels[uid_key],
                dtype=torch.float32,
            )
            chexbert_mask = torch.tensor(1.0, dtype=torch.float32)
        else:
            chexbert_labels = torch.zeros(14, dtype=torch.float32)
            chexbert_mask = torch.tensor(0.0, dtype=torch.float32)

        # Metadata
        if self.require_both_views and self.projection_types:
            metadata = {
                "uid": sample["study_id"],
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
            "uid": sample["study_id"],
            "filename": f"{sample['dicom_id']}.jpg",
            "projection": sample.get("view_position", ""),
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
