"""
PyTorch Lightning DataModule for Chest X-ray Report Generation (MIMIC-CXR).
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from transformers import AutoProcessor
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd

from data.dataset import ChestXrayDataset, collate_fn

logger = logging.getLogger(__name__)


class ChestXrayDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for chest X-ray report generation (MIMIC-CXR).

    Works with a pre-built balanced CSV produced by
    ``data.create_balanced_sample``.  Splits are patient-level
    (by ``subject_id``) to prevent data leakage.

    Args:
        data_root: Path to the MIMIC-CXR dataset directory.
        balanced_csv: Path to balanced subset CSV.
        batch_size: Batch size for dataloaders.
        num_workers: Number of dataloader workers.
        image_size: Target image size.
        projection_type: Canonical projection filter.
        projection_types: List of projections for multi-view.
        require_both_views: Whether to require both views.
        train_ratio: Training set ratio.
        val_ratio: Validation set ratio.
        seed: Random seed for splitting.
        splits_file: Path to save/load split JSON.
        text_output_template: Template for structured report output.
        text_max_length: Maximum text length.
        image_mean: Image normalization mean.
        image_std: Image normalization std.
        augmentation_config: Augmentation hyperparameters.
        use_siglip_processor: Whether to use SigLIP processor stats.
        processor_model: SigLIP processor model name.
        sampling_config: Sampling strategy configuration.
    """

    def __init__(
        self,
        data_root: str = "mimic-cxr-dataset",
        balanced_csv: str = "outputs/mimic_cxr_balanced.csv",
        batch_size: int = 8,
        num_workers: int = 4,
        image_size: int = 384,
        projection_type: str = "Frontal",
        projection_types: Optional[List[str]] = None,
        require_both_views: bool = False,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
        splits_file: Optional[str] = None,
        text_output_template: Optional[str] = None,
        text_max_length: int = 512,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        augmentation_config: Optional[Dict[str, float]] = None,
        use_siglip_processor: bool = False,
        processor_model: Optional[str] = None,
        sampling_config: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.balanced_csv = Path(balanced_csv)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.projection_type = projection_type
        self.projection_types = projection_types
        self.require_both_views = require_both_views
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - train_ratio - val_ratio
        self.seed = seed
        self.splits_file = splits_file
        self.text_output_template = text_output_template
        self.text_max_length = text_max_length
        self.image_mean = image_mean
        self.image_std = image_std
        self.augmentation_config = augmentation_config or {}
        self.sampling_config = sampling_config or {}

        # Optionally align preprocessing with SigLIP processor
        if use_siglip_processor and processor_model:
            processor = AutoProcessor.from_pretrained(processor_model)
            image_processor = getattr(processor, "image_processor", None)
            if image_processor is not None:
                if hasattr(image_processor, "size") and isinstance(image_processor.size, dict):
                    self.image_size = image_processor.size.get("height", self.image_size)
                if hasattr(image_processor, "image_mean"):
                    self.image_mean = image_processor.image_mean
                if hasattr(image_processor, "image_std"):
                    self.image_std = image_processor.image_std

        self.splits = None
        self.chexbert_labels = {}

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """Verify that the dataset and balanced CSV exist."""
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"MIMIC-CXR dataset directory not found: {self.data_root}\n"
                f"Please ensure the dataset is placed at the expected path."
            )
        if not self.balanced_csv.exists():
            raise FileNotFoundError(
                f"Balanced CSV not found: {self.balanced_csv}\n"
                f"Run `python main.py balance` first to create the balanced subset."
            )
        logger.info(f"Dataset root: {self.data_root}")
        logger.info(f"Balanced CSV: {self.balanced_csv}")

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage."""
        # Load or create splits
        if self.splits is None:
            self.splits = self._get_or_create_splits()

        # Load CheXbert labels if available
        self._load_chexbert_labels()

        if stage == "fit" or stage is None:
            self.train_dataset = ChestXrayDataset(
                data_root=str(self.data_root),
                balanced_csv=str(self.balanced_csv),
                study_ids=self.splits["train"],
                split="train",
                image_size=self.image_size,
                projection_type=self.projection_type,
                projection_types=self.projection_types,
                require_both_views=self.require_both_views,
                chexbert_labels=self.chexbert_labels,
                text_output_template=self.text_output_template,
                text_max_length=self.text_max_length,
                image_mean=self.image_mean,
                image_std=self.image_std,
                augmentation_config=self.augmentation_config,
            )

            self.val_dataset = ChestXrayDataset(
                data_root=str(self.data_root),
                balanced_csv=str(self.balanced_csv),
                study_ids=self.splits["val"],
                split="val",
                image_size=self.image_size,
                projection_type=self.projection_type,
                projection_types=self.projection_types,
                require_both_views=self.require_both_views,
                chexbert_labels=self.chexbert_labels,
                text_output_template=self.text_output_template,
                text_max_length=self.text_max_length,
                image_mean=self.image_mean,
                image_std=self.image_std,
            )

        if stage == "test" or stage is None:
            self.test_dataset = ChestXrayDataset(
                data_root=str(self.data_root),
                balanced_csv=str(self.balanced_csv),
                study_ids=self.splits["test"],
                split="test",
                image_size=self.image_size,
                projection_type=self.projection_type,
                projection_types=self.projection_types,
                require_both_views=self.require_both_views,
                chexbert_labels=self.chexbert_labels,
                text_output_template=self.text_output_template,
                text_max_length=self.text_max_length,
                image_mean=self.image_mean,
                image_std=self.image_std,
            )

    def _get_or_create_splits(self) -> Dict[str, List[int]]:
        """Load existing splits or create new patient-level splits."""
        # Try to load existing splits
        if self.splits_file:
            splits_path = Path(self.splits_file)
            if splits_path.exists():
                logger.info(f"Loading splits from {splits_path}")
                with open(splits_path, "r") as f:
                    return json.load(f)

        # Create new splits from balanced CSV
        logger.info("Creating new patient-level splits from balanced CSV ...")
        balanced_df = pd.read_csv(self.balanced_csv)

        # Get unique patients
        patient_study_map = (
            balanced_df.groupby("subject_id")["study_id"]
            .apply(list)
            .to_dict()
        )
        unique_patients = list(patient_study_map.keys())

        # Split by patient (subject_id)
        train_patients, temp_patients = train_test_split(
            unique_patients,
            test_size=(self.val_ratio + self.test_ratio),
            random_state=self.seed,
        )

        val_relative_size = self.val_ratio / (self.val_ratio + self.test_ratio)
        val_patients, test_patients = train_test_split(
            temp_patients,
            test_size=(1 - val_relative_size),
            random_state=self.seed,
        )

        # Map patients -> study_ids
        def _patient_to_studies(patients):
            studies = []
            for pid in patients:
                studies.extend(patient_study_map[pid])
            return studies

        splits = {
            "train": _patient_to_studies(train_patients),
            "val": _patient_to_studies(val_patients),
            "test": _patient_to_studies(test_patients),
        }

        # Save splits
        if self.splits_file:
            splits_path = Path(self.splits_file)
            splits_path.parent.mkdir(parents=True, exist_ok=True)
            with open(splits_path, "w") as f:
                json.dump(splits, f, indent=2)
            logger.info(f"Saved splits to {splits_path}")

        logger.info(
            f"Splits: train={len(splits['train'])} studies "
            f"({len(train_patients)} patients), "
            f"val={len(splits['val'])} studies "
            f"({len(val_patients)} patients), "
            f"test={len(splits['test'])} studies "
            f"({len(test_patients)} patients)"
        )
        return splits

    def _load_chexbert_labels(self):
        """Load pre-computed CheXbert labels if available."""
        labels_path = Path("outputs/chexbert_labels.json")
        if labels_path.exists():
            logger.info("Loading pre-computed CheXbert labels ...")
            with open(labels_path, "r") as f:
                self.chexbert_labels = json.load(f)
        else:
            logger.info("No pre-computed CheXbert labels found. Will use defaults.")

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        sampler = None
        if self.sampling_config.get("enable", False):
            target_ratio = float(self.sampling_config.get("abnormal_ratio", 0.5))
            strategy = self.sampling_config.get("strategy", "condition_aware")
            weights = self.train_dataset.get_sampling_weights(
                target_abnormal_ratio=target_ratio,
                strategy=strategy,
            )
            if weights is not None:
                sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
                logger.info(f"Using {strategy} weighted sampling with {len(weights)} samples")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
