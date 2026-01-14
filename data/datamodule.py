"""
PyTorch Lightning DataModule for Chest X-ray Report Generation.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import kagglehub
from transformers import AutoProcessor
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pandas as pd

from data.dataset import ChestXrayDataset, collate_fn

logger = logging.getLogger(__name__)


class ChestXrayDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for chest X-ray report generation.
    
    Handles data downloading, splitting, and loading.
    
    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of dataloader workers
        image_size: Target image size
        projection_type: Filter by projection type
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        seed: Random seed for splitting
    """
    
    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 4,
        image_size: int = 384,
        projection_type: str = "Frontal",
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
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.projection_type = projection_type
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
        
        self.data_root = None
        self.splits = None
        self.chexbert_labels = {}
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def prepare_data(self):
        """Download data if not present."""
        logger.info("Downloading dataset via kagglehub...")
        self.data_root = Path(kagglehub.dataset_download(
            "raddar/chest-xrays-indiana-university"
        ))
        logger.info(f"Dataset downloaded to: {self.data_root}")
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage."""
        if self.data_root is None:
            self.data_root = Path(kagglehub.dataset_download(
                "raddar/chest-xrays-indiana-university"
            ))
        
        # Load or create splits
        if self.splits is None:
            self.splits = self._get_or_create_splits()
        
        # Load CheXbert labels if available
        self._load_chexbert_labels()
        
        if stage == "fit" or stage is None:
            self.train_dataset = ChestXrayDataset(
                data_root=self.data_root,
                uids=self.splits['train'],
                split="train",
                image_size=self.image_size,
                projection_type=self.projection_type,
                chexbert_labels=self.chexbert_labels,
                text_output_template=self.text_output_template,
                text_max_length=self.text_max_length,
                image_mean=self.image_mean,
                image_std=self.image_std,
                augmentation_config=self.augmentation_config,
            )
            
            self.val_dataset = ChestXrayDataset(
                data_root=self.data_root,
                uids=self.splits['val'],
                split="val",
                image_size=self.image_size,
                projection_type=self.projection_type,
                chexbert_labels=self.chexbert_labels,
                text_output_template=self.text_output_template,
                text_max_length=self.text_max_length,
                image_mean=self.image_mean,
                image_std=self.image_std,
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = ChestXrayDataset(
                data_root=self.data_root,
                uids=self.splits['test'],
                split="test",
                image_size=self.image_size,
                projection_type=self.projection_type,
                chexbert_labels=self.chexbert_labels,
                text_output_template=self.text_output_template,
                text_max_length=self.text_max_length,
                image_mean=self.image_mean,
                image_std=self.image_std,
            )
    
    def _get_or_create_splits(self) -> Dict[str, List[int]]:
        """Load existing splits or create new ones."""
        # Try to load existing splits
        if self.splits_file:
            splits_path = Path(self.splits_file)
            if splits_path.exists():
                logger.info(f"Loading splits from {splits_path}")
                with open(splits_path, 'r') as f:
                    return json.load(f)
        
        # Create new splits
        logger.info("Creating new patient-level splits...")
        reports_df = pd.read_csv(self.data_root / "indiana_reports.csv")
        unique_uids = reports_df['uid'].unique().tolist()
        
        # Split by patient
        train_uids, temp_uids = train_test_split(
            unique_uids,
            test_size=(self.val_ratio + self.test_ratio),
            random_state=self.seed
        )
        
        val_relative_size = self.val_ratio / (self.val_ratio + self.test_ratio)
        val_uids, test_uids = train_test_split(
            temp_uids,
            test_size=(1 - val_relative_size),
            random_state=self.seed
        )
        
        splits = {
            'train': train_uids,
            'val': val_uids,
            'test': test_uids,
        }
        
        # Save splits
        if self.splits_file:
            splits_path = Path(self.splits_file)
            splits_path.parent.mkdir(parents=True, exist_ok=True)
            with open(splits_path, 'w') as f:
                json.dump(splits, f, indent=2)
            logger.info(f"Saved splits to {splits_path}")
        
        logger.info(f"Splits: train={len(train_uids)}, val={len(val_uids)}, test={len(test_uids)}")
        return splits
    
    def _load_chexbert_labels(self):
        """Load pre-computed CheXbert labels if available."""
        labels_path = Path("outputs/chexbert_labels.json")
        if labels_path.exists():
            logger.info("Loading pre-computed CheXbert labels...")
            with open(labels_path, 'r') as f:
                self.chexbert_labels = json.load(f)
        else:
            logger.info("No pre-computed CheXbert labels found. Will use defaults.")
    
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
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
