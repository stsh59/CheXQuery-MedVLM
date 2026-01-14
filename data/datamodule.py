import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import Optional

from data.dataset import IUXrayDataset, IUXrayContrastiveDataset
from data.split import load_data_splits, create_data_splits
from utils.config import SIGLIP_MODEL_NAME, RANDOM_SEED
from utils.logger import setup_logger

logger = setup_logger(__name__)


class IUXrayDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for IU-Xray dataset.
    
    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        use_contrastive: Whether to use contrastive dataset
        projection_type: Filter by projection type ('Frontal', 'Lateral', or None)
        max_text_length: Maximum text length for tokenization
    """
    
    def __init__(
        self,
        batch_size: int = 16,
        num_workers: int = 4,
        use_contrastive: bool = True,
        projection_type: Optional[str] = None,
        max_text_length: int = 256
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_contrastive = use_contrastive
        self.projection_type = projection_type
        self.max_text_length = max_text_length
        
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        try:
            splits = load_data_splits()
        except FileNotFoundError:
            logger.info("Creating new data splits...")
            splits = create_data_splits()
        
        if self.use_contrastive:
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(SIGLIP_MODEL_NAME)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            
            if stage == 'fit' or stage is None:
                self.train_dataset = IUXrayContrastiveDataset(
                    splits['train'],
                    self.tokenizer,
                    max_length=self.max_text_length,
                    projection_type=self.projection_type
                )
                
                self.val_dataset = IUXrayContrastiveDataset(
                    splits['val'],
                    self.tokenizer,
                    max_length=self.max_text_length,
                    projection_type=self.projection_type
                )
            
            if stage == 'test':
                self.test_dataset = IUXrayContrastiveDataset(
                    splits['test'],
                    self.tokenizer,
                    max_length=self.max_text_length,
                    projection_type=self.projection_type
                )
        else:
            if stage == 'fit' or stage is None:
                self.train_dataset = IUXrayDataset(
                    splits['train'],
                    projection_type=self.projection_type
                )
                
                self.val_dataset = IUXrayDataset(
                    splits['val'],
                    projection_type=self.projection_type
                )
            
            if stage == 'test':
                self.test_dataset = IUXrayDataset(
                    splits['test'],
                    projection_type=self.projection_type
                )
    
    def train_dataloader(self):
        """Return train dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )


class IUXrayGenerationDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for report generation task.
    Uses regular IUXrayDataset (not contrastive).
    """
    
    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 4,
        projection_type: Optional[str] = None
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.projection_type = projection_type
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        try:
            splits = load_data_splits()
        except FileNotFoundError:
            logger.info("Creating new data splits...")
            splits = create_data_splits()
        
        if stage == 'fit' or stage is None:
            self.train_dataset = IUXrayDataset(
                splits['train'],
                projection_type=self.projection_type
            )
            
            self.val_dataset = IUXrayDataset(
                splits['val'],
                projection_type=self.projection_type
            )
        
        if stage == 'test':
            self.test_dataset = IUXrayDataset(
                splits['test'],
                projection_type=self.projection_type
            )
    
    def collate_fn(self, batch):
        """Custom collate function."""
        images = []
        texts = []
        metadatas = []
        
        for item in batch:
            images.append(item[0])
            texts.append(item[1])
            metadatas.append(item[2])
        
        import torch
        images = torch.stack(images)
        
        return images, texts, metadatas
    
    def train_dataloader(self):
        """Return train dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

