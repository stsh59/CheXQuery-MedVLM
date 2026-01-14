import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Optional, List, Tuple
from torchvision import transforms

from utils.config import (
    IMAGES_DIR, REPORTS_CSV, PROJECTIONS_CSV,
    IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD
)
from utils.tokenizer_utils import clean_medical_text, extract_impression_only, preprocess_findings_and_impression
from utils.logger import setup_logger

logger = setup_logger(__name__)


class IUXrayDataset(Dataset):
    """
    IU-Xray Dataset for image-text pairs.
    
    Args:
        uids: List of patient uids to include
        transform: Optional image transformations
        use_findings: Whether to include findings in text (default: False, impression only)
        projection_type: Filter by projection type ('Frontal', 'Lateral', or None for all)
    """
    
    def __init__(
        self,
        uids: List[int],
        transform: Optional[transforms.Compose] = None,
        use_findings: bool = True,
        projection_type: Optional[str] = None
    ):
        self.uids = set(uids)
        self.transform = transform or self._get_default_transform()
        self.use_findings = use_findings
        self.projection_type = projection_type
        
        self._load_data()
    
    def _get_default_transform(self):
        """Get default image transforms for SigLIP."""
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
        ])
    
    def _load_data(self):
        """Load and merge reports and projections data."""
        logger.info("Loading dataset...")
        
        reports_df = pd.read_csv(REPORTS_CSV)
        projections_df = pd.read_csv(PROJECTIONS_CSV)
        
        reports_df = reports_df[reports_df['uid'].isin(self.uids)]
        
        merged_df = projections_df.merge(reports_df, on='uid', how='inner')
        
        if self.projection_type:
            merged_df = merged_df[merged_df['projection'] == self.projection_type]
        
        # Validate and filter images
        logger.info("Validating image files...")
        valid_indices = []
        image_filtered = 0
        
        for idx in range(len(merged_df)):
            row = merged_df.iloc[idx]
            image_path = IMAGES_DIR / row['filename']
            
            if not image_path.exists():
                logger.warning(f"Missing image: {row['filename']} (UID: {row['uid']})")
                image_filtered += 1
                continue
            
            try:
                img = Image.open(image_path).convert('RGB')
                img.close()
                valid_indices.append(idx)
            except Exception as e:
                logger.warning(f"Corrupted image: {row['filename']} (UID: {row['uid']}): {e}")
                image_filtered += 1
        
        merged_df = merged_df.iloc[valid_indices].reset_index(drop=True)
        logger.info(f"Filtered {image_filtered} invalid images")
        
        # Drop NaN impressions
        merged_df = merged_df.dropna(subset=['impression'])
        
        # Filter out empty/whitespace-only impressions (safe type conversion)
        merged_df = merged_df[merged_df['impression'].astype(str).str.strip().str.len() > 0]
        
        # Ensure at least findings OR impression has meaningful content
        def has_valid_text(row):
            findings = row.get('findings', '')
            impression = row.get('impression', '')
            
            # Check if findings has content (not NaN, not empty, not whitespace)
            has_findings = pd.notna(findings) and str(findings).strip() != ''
            
            # Check if impression has content (already filtered above, but double-check)
            has_impression = pd.notna(impression) and str(impression).strip() != ''
            
            # At least one must have real content
            return has_findings or has_impression
        
        merged_df = merged_df[merged_df.apply(has_valid_text, axis=1)]
        
        # Safety check: ensure dataset is not empty after filtering
        if len(merged_df) == 0:
            logger.error("No valid samples remaining after image and text validation!")
            raise ValueError(f"Dataset empty after filtering. Check data quality.")
        
        logger.info(f"After text validation: {len(merged_df)} samples with valid medical text")
        
        self.data = merged_df.reset_index(drop=True)
        
        logger.info(f"Loaded {len(self.data)} image-text pairs")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, dict]:
        """
        Get an item from the dataset.
        
        Returns:
            image: Preprocessed image tensor
            text: Medical report text
            metadata: Dictionary with uid, filename, projection
        """
        row = self.data.iloc[idx]
        
        image_path = IMAGES_DIR / row['filename']
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        # Convert NaN to empty string to avoid "nan" strings
        findings = '' if pd.isna(row.get('findings')) else str(row['findings'])
        impression = '' if pd.isna(row.get('impression')) else str(row['impression'])
        
        # Use utility function with smart fallback
        text = preprocess_findings_and_impression(findings, impression)
        
        metadata = {
            'uid': row['uid'],
            'filename': row['filename'],
            'projection': row['projection']
        }
        
        return image, text, metadata


class IUXrayContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning (image-text pairs).
    Returns images and text separately for contrastive loss.
    """
    
    def __init__(
        self,
        uids: List[int],
        tokenizer,
        max_length: int = 256,
        transform: Optional[transforms.Compose] = None,
        projection_type: Optional[str] = None,
        use_findings: bool = True
    ):
        self.uids = set(uids)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform or self._get_default_transform()
        self.projection_type = projection_type
        self.use_findings = use_findings
        
        self._load_data()
    
    def _get_default_transform(self):
        """Get default image transforms."""
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
        ])
    
    def _load_data(self):
        """Load and process data."""
        reports_df = pd.read_csv(REPORTS_CSV)
        projections_df = pd.read_csv(PROJECTIONS_CSV)
        
        reports_df = reports_df[reports_df['uid'].isin(self.uids)]
        merged_df = projections_df.merge(reports_df, on='uid', how='inner')
        
        if self.projection_type:
            merged_df = merged_df[merged_df['projection'] == self.projection_type]
        
        # Validate and filter images
        logger.info("Validating image files...")
        valid_indices = []
        image_filtered = 0
        
        for idx in range(len(merged_df)):
            row = merged_df.iloc[idx]
            image_path = IMAGES_DIR / row['filename']
            
            if not image_path.exists():
                logger.warning(f"Missing image: {row['filename']} (UID: {row['uid']})")
                image_filtered += 1
                continue
            
            try:
                img = Image.open(image_path).convert('RGB')
                img.close()
                valid_indices.append(idx)
            except Exception as e:
                logger.warning(f"Corrupted image: {row['filename']} (UID: {row['uid']}): {e}")
                image_filtered += 1
        
        merged_df = merged_df.iloc[valid_indices].reset_index(drop=True)
        logger.info(f"Filtered {image_filtered} invalid images")
        
        # Drop NaN impressions
        merged_df = merged_df.dropna(subset=['impression'])
        
        # Filter out empty/whitespace-only impressions (safe type conversion)
        merged_df = merged_df[merged_df['impression'].astype(str).str.strip().str.len() > 0]
        
        # Ensure at least findings OR impression has meaningful content
        def has_valid_text(row):
            findings = row.get('findings', '')
            impression = row.get('impression', '')
            has_findings = pd.notna(findings) and str(findings).strip() != ''
            has_impression = pd.notna(impression) and str(impression).strip() != ''
            return has_findings or has_impression
        
        merged_df = merged_df[merged_df.apply(has_valid_text, axis=1)]
        
        # Safety check: ensure dataset is not empty after filtering
        if len(merged_df) == 0:
            logger.error("No valid samples remaining after image and text validation!")
            raise ValueError(f"Dataset empty after filtering. Check data quality.")
        
        logger.info(f"After text validation: {len(merged_df)} samples with valid medical text")
        
        self.data = merged_df.reset_index(drop=True)
        logger.info(f"Loaded {len(self.data)} image-text pairs for contrastive learning")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int):
        """Get image and tokenized text."""
        row = self.data.iloc[idx]
        
        image_path = IMAGES_DIR / row['filename']
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        if self.use_findings:
            # Convert NaN to empty string
            findings = '' if pd.isna(row.get('findings')) else str(row['findings'])
            impression = '' if pd.isna(row.get('impression')) else str(row['impression'])
            text = preprocess_findings_and_impression(findings, impression)
        else:
            text = extract_impression_only(str(row['impression']))
        
        text_encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        if 'attention_mask' not in text_encoded:
            # Create attention mask manually: 1 for non-pad, 0 for pad
            pad_token_id = self.tokenizer.pad_token_id
            input_ids = text_encoded['input_ids']
            attention_mask = (input_ids != pad_token_id).long()
            text_encoded['attention_mask'] = attention_mask

        return {
            'image': image,
            'input_ids': text_encoded['input_ids'].squeeze(0),
            'attention_mask': text_encoded['attention_mask'].squeeze(0),
            'uid': row['uid']
        }

