"""
Patient-level train/val/test split to prevent data leakage.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List
from sklearn.model_selection import train_test_split

from utils.config import (
    REPORTS_CSV, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED, OUTPUT_DIR
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


def create_data_splits(
    save_dir: Path = OUTPUT_DIR / "splits",
    random_seed: int = RANDOM_SEED
) -> Dict[str, List[int]]:
    """
    Create train/val/test splits at the patient level.
    
    Args:
        save_dir: Directory to save split indices
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with 'train', 'val', 'test' lists of uids
    """
    logger.info("Loading reports CSV...")
    reports_df = pd.read_csv(REPORTS_CSV)
    
    unique_uids = reports_df['uid'].unique()
    logger.info(f"Found {len(unique_uids)} unique patients")
    
    val_test_ratio = VAL_RATIO + TEST_RATIO
    
    train_uids, temp_uids = train_test_split(
        unique_uids,
        test_size=val_test_ratio,
        random_state=random_seed
    )
    
    val_relative_size = VAL_RATIO / val_test_ratio
    val_uids, test_uids = train_test_split(
        temp_uids,
        test_size=(1 - val_relative_size),
        random_state=random_seed
    )
    
    splits = {
        'train': train_uids.tolist(),
        'val': val_uids.tolist(),
        'test': test_uids.tolist()
    }
    
    logger.info(f"Split sizes - Train: {len(train_uids)}, Val: {len(val_uids)}, Test: {len(test_uids)}")
    
    save_dir.mkdir(parents=True, exist_ok=True)
    split_file = save_dir / "data_splits.json"
    
    with open(split_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    logger.info(f"Saved splits to {split_file}")
    
    return splits


def load_data_splits(split_file: Path = OUTPUT_DIR / "splits" / "data_splits.json") -> Dict[str, List[int]]:
    """
    Load existing data splits.
    
    Args:
        split_file: Path to split JSON file
    
    Returns:
        Dictionary with 'train', 'val', 'test' lists of uids
    """
    if not split_file.exists():
        logger.warning(f"Split file {split_file} not found. Creating new splits...")
        return create_data_splits()
    
    with open(split_file, 'r') as f:
        splits = json.load(f)
    
    logger.info(f"Loaded splits from {split_file}")
    logger.info(f"Split sizes - Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    
    return splits


if __name__ == "__main__":
    splits = create_data_splits()
    print("Data splits created successfully!")
    print(f"Train: {len(splits['train'])} patients")
    print(f"Val: {len(splits['val'])} patients")
    print(f"Test: {len(splits['test'])} patients")

