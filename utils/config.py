"""
Global configuration and paths for the Medical-SigLIP project.
"""
import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = Path("/dev/shm/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2")
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Data paths
IMAGES_DIR = DATA_ROOT / "images" / "images_normalized"
REPORTS_CSV = DATA_ROOT / "indiana_reports.csv"
PROJECTIONS_CSV = DATA_ROOT / "indiana_projections.csv"

# Model configurations
SIGLIP_MODEL_NAME = "google/siglip-base-patch16-224"
BIOGPT_MODEL_NAME = "microsoft/biogpt"

# Image preprocessing
IMAGE_SIZE = 224
IMAGE_MEAN = [0.5, 0.5, 0.5]
IMAGE_STD = [0.5, 0.5, 0.5]

# Text preprocessing
MAX_TEXT_LENGTH = 64
MAX_GENERATION_LENGTH = 256

# Training hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 1

# PEFT configurations
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "out_proj"]

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# Device configuration
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

# Logging
LOG_INTERVAL = 10
SAVE_INTERVAL = 1

# Ensure directories exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

