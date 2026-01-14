#!/bin/bash
#SBATCH --job-name=chexquery-p2
#SBATCH --output=outputs/logs/phase2_%j.out
#SBATCH --error=outputs/logs/phase2_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ============================================
# CheXQuery-MedVLM Phase 2 Training: End-to-End Fine-tuning
# ============================================

echo "============================================"
echo "Starting Phase 2: End-to-End Fine-tuning"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "============================================"

# Load modules
module load cuda/11.8
module load python/3.10

# Activate environment
source venv/bin/activate

# Set environment variables
export PYTHONUNBUFFERED=1
export TRANSFORMERS_CACHE=/scratch/$USER/.cache/huggingface
export HF_HOME=/scratch/$USER/.cache/huggingface

# Create output directories
mkdir -p outputs/logs
mkdir -p outputs/checkpoints/phase2

# Find best Phase 1 checkpoint
PHASE1_CKPT=$(ls -t outputs/checkpoints/phase1/*.ckpt | head -1)

if [ -z "$PHASE1_CKPT" ]; then
    echo "Error: No Phase 1 checkpoint found!"
    exit 1
fi

echo "Using Phase 1 checkpoint: $PHASE1_CKPT"

# Run Phase 2 training
python main.py train \
    --model-config configs/model_config.yaml \
    --train-config configs/train_config.yaml \
    --data-config configs/data_config.yaml \
    --phase 2 \
    --checkpoint-dir outputs/checkpoints \
    --resume "$PHASE1_CKPT"

echo "============================================"
echo "Phase 2 training complete!"
echo "============================================"
