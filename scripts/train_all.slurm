#!/bin/bash
#SBATCH --job-name=chexquery-all
#SBATCH --output=outputs/logs/train_all_%j.out
#SBATCH --error=outputs/logs/train_all_%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ============================================
# CheXQuery-MedVLM Full Training Pipeline
# ============================================

echo "============================================"
echo "Starting Full Training Pipeline"
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
mkdir -p outputs/checkpoints

# Run all phases
python main.py train \
    --model-config configs/model_config.yaml \
    --train-config configs/train_config.yaml \
    --data-config configs/data_config.yaml \
    --all-phases

echo "============================================"
echo "Full training complete!"
echo "============================================"
