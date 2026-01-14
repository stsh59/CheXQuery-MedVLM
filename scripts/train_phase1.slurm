#!/bin/bash
#SBATCH --job-name=chexquery-p1
#SBATCH --output=outputs/logs/phase1_%j.out
#SBATCH --error=outputs/logs/phase1_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ============================================
# CheXQuery-MedVLM Phase 1 Training: Query Alignment
# ============================================

echo "============================================"
echo "Starting Phase 1: Query Alignment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "============================================"

# Load modules (adjust for your cluster)
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
mkdir -p outputs/checkpoints/phase1

# Run Phase 1 training
python main.py train \
    --model-config configs/model_config.yaml \
    --train-config configs/train_config.yaml \
    --data-config configs/data_config.yaml \
    --phase 1 \
    --checkpoint-dir outputs/checkpoints

echo "============================================"
echo "Phase 1 training complete!"
echo "============================================"
