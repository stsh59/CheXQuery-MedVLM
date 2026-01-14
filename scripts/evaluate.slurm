#!/bin/bash
#SBATCH --job-name=chexquery-eval
#SBATCH --output=outputs/logs/eval_%j.out
#SBATCH --error=outputs/logs/eval_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ============================================
# CheXQuery-MedVLM Evaluation
# ============================================

echo "============================================"
echo "Starting Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "============================================"

# Load modules
module load cuda/11.8
module load python/3.10

# Activate environment
source venv/bin/activate

# Set environment variables
export PYTHONUNBUFFERED=1
export TRANSFORMERS_CACHE=/scratch/$USER/.cache/huggingface

# Find best checkpoint
BEST_CKPT=${1:-$(ls -t outputs/checkpoints/phase2/*.ckpt | head -1)}

if [ -z "$BEST_CKPT" ]; then
    echo "Error: No checkpoint found!"
    exit 1
fi

echo "Evaluating checkpoint: $BEST_CKPT"

# Run evaluation
python main.py evaluate \
    --checkpoint "$BEST_CKPT" \
    --output-dir outputs/evaluation \
    --batch-size 16 \
    --num-beams 4 \
    --split test

echo "============================================"
echo "Evaluation complete!"
echo "============================================"
