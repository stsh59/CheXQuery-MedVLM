#!/bin/bash
#SBATCH --job-name=lora_training
#SBATCH --output=outputs/logs/lora_%j.out
#SBATCH --error=outputs/logs/lora_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Node: $SLURM_NODELIST"

# Navigate to project directory
cd "$SLURM_SUBMIT_DIR"

# Create directories
mkdir -p outputs/logs checkpoints

# Set environment
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Check GPU
nvidia-smi

# Run training
echo "Starting LoRA training..."
srun apptainer exec --no-home --bind /dev/shm:/dev/shm --nv /home/shared/sif/csci-2025-Fall.sif \
  bash -c "export HOME=/dev/shm; \
           export KAGGLEHUB_CACHE_DIR=/dev/shm/kagglehub; \
           python main.py train_siglip \
             --peft_method lora \
             --num_epochs 15 \
             --batch_size 32"

echo "Training completed at: $(date)"

