#!/bin/bash
# Prepare data splits - run this once before training

echo "Preparing data splits..."

# Navigate to project directory
cd "$(dirname "$0")"

# Create directories
mkdir -p outputs/splits outputs/logs

# Prepare data with proper bind mount
srun apptainer exec --no-home --bind /dev/shm:/dev/shm --nv /home/shared/sif/csci-2025-Fall.sif \
  bash -c "export HOME=/dev/shm; \
           export KAGGLEHUB_CACHE_DIR=/dev/shm/kagglehub; \
           python main.py prepare_data"

echo "Data preparation complete!"

