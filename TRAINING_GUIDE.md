# Training Guide for Medical-SigLIP

## Quick Start

### Step 1: Prepare Data (Run Once)
```bash
chmod +x prepare_data.sh
./prepare_data.sh
```

### Step 2: Submit Training Jobs
```bash
chmod +x train_lora.slurm train_qlora.slurm
sbatch train_lora.slurm
sbatch train_qlora.slurm
```

### Step 3: Monitor Jobs
```bash
# Check job status
squeue -u $USER

# View live output
tail -f outputs/logs/lora_*.out
tail -f outputs/logs/qlora_*.out

# View errors
tail -f outputs/logs/lora_*.err
tail -f outputs/logs/qlora_*.err
```

## Files Created

- **train_lora.slurm**: SLURM script for LoRA training
- **train_qlora.slurm**: SLURM script for QLoRA training
- **prepare_data.sh**: Script to prepare data splits (run once)

## Output Locations

- **Checkpoints**: `checkpoints/siglip_lora/` and `checkpoints/siglip_qlora/`
- **Logs**: `outputs/logs/lora_*.out` and `outputs/logs/qlora_*.out`
- **Errors**: `outputs/logs/lora_*.err` and `outputs/logs/qlora_*.err`

## Important Notes

1. **Partition Name**: Update `--partition=gpu` in both .slurm files if your cluster uses a different partition name. Check with `sinfo`.

2. **Time Limit**: Adjust `--time=48:00:00` based on your training needs.

3. **Container Path**: The scripts use `/home/shared/sif/csci-2025-Fall.sif`. Verify this path is correct on your cluster.

4. **Data**: The dataset will be automatically downloaded to `~/.cache/kagglehub/` when you run `prepare_data.sh`.

5. **Both Jobs**: You can submit both jobs at the same time - they will run independently.

## Troubleshooting

**Job fails immediately:**
- Check the error log: `cat outputs/logs/lora_*.err`
- Verify container path exists
- Check partition name is correct

**Training crashes:**
- Check GPU availability: `nvidia-smi`
- Verify data is accessible
- Check disk space: `df -h ~`

**Job stuck in queue:**
- Check cluster status: `sinfo`
- Verify your GPU quota: `sacctmgr show assoc user=$USER`

## After Training

Results will be in:
- `checkpoints/siglip_lora/` - LoRA model checkpoints
- `checkpoints/siglip_qlora/` - QLoRA model checkpoints

To compare results:
```bash
srun apptainer exec --nv /home/shared/sif/csci-2025-Fall.sif python main.py compare
```

