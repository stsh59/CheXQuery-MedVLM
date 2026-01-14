# Medical-SigLIP: Fine-Tuning with PEFT for Medical Image Understanding

**Built with PyTorch Lightning**

This project implements parameter-efficient fine-tuning (PEFT) of Medical-SigLIP using LoRA and QLoRA for medical image-text alignment, integrated with BioGPT for automated radiology report generation. All training and evaluation uses PyTorch Lightning for clean, scalable, and production-ready code.

## Project Overview

The project fine-tunes a vision-language model (SigLIP) on chest X-ray images and their corresponding radiology reports using parameter-efficient methods (LoRA and QLoRA). The fine-tuned encoder is then integrated with BioGPT to generate medical impressions from X-ray images.

### Key Features

- **PyTorch Lightning Architecture**: All code follows Lightning best practices with LightningModule and LightningDataModule
- **Parameter-Efficient Fine-Tuning**: Implements both LoRA and QLoRA for efficient adaptation of large models
- **Contrastive Learning**: Aligns X-ray images with medical text using contrastive loss
- **Medical Report Generation**: Generates clinical impressions using BioGPT
- **Comprehensive Evaluation**: BLEU, ROUGE, METEOR, and cosine similarity metrics
- **LoRA vs QLoRA Comparison**: Side-by-side comparison with automatic profiling
- **Automatic Multi-GPU**: Seamless distributed training support via Lightning
- **Built-in Callbacks**: ModelCheckpoint, EarlyStopping, LearningRateMonitor

## Dataset

**IU-Xray Dataset**
- 3,851 unique chest X-ray cases
- 7,470 images (Frontal and Lateral projections)
- Detailed radiology reports with findings and impressions
- Downloaded automatically via Kaggle Hub

## Installation

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

## Project Structure

```
visualTextHealth/
├── data/                   # Data loading (Lightning DataModules)
│   ├── dataset.py         # PyTorch datasets
│   ├── datamodule.py      # LightningDataModule
│   └── split.py           # Train/val/test splitting
├── models/                 # Model architectures (LightningModules)
│   ├── medical_siglip_lightning.py  # SigLIP LightningModule
│   ├── full_pipeline_lightning.py   # Full pipeline LightningModule
│   ├── peft_config.py     # PEFT configurations
│   ├── projection.py      # Projection layer
│   └── biogpt_generator.py # BioGPT integration
├── train/                  # Training scripts (Lightning Trainer)
│   ├── train_siglip_lightning.py    # SigLIP training
│   ├── train_full_lightning.py      # Full pipeline training
│   └── train_config.yaml  # Hyperparameters
├── eval/                   # Evaluation scripts
│   ├── metrics.py         # Evaluation metrics
│   ├── evaluate_lightning.py        # Model evaluation
│   └── qualitative_analysis_lightning.py # Visualizations
├── experiments/            # Comparison experiments
│   └── compare_lora_qlora_lightning.py # LoRA vs QLoRA
├── utils/                  # Utility functions
│   ├── config.py          # Global configuration
│   ├── logger.py          # Logging setup
│   ├── tokenizer_utils.py # Text preprocessing
│   └── checkpoint.py      # Model checkpointing
├── main.py                 # Main CLI (uses Lightning Trainer)
└── requirements.txt        # Dependencies (includes pytorch-lightning)
```

## Usage

### 1. Prepare Data

```bash
python main.py prepare_data
```

This creates train/val/test splits at the patient level to prevent data leakage.

### 2. Train SigLIP Encoder

**Train with LoRA:**
```bash
python main.py train_siglip --peft_method lora --num_epochs 10
```

**Train with QLoRA:**
```bash
python main.py train_siglip --peft_method qlora --num_epochs 10
```

### 3. Train Full Pipeline

```bash
python main.py train_full --siglip_checkpoint checkpoints/siglip_lora/best_model.pt
```

This trains the projection layer and fine-tunes BioGPT while keeping the SigLIP encoder frozen.

### 4. Evaluate Model

```bash
python main.py evaluate --checkpoint checkpoints/full_pipeline/best_model.pt --split test
```

### 5. Qualitative Analysis

```bash
python main.py qualitative --checkpoint checkpoints/full_pipeline/best_model.pt --num_samples 10
```

### 6. Compare LoRA vs QLoRA

```bash
python main.py compare
```

This trains both methods and compares:
- Training time
- Memory usage (RAM and GPU)
- Validation loss
- Generates comparison plots

## Configuration

Edit `train/train_config.yaml` to customize hyperparameters:

```yaml
# Data
batch_size: 16
projection_type: 'Frontal'  # 'Frontal', 'Lateral', or null

# Training
num_epochs: 10
learning_rate: 1e-4
warmup_steps: 500

# Model
peft_method: 'lora'  # 'lora' or 'qlora'
temperature: 0.07

# LoRA parameters are in utils/config.py
```

## Evaluation Metrics

The project computes the following metrics:

- **BLEU (1-4)**: N-gram overlap with reference reports
- **ROUGE (1, 2, L)**: Recall-oriented metrics
- **METEOR**: Considers synonyms and stemming
- **Cosine Similarity**: Embedding alignment between images and text

## Expected Results

Based on the project goals:

1. **Fine-tuned Medical-SigLIP**: Improved image-text alignment for chest X-rays
2. **Automated Report Generation**: Clear, context-aware medical impressions
3. **Efficiency Comparison**: QLoRA achieves comparable accuracy to LoRA with reduced memory and computation

## Hardware Requirements

**Minimum:**
- GPU: NVIDIA GPU with 8GB VRAM (for LoRA)
- RAM: 16GB
- Storage: 20GB

**Recommended:**
- GPU: NVIDIA GPU with 16GB+ VRAM
- RAM: 32GB
- Storage: 50GB

**For QLoRA:**
- Can run on GPUs with as little as 6GB VRAM due to 4-bit quantization

## Troubleshooting

### Out of Memory Error
- Reduce `batch_size` in config
- Use QLoRA instead of LoRA
- Enable gradient checkpointing
- Use mixed precision training (enabled by default)

### Slow Training
- Increase `batch_size` if memory allows
- Reduce `num_workers` in DataLoader
- Use faster storage (SSD) for dataset

### Dataset Not Found
- Ensure kagglehub is installed
- Check internet connection for dataset download
- Verify dataset path in `utils/config.py`

## Citation

If you use this code, please cite the relevant papers:

```bibtex
@article{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal={arXiv preprint arXiv:2106.09685},
  year={2021}
}

@article{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2305.14314},
  year={2023}
}
```

## License

This project is for educational and research purposes.

## Contact

For questions or issues, please open an issue on the repository.

