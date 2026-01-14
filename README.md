# CheXQuery-MedVLM

**Anatomical Region-Guided Medical Vision-Language Model for Chest X-ray Report Generation**

A novel architecture featuring CheXbert-initialized condition queries and anatomical region queries for high-quality, structured medical report generation.

## 🔬 Architecture Highlights

### Novel Contributions
1. **CheXbert-Initialized Condition Queries**: 14 learnable queries initialized from BioBERT embeddings of CheXbert conditions, providing pathology-aware visual attention
2. **Anatomical Region Queries**: 6 learnable queries for spatial grounding of cardiac, lung, mediastinum, diaphragm, and spine regions
3. **Gated Fusion with Query Pooling**: Adaptive balance between global (CLS) and local (query) visual information
4. **Multi-task Learning**: Joint generation and auxiliary CheXbert classification for clinical accuracy

### Architecture Overview
```
Image → SigLIP (LoRA) → Patch Tokens (576)
                              ↓
    [Condition Queries (14)] + [Anatomical Queries (6)]
                              ↓
                    Cross-Attention (2 layers)
                              ↓
                    Gated Fusion + Query Pooling
                              ↓
                    Visual Tokens (11 = 1 CLS + 10 pooled)
                              ↓
                    Flan-T5 Decoder (LoRA)
                              ↓
        Findings: [text] | Impression: [text]
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd chexquery-medvlm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Prepare Data

```bash
# Download dataset and create splits
python main.py prepare --splits-file outputs/splits/data_splits.json
```

### Training

```bash
# Train all phases sequentially
python main.py train --all-phases

# Or train specific phase
python main.py train --phase 1  # Query Alignment
python main.py train --phase 2 --resume outputs/checkpoints/phase1/best.ckpt  # End-to-End
```

### Evaluation

```bash
python main.py evaluate --checkpoint outputs/checkpoints/phase2/best.ckpt
```

### Generate Reports

```bash
python main.py generate --checkpoint outputs/checkpoints/phase2/best.ckpt --images path/to/image.png
```

### Visualize Attention

```bash
python main.py visualize --checkpoint outputs/checkpoints/phase2/best.ckpt --images path/to/image.png
```

## 📁 Project Structure

```
chexquery-medvlm/
├── configs/                    # Configuration files
│   ├── model_config.yaml       # Model architecture
│   ├── train_config.yaml       # Training settings
│   ├── data_config.yaml        # Data settings
│   └── eval_config.yaml        # Evaluation settings
├── data/                       # Data loading
│   ├── dataset.py              # Dataset class
│   ├── datamodule.py           # Lightning DataModule
│   ├── preprocessing.py        # Text preprocessing
│   └── augmentations.py        # Image augmentations
├── models/                     # Model components
│   ├── vision_encoder.py       # SigLIP encoder
│   ├── condition_queries.py    # CheXbert-initialized queries
│   ├── anatomical_queries.py   # Anatomical region queries
│   ├── cross_attention.py      # Cross-attention module
│   ├── gated_fusion.py         # Gated fusion + pooling
│   ├── text_decoder.py         # Flan-T5 decoder
│   ├── auxiliary_head.py       # Classification head
│   └── chexquery_medvlm.py     # Full model integration
├── training/                   # Training pipeline
│   ├── lightning_module.py     # PyTorch Lightning module
│   └── trainer.py              # Training orchestration
├── evaluation/                 # Evaluation
│   ├── metrics.py              # BLEU, ROUGE, METEOR, BERTScore, CheXbert
│   └── evaluate.py             # Evaluation script
├── visualization/              # Interpretability
│   └── attention_viz.py        # Attention visualization
├── scripts/                    # SLURM scripts
│   ├── train_phase1.slurm
│   ├── train_phase2.slurm
│   ├── train_all.slurm
│   └── evaluate.slurm
├── outputs/                    # Results
│   ├── checkpoints/
│   ├── logs/
│   ├── evaluation/
│   └── visualizations/
├── main.py                     # CLI entry point
├── requirements.txt
└── README.md
```

## ⚙️ Configuration

### Model Configuration (`configs/model_config.yaml`)

Key settings:
- `vision_encoder.model_name`: SigLIP model variant
- `condition_queries.num_queries`: Number of condition queries (14)
- `anatomical_queries.num_queries`: Number of anatomical queries (6)
- `cross_attention.num_layers`: Cross-attention layers (2)
- `gated_fusion.num_pool_queries`: Pooled query tokens (10)
- `text_decoder.model_name`: Flan-T5 variant

### Training Configuration (`configs/train_config.yaml`)

Three-phase training:
1. **Phase 1 (Query Alignment)**: Freeze vision encoder and decoder, train queries + cross-attention
2. **Phase 2 (End-to-End)**: Fine-tune all with LoRA, joint loss
3. **Phase 3 (Generation)**: Optional, focus on generation quality

## 📊 Expected Results

| Metric | Target | Description |
|--------|--------|-------------|
| BLEU-1 | ~0.45 | Unigram overlap |
| BLEU-4 | ~0.22 | 4-gram overlap |
| ROUGE-L | ~0.48 | Longest common subsequence |
| METEOR | ~0.40 | Semantic matching |
| BERTScore F1 | ~0.91 | Contextual similarity |
| CheXbert F1 | ~0.52 | Clinical accuracy |

## 🔧 Hardware Requirements

- **GPU**: NVIDIA A5000 (24GB) or equivalent
- **Memory**: 64GB RAM recommended
- **Storage**: ~50GB for data and checkpoints

## 📚 Citation

If you use this code, please cite:

```bibtex
@article{chexquery2024,
  title={CheXQuery-MedVLM: Anatomical Region-Guided Vision-Language Model for Chest X-ray Report Generation},
  author={Your Name},
  year={2024}
}
```

## 📄 License

MIT License - see LICENSE file for details.
