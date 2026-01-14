# CheXQuery-MedVLM: Complete Methodology & Architecture

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Architecture Overview](#3-architecture-overview)
4. [Component Details](#4-component-details)
5. [Data Pipeline](#5-data-pipeline)
6. [Training Strategy](#6-training-strategy)
7. [Loss Functions](#7-loss-functions)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Novel Contributions](#9-novel-contributions)
10. [Technical Specifications](#10-technical-specifications)

---

## 1. Executive Summary

**CheXQuery-MedVLM** is a novel Vision-Language Model designed for automatic chest X-ray report generation. The model generates structured radiology reports in the format:

```
Findings: [detailed observations] | Impression: [clinical conclusion]
```

### Key Innovations:
1. **CheXbert-Initialized Condition Queries**: 14 learnable queries initialized from BioBERT embeddings
2. **Anatomical Region Queries**: 6 learnable queries for spatial grounding
3. **Gated Fusion Module**: Adaptive balance between global and local visual information
4. **Multi-task Learning**: Joint generation and CheXbert classification

---

## 2. Problem Statement

### 2.1 Why Automatic Report Generation?
- Radiologists face increasing workload
- Report writing is time-consuming
- Automated systems can assist and reduce errors

### 2.2 Challenges with Existing Approaches

| Challenge | Description |
|-----------|-------------|
| Generic Features | Standard VLMs don't understand medical image specifics |
| Single Token | Many models compress entire image to one token, losing spatial info |
| No Clinical Knowledge | Models don't know about medical conditions |
| Fluency vs Accuracy | Models optimize for fluent text, not clinical correctness |

### 2.3 Our Solution
Query-based architecture with:
- Condition-aware queries that look for specific pathologies
- Anatomical queries that focus on specific body regions
- Auxiliary classification for clinical accuracy supervision

---

## 3. Architecture Overview

### 3.1 High-Level Flow

```
┌─────────────────┐
│  Chest X-ray    │
│  [3, 384, 384]  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│       SigLIP Vision Encoder (LoRA)       │
│  ┌─────────┐    ┌──────────────────┐    │
│  │   CLS   │    │  Patch Tokens    │    │
│  │ [1,768] │    │   [576, 768]     │    │
│  └─────────┘    └──────────────────┘    │
└────────┬─────────────────┬──────────────┘
         │                 │
         │    ┌────────────┴────────────┐
         │    │                         │
         │    ▼                         ▼
         │  ┌─────────────┐  ┌─────────────────┐
         │  │  Condition  │  │   Anatomical    │
         │  │   Queries   │  │    Queries      │
         │  │  [14, 768]  │  │    [6, 768]     │
         │  │ (BioBERT)   │  │   (Xavier)      │
         │  └──────┬──────┘  └────────┬────────┘
         │         │                  │
         │         └────────┬─────────┘
         │                  │ Concatenate
         │                  ▼
         │         ┌────────────────┐
         │         │ Combined Queries│
         │         │   [20, 768]    │
         │         └───────┬────────┘
         │                 │
         │    ┌────────────┴─────────────┐
         │    │   Cross-Attention (2L)    │
         │    │  Q: Queries K,V: Patches  │
         │    └────────────┬─────────────┘
         │                 │
         │         ┌───────┴──────────┐
         │         │ Attended Queries │
         │         │    [20, 768]     │
         │         └───────┬──────────┘
         │                 │
         │    ┌────────────┼────────────────┐
         │    │            │                │
         │    │     ┌──────┴──────┐        │
         │    │     │ Split: 14+6 │        │
         │    │     └──────┬──────┘        │
         │    │            │               │
         │    │     ┌──────┴──────┐        │
         │    │     │  Auxiliary  │        │
         │    │     │    Head     │◄───────┤ CheXbert Labels
         │    │     │  [B, 14]    │        │
         │    │     └─────────────┘        │
         │    │                            │
         └────┼────────────┬───────────────┘
              │            │
              ▼            ▼
         ┌────────────────────────────┐
         │      Gated Fusion          │
         │  ┌──────┐  ┌────────────┐  │
         │  │ Gate │  │  Pooling   │  │
         │  │ σ()  │  │  20→10     │  │
         │  └──────┘  └────────────┘  │
         │      Output: [11, 768]     │
         │   (1 CLS + 10 pooled)      │
         └────────────┬───────────────┘
                      │
                      ▼
         ┌─────────────────────────────┐
         │    Flan-T5 Decoder (LoRA)   │
         │  Visual tokens as encoder   │
         │  Cross-attention → Generate │
         └────────────┬────────────────┘
                      │
                      ▼
         ┌─────────────────────────────┐
         │  Findings: ... | Impression │
         └─────────────────────────────┘
```

### 3.2 Dimension Summary

| Stage | Tensor Shape | Description |
|-------|--------------|-------------|
| Input Image | `[B, 3, 384, 384]` | RGB, 384×384 pixels |
| After Patching | `[B, 576, 768]` | 576 patches × 768 dim |
| CLS Token | `[B, 768]` | Global representation |
| Condition Queries | `[B, 14, 768]` | 14 conditions |
| Anatomical Queries | `[B, 6, 768]` | 6 regions |
| Combined Queries | `[B, 20, 768]` | All queries |
| After Cross-Attention | `[B, 20, 768]` | Attended queries |
| After Gated Fusion | `[B, 11, 768]` | Final visual tokens |
| Generated Text | Variable | Max 512 tokens |

---

## 4. Component Details

### 4.1 Vision Encoder (SigLIP)

**File:** `models/vision_encoder.py`

#### What is SigLIP?
SigLIP (Sigmoid Loss for Image-text Pre-training) is Google's vision encoder pre-trained on image-text pairs. We use `google/siglip-base-patch16-384`.

#### How it Works:
1. **Patch Embedding**: Image (384×384) → 576 patches (16×16 each)
2. **Position Embedding**: Add positional information
3. **Transformer**: 12 layers of self-attention
4. **Output**: CLS token + 576 patch tokens

#### LoRA Configuration:
```yaml
rank: 8
alpha: 16
dropout: 0.05
target_modules: ["q_proj", "v_proj"]
```

#### Why LoRA?
- Full fine-tuning: 86M parameters
- With LoRA: ~300K trainable parameters
- Maintains pre-trained knowledge while adapting

---

### 4.2 Condition Query Module

**File:** `models/condition_queries.py`

#### Purpose:
Learn to detect 14 CheXbert medical conditions in images.

#### The 14 Conditions:
| # | Condition | Description |
|---|-----------|-------------|
| 0 | No Finding | Normal study |
| 1 | Enlarged Cardiomediastinum | Widened mediastinum |
| 2 | Cardiomegaly | Enlarged heart |
| 3 | Lung Opacity | Unclear lung regions |
| 4 | Lung Lesion | Mass or nodule |
| 5 | Edema | Fluid in lungs |
| 6 | Consolidation | Filled air spaces |
| 7 | Pneumonia | Lung infection |
| 8 | Atelectasis | Collapsed lung |
| 9 | Pneumothorax | Air in pleural space |
| 10 | Pleural Effusion | Fluid around lung |
| 11 | Pleural Other | Other pleural issues |
| 12 | Fracture | Broken bone |
| 13 | Support Devices | Tubes, lines, etc. |

#### Initialization Process:
```python
# For each condition (e.g., "pneumonia"):
1. tokenizer = BioBERT.tokenizer("pneumonia")
2. embedding = BioBERT.model(tokenizer).CLS
3. query[i] = embedding  # [768]

# Result: [14, 768] initialized queries
```

#### Why BioBERT Initialization?
- Random init: Queries start with no medical knowledge
- BioBERT init: Queries already "understand" condition meanings
- Faster convergence, better performance

---

### 4.3 Anatomical Query Module

**File:** `models/anatomical_queries.py`

#### Purpose:
Focus attention on specific anatomical regions of chest X-rays.

#### The 6 Regions:
| # | Region | Clinical Focus |
|---|--------|----------------|
| 0 | Cardiac | Heart size, shape, silhouette |
| 1 | Left Lung | Left lung field abnormalities |
| 2 | Right Lung | Right lung field abnormalities |
| 3 | Mediastinum | Central chest structures |
| 4 | Diaphragm | Diaphragm, costophrenic angles |
| 5 | Spine | Vertebral column, bones |

#### Initialization:
Xavier uniform (no pre-trained embeddings for spatial regions)

```python
nn.init.xavier_uniform_(self.queries)  # [6, 768]
```

---

### 4.4 Cross-Attention Module

**File:** `models/cross_attention.py`

#### Purpose:
Allow queries to "look at" relevant image patches.

#### Architecture:
```
CrossAttentionModule
├── Layer 1
│   ├── MultiheadAttention(768, 8 heads)
│   ├── LayerNorm
│   ├── FFN(768 → 3072 → 768)
│   └── LayerNorm
└── Layer 2
    ├── MultiheadAttention(768, 8 heads)
    ├── LayerNorm
    ├── FFN(768 → 3072 → 768)
    └── LayerNorm
```

#### How Cross-Attention Works:
```
Input:
  Q (Queries): [B, 20, 768]    # What we're looking for
  K, V (Patches): [B, 576, 768] # Where we're looking

Compute:
  Attention = softmax(Q @ K.T / √768)  # [B, 20, 576]
  Output = Attention @ V               # [B, 20, 768]

Result:
  Each query now contains information from relevant patches
  Attention weights show which patches each query focused on
```

---

### 4.5 Gated Fusion Module

**File:** `models/gated_fusion.py`

#### Purpose:
Intelligently combine global (CLS) and local (query) information.

#### Gate Mechanism:
```python
# Compute gate value
query_mean = queries.mean(dim=1)  # [B, 768]
gate_input = torch.cat([cls, query_mean], dim=-1)  # [B, 1536]
gate = sigmoid(MLP(gate_input))  # [B, 1] ∈ [0, 1]

# Apply gating
cls_gated = gate * cls
queries_weighted = (1 - gate) * queries
```

#### Query Pooling:
```python
# Compress 20 queries to 10
pool_queries = learnable([10, 768])
pooled = CrossAttention(
    Q=pool_queries,    # [B, 10, 768]
    K=queries,         # [B, 20, 768]
    V=queries
)  # Output: [B, 10, 768]

# Final visual tokens
visual_tokens = concat([cls_gated, pooled])  # [B, 11, 768]
```

---

### 4.6 Text Decoder (Flan-T5)

**File:** `models/text_decoder.py`

#### Why Flan-T5?
- **Encoder-Decoder**: Perfect for vision-to-text
- **Cross-Attention**: Built-in attention to visual tokens
- **Instruction-Tuned**: Better at structured output
- **Efficient**: Base model fits on single GPU

#### How We Use It:
```python
# Standard T5:
encoder(text) → hidden → decoder → output

# Our usage:
visual_tokens → used AS encoder hidden → decoder → report

# We bypass T5's encoder entirely!
# Our 11 visual tokens act as encoder output
```

#### Prompt-Guided Decoding (Enabled)
We prepend a short prompt to the decoder to enforce the exact output structure:

- **Prompt source:** `configs/data_config.yaml` → `prompt_template` (same as `get_prompt_template()`)
- **Training:** prompt + target text → tokenized → decoder_input_ids/labels
- **Inference:** prompt tokenized → passed as `decoder_input_ids` to `generate()`

This keeps the decoder aligned to the required format:
```
Findings: ... | Impression: ...
```

#### Decoding Strategy (Deterministic Beam Search)
- **Prompt prefix:** Ensures the decoder starts from the required format (`Findings: ... | Impression: ...`).
- **Search:** Beam search (deterministic, no sampling)
  - `num_beams = 4`: Keep the top 4 partial hypotheses at each step (explore 4 candidate reports in parallel).
  - `no_repeat_ngram_size = 3`: Prevent repeating any trigram (reduces repetition artifacts).
  - `length_penalty = 1.0`: Neutral; doesn’t favor longer or shorter outputs.
  - `early_stopping = True`: Stop when all beams finish; prevents overly long outputs.
  - `min_length = 20`, `max_length = 512`: Bounds for report length.

**Why this strategy?**
- Clinical reports value consistency over randomness → deterministic beam search is preferred to sampling.
- Prompt + beam search enforces the structured format and reduces format drift.
- Trigram blocking helps avoid repetitive phrasing common in medical text generation.

#### LoRA Configuration:
```yaml
rank: 16
alpha: 32
dropout: 0.05
target_modules: ["q", "k", "v", "o"]
```

---

### 4.7 Auxiliary Classification Head

**File:** `models/auxiliary_head.py`

#### Purpose:
Predict which of 14 conditions are present (multi-label classification).

#### Architecture:
```
For each condition query [768]:
├── Linear(768 → 384)
├── GELU
├── Dropout(0.2)
├── Linear(384 → 1)
└── Output: logit

All 14 queries → [B, 14] logits
```

#### Why Auxiliary Classification?
1. Forces queries to learn condition-relevant features
2. Provides direct clinical supervision
3. Enables CheXbert F1 evaluation during training
4. Acts as regularization

---

## 5. Data Pipeline

### 5.1 Dataset

**Dataset:** Indiana University Chest X-ray (IU X-ray)
**Source:** Kaggle (raddar/chest-xrays-indiana-university)

| Property | Value |
|----------|-------|
| Total Images | ~7,470 |
| Total Reports | ~3,955 |
| Projections | Frontal, Lateral |
| We Use | Frontal only |

### 5.2 Data Splits

```
Strategy: Patient-level split (prevents data leakage)

Train:      70%
Validation: 15%
Test:       15%
```

**Why Patient-Level?**
- Same patient may have multiple X-rays
- If same patient in train/test → model memorizes patient patterns
- Patient-level ensures true generalization

### 5.3 Image Preprocessing

> ⚠️ **IMPORTANT**: Medical image augmentations must preserve anatomical validity!

**Training (Medically-Valid Augmentations):**
```python
transforms.Compose([
    Resize(384, 384),
    # NO RandomHorizontalFlip! Heart is on LEFT side - flipping is anatomically invalid
    RandomRotation(degrees=5),              # Conservative (patient positioning variance)
    RandomAffine(translate=0.03, scale=(0.97, 1.03)),  # Minimal
    ColorJitter(brightness=0.05, contrast=0.05),       # Conservative
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

**Why NO Horizontal Flip?**
- Heart is anatomically on the **LEFT** side in normal patients
- Horizontal flip would create "dextrocardia" (heart on right) - occurs in only 0.01% of population
- Would corrupt anatomical query learning (cardiac query would attend to wrong region)
- Generated reports would have incorrect laterality

**Validation/Test:**
```python
transforms.Compose([
    Resize(384, 384),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

### 5.4 Text Preprocessing

**Input (from CSV):**
```
findings: "The heart is normal in size..."
impression: "No acute cardiopulmonary abnormality."
```

**Output (structured):**
```
"Findings: The heart is normal in size... | Impression: No acute cardiopulmonary abnormality."
```

**Cleaning Steps:**
1. Strip whitespace
2. **Remove sentences containing PHI markers (XXXX)** - entire sentences with de-identified data are removed to prevent model from generating placeholder text
3. Normalize whitespace
4. Fix sentence spacing

> **Why remove instead of replace?** If we replaced XXXX with `[REDACTED]`, the model might learn to generate `[REDACTED]` in outputs. By removing entire PHI-containing sentences, we ensure clean training data and clean generation.

---

## 6. Training Strategy

### 6.1 Three-Phase Training (Staged for Stability)

Training everything at once is unstable—randomly initialized queries can disrupt pre-trained components. We use a **staged training schedule**: first align queries with encoders/decoder frozen, then fine-tune end-to-end. This is a pragmatic stabilization step, not a separate curriculum-learning algorithm.

> **Why we do NOT fine-tune SigLIP in Phase 1 (the deeper rationale)**
> 
> In theory, pre-finetuning SigLIP on chest X-rays could make query–patch alignment easier. In practice, doing so before Phase 1 often makes the system worse:
> 
> - **Gradient dominance:** If SigLIP is trainable in Phase 1, the huge vision backbone (~86M params) absorbs the learning signal; the tiny queries (~15K params) don’t learn sharp attention.
> - **Shortcut learning:** SigLIP starts encoding disease labels directly; queries become unnecessary and attention becomes diffuse—losing explicit grounding.
> - **Query collapse:** Queries degenerate into bias vectors instead of selectors; cross-attention becomes soft pooling, undermining interpretability.
> 
> We intentionally keep SigLIP frozen in Phase 1 so alignment is “hard,” forcing queries to learn true discrimination and sharp attention. Then, in Phase 2, we gently adapt SigLIP via LoRA:
> 
> - **Phase 1:** Fix feature space; train selectors (queries); learn grounding with auxiliary CheXbert loss.
> - **Phase 2:** Slightly bend the feature space (LoRA) to refine subtle cues while preserving query-based reasoning and stability.
> 
> **Analogy:** Train the “detectives” (queries) to investigate evidence even when the base features aren’t specialized; only later allow the feature extractor to adapt, once the detectives already know how to focus on the right evidence.

#### Phase 1: Query Alignment (5 epochs)

**Goal:** Teach queries to attend to relevant patches

| Component | Status |
|-----------|--------|
| Vision Encoder | ❄️ Frozen |
| Condition Queries | 🔥 Training |
| Anatomical Queries | 🔥 Training |
| Cross-Attention | 🔥 Training |
| Gated Fusion | 🔥 Training |
| Auxiliary Head | 🔥 Training |
| Text Decoder | ❄️ Frozen |

**Hyperparameters:**
```yaml
batch_size: 16 (effective: 32)
learning_rate: 1e-4
warmup_steps: 200
loss_weights:
  generation: 1.0
  auxiliary: 0.5
```

#### Phase 2: End-to-End Fine-tuning (15 epochs)

**Goal:** Fine-tune entire model with LoRA

| Component | Status |
|-----------|--------|
| Vision Encoder | 🔶 LoRA only |
| Condition Queries | 🔥 Training |
| Anatomical Queries | 🔥 Training |
| Cross-Attention | 🔥 Training |
| Gated Fusion | 🔥 Training |
| Auxiliary Head | 🔥 Training |
| Text Decoder | 🔶 LoRA only |

**Hyperparameters:**
```yaml
batch_size: 8 (effective: 32)
learning_rate: 5e-5
warmup_steps: 500
label_smoothing: 0.1
loss_weights:
  generation: 1.0
  auxiliary: 0.3
```

#### Phase 3: Generation Fine-tuning (Optional, 5 epochs)

**Goal:** Polish generation quality

**Changes:**
- Auxiliary weight: 0.0 (disabled)
- Vision encoder: Fully frozen
- Learning rate: 2e-5

---

## 7. Loss Functions

### 7.1 Total Loss

```
L_total = λ_gen × L_generation + λ_aux × L_auxiliary
```

### 7.2 Generation Loss

Standard cross-entropy for language modeling:

```
L_generation = -Σ log P(token_t | token_1...t-1, visual_tokens)
```

- Computed by T5 forward pass
- Label smoothing: 0.1 in Phase 2+

### 7.3 Auxiliary Loss

Binary cross-entropy for multi-label classification:

```
L_auxiliary = -Σ [y_i × log(σ(z_i)) + (1-y_i) × log(1-σ(z_i))]
```

- Input: Condition query embeddings [B, 14, 768]
- Output: Logits [B, 14]
- Targets: Binary labels from CheXbert

### 7.4 Loss Weight Schedule

| Phase | λ_gen | λ_aux | Rationale |
|-------|-------|-------|-----------|
| Phase 1 | 1.0 | 0.5 | Strong auxiliary signal |
| Phase 2 | 1.0 | 0.3 | Balance both tasks |
| Phase 3 | 1.0 | 0.0 | Focus on generation |

---

## 8. Evaluation Metrics

### 8.1 Text Similarity Metrics

#### BLEU (Bilingual Evaluation Understudy)
- Measures n-gram precision
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- Higher = better, range [0, 1]

#### ROUGE (Recall-Oriented Understudy)
- Measures n-gram recall
- ROUGE-1, ROUGE-2, ROUGE-L (longest common subsequence)
- Higher = better, range [0, 1]

#### METEOR
- Considers synonyms and stemming
- Better correlation with human judgment
- Higher = better, range [0, 1]

### 8.2 Semantic Similarity

#### BERTScore
- Uses BERT embeddings for semantic similarity
- Model: microsoft/deberta-xlarge-mnli
- Outputs: Precision, Recall, F1

### 8.3 Clinical Accuracy

#### CheXbert F1 (Most Important!)
```
Process:
1. Run CheXbert on generated report → predicted labels [14]
2. Run CheXbert on reference report → ground truth labels [14]
3. Compare predictions with ground truth

Outputs:
- Precision: Of predicted conditions, how many correct?
- Recall: Of actual conditions, how many found?
- F1: Harmonic mean
```

### 8.4 Text Normalization

Before computing metrics (comprehensive normalization):
1. Handle empty/None values gracefully
2. Lowercase all text (case-insensitive comparison)
3. Remove structural prefixes ("Findings:", "Impression:", "|")
4. Normalize unicode characters (smart quotes → straight quotes, en-dash → hyphen)
5. Remove punctuation (so "normal." and "normal" match as same token)
6. Normalize whitespace
7. Tokenize with NLTK word_tokenize
8. Skip invalid pairs (empty reference or hypothesis)

> **Why remove punctuation?** Punctuation attached to words creates different tokens ("normal." vs "normal"), unfairly penalizing BLEU/ROUGE scores for semantically identical content.

---

## 9. Novel Contributions

### 9.1 CheXbert-Initialized Condition Queries

**What's New:**
First to initialize visual queries from medical NLP model embeddings.

**How It Helps:**
- Bridges vision and medical language
- Queries start with semantic understanding
- Faster convergence

### 9.2 Dual Query System

**What's New:**
Combination of condition queries (what) + anatomical queries (where).

**How It Helps:**
- Semantic grounding (conditions)
- Spatial grounding (anatomy)
- Comprehensive image understanding

### 9.3 Adaptive Gated Fusion

**What's New:**
Learned gate to balance global vs local information.

**How It Helps:**
- Normal images: More global info
- Abnormal images: More local detail
- Per-image adaptation

### 9.4 Multi-task Learning

**What's New:**
Joint optimization of generation + classification.

**How It Helps:**
- Direct clinical supervision
- Better query learning
- Regularization effect

### 9.5 Comparison with Existing Methods

| Method | Visual Tokens | Query Init | Clinical Supervision |
|--------|---------------|------------|---------------------|
| R2Gen | Single CLS | N/A | No |
| CMN | Memory | Random | No |
| BLIP-2 | 32 learned | Random | No |
| **Ours** | **11 (1+10)** | **BioBERT** | **Yes** |

---

## 10. Technical Specifications

### 10.1 Model Size

| Component | Total Params | Trainable (LoRA) |
|-----------|-------------|------------------|
| SigLIP | ~86M | ~300K |
| Condition Queries | 10.7K | 10.7K |
| Anatomical Queries | 4.6K | 4.6K |
| Cross-Attention | ~12M | ~12M |
| Gated Fusion | ~2M | ~2M |
| Auxiliary Head | ~300K | ~300K |
| Flan-T5 | ~248M | ~1.5M |
| **Total** | **~350M** | **~16M (4.5%)** |

### 10.2 Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 16GB | 24GB (A5000) |
| RAM | 32GB | 64GB |
| Storage | 20GB | 50GB |
| Training Time | ~12h (A5000) | ~8h (A100) |

### 10.3 Expected Performance

| Metric | Target |
|--------|--------|
| BLEU-1 | ~0.45 |
| BLEU-4 | ~0.22 |
| ROUGE-L | ~0.48 |
| METEOR | ~0.40 |
| BERTScore F1 | ~0.91 |
| CheXbert F1 | ~0.52 |

---

## 11. Training Guide & Phase 3 Notes

### Step-by-Step Training
```
python main.py prepare \
  --splits-file outputs/splits/data_splits.json

python main.py train --phase 1 \
  --model-config configs/model_config.yaml \
  --train-config configs/train_config.yaml \
  --data-config configs/data_config.yaml \
  --checkpoint-dir outputs/checkpoints

# Use best Phase 1 checkpoint
PHASE1_CKPT=$(ls -t outputs/checkpoints/phase1/*.ckpt | head -1)

python main.py train --phase 2 \
  --model-config configs/model_config.yaml \
  --train-config configs/train_config.yaml \
  --data-config configs/data_config.yaml \
  --checkpoint-dir outputs/checkpoints \
  --resume "$PHASE1_CKPT"

BEST_CKPT=$(ls -t outputs/checkpoints/phase2/*.ckpt | head -1)

python main.py evaluate \
  --checkpoint "$BEST_CKPT" \
  --model-config configs/model_config.yaml \
  --train-config configs/train_config.yaml \
  --data-config configs/data_config.yaml \
  --output-dir outputs/evaluation \
  --batch-size 16 \
  --num-beams 4 \
  --split test

python main.py generate \
  --checkpoint "$BEST_CKPT" \
  --model-config configs/model_config.yaml \
  --train-config configs/train_config.yaml \
  --data-config configs/data_config.yaml \
  --images path/to/image_or_dir \
  --max-length 512 \
  --num-beams 4


python main.py visualize \
  --checkpoint "$BEST_CKPT" \
  --model-config configs/model_config.yaml \
  --train-config configs/train_config.yaml \
  --images path/to/image.png \
  --output-dir outputs/visualizations
```

### Why Phase 3 is Optional
- The architecture is fully trained after Phase 1 + Phase 2.
- Phase 3 adds no new components; it just polishes generation (fluency/format) with vision frozen and aux loss off.
- Skip if Phase 2 metrics are satisfactory or data is small; run if you want a slight fluency boost without touching the vision backbone.

---

## References

1. SigLIP: Sigmoid Loss for Language Image Pre-Training (Zhai et al., 2023)
2. BioBERT: A pre-trained biomedical language representation model (Lee et al., 2020)
3. CheXbert: Combining Automatic Labelers and Expert Annotations (Smit et al., 2020)
4. Flan-T5: Scaling Instruction-Finetuned Language Models (Chung et al., 2022)
5. LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)

---

*Document generated for CheXQuery-MedVLM architecture*
