# 🤟 Continuous Sign Language Translation (CSLT)
### Academic Project — Deep Learning Module | Data Engineering

> **Automatic translation of American Sign Language (ASL) into English text**  
> from raw videos of the **How2Sign** dataset, using Transfer Learning from **PHOENIX-2014T**.

---

## 👥 Team

| Role | Name | Major | School |
|------|------|-------|--------|
| 🎓 Student | **Amina Dourdi** | Data Engineering | ENSAH |
| 🎓 Student | **Firdawss El Haddouchi** | Data Engineering | ENSAH |
| 🎓 Student | **Oumaima El Ghalbouni** | Data Engineering | ENSAH |

> 🏫 **ENSAH** — National School of Applied Sciences of Al Hoceima  
> 📅 **Academic Year** : 2025 – 2026  
> 📌 **Module** : Deep Learning

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Full Architecture](#-full-architecture)
3. [Datasets Used](#-datasets-used)
4. [Tech Stack](#-tech-stack)
5. [Project Structure](#-project-structure)
6. [Phase 1 — Data Preprocessing](#-phase-1--data-preprocessing)
7. [Phase 2 — Sanity Check](#-phase-2--sanity-check)
8. [Phase 3 — Model Fine-Tuning](#-phase-3--model-fine-tuning)
9. [Phase 4 — Evaluation](#-phase-4--evaluation)
10. [Installation & Usage](#-installation--usage)
11. [Expected Results](#-expected-results)

---

## 🎯 Project Overview

This project implements an end-to-end **Continuous Sign Language Translation (CSLT)** system. The goal is to take a video of a person signing in **ASL (American Sign Language)** and automatically generate the **corresponding English sentence**.

### Main Constraint
> ❌ No manual assembly of encoders and decoders from different repositories.  
> ✅ Use a **pre-trained End-to-End model** (SignJoey on PHOENIX-2014T) adapted through **Fine-Tuning** to How2Sign.

### Key Technical Choice: Key-points Instead of Raw Videos
Instead of feeding raw video frames (very heavy) into a CNN, we use **Google MediaPipe Holistic** to extract **skeleton key-points** (hands, body, face) from each frame. This allows us to:
- ✅ Reduce storage from ~50 GB to ~500 MB
- ✅ Train on a regular CPU / free Google Colab
- ✅ Deploy in real-time on a web platform (no GPU required)
- ✅ Make the model insensitive to clothing color or background

---

## 🏗️ Full Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      FULL PROJECT PIPELINE                       │
└─────────────────────────────────────────────────────────────────┘

  [INPUT]  Raw video .mp4 (How2Sign — ASL)
      │
      ▼
  ┌─────────────────────────────────────┐
  │  PHASE 1 : DATA PREPROCESSING       │  (preprocessing.py)
  │                                     │
  │  1. Video cleaning & validation     │
  │  2. MediaPipe Holistic              │  ← REPLACES the CNN
  │     → 543 keypoints × 3 coords     │
  │     → Format [T, 1629]             │
  │  3. Z-score normalization           │
  │  4. Padding/Truncation → [200,1629] │
  │  5. Save as .npy files              │
  │  6. Text tokenization               │
  │     → 15,000-word vocabulary       │
  └─────────────────────────────────────┘
      │
      ▼  .npy files + token indices
  ┌─────────────────────────────────────┐
  │  PHASE 2 : MODEL (Fine-Tuning)      │  (fine_tune_cslt.py)
  │                                     │
  │  ┌─────────────────────────────┐    │
  │  │ Keypoint Embedding Layer    │    │  🆕 New layer
  │  │ Linear(1629 → 512)          │    │  Trained from scratch
  │  └────────────┬────────────────┘    │
  │               ▼                     │
  │  ┌─────────────────────────────┐    │
  │  │ Transformer Encoder         │    │  ❄️ FROZEN
  │  │ (SignJoey — PHOENIX-2014T)  │    │  Pre-trained weights
  │  │ 6 layers, 8 heads           │    │  kept intact
  │  └────────────┬────────────────┘    │
  │               ▼                     │
  │  ┌─────────────────────────────┐    │
  │  │ Transformer Decoder         │    │  🔥 FINE-TUNED
  │  │ (SignJoey — PHOENIX-2014T)  │    │  lr = 1e-5
  │  │ Cross-Attention             │    │
  │  └────────────┬────────────────┘    │
  │               ▼                     │
  │  ┌─────────────────────────────┐    │
  │  │ Vocabulary Classifier       │    │  🆕 REPLACED
  │  │ Linear(512 → 15,000)        │    │  How2Sign vocabulary
  │  └─────────────────────────────┘    │
  └─────────────────────────────────────┘
      │
      ▼
  [OUTPUT]  English sentence: "The woman is walking to school"
```

### Legend
| Symbol | Meaning |
|--------|---------|
| ❄️ FROZEN | Pre-trained weights from PHOENIX-2014T, never modified |
| 🔥 FINE-TUNED | Pre-trained weights updated slowly (lr = 1e-5) |
| 🆕 NEW | Randomly initialized layer, fully trained on How2Sign |

---

## 📦 Datasets Used

### 1. How2Sign (Main Dataset — Fine-Tuning)
| Property | Value |
|----------|-------|
| Language | ASL — American Sign Language |
| Content | ~35,000 signed sentences |
| Video storage | ~50 GB |
| Annotation format | `.csv` (English sentences) |
| Link | [how2sign.github.io](https://how2sign.github.io) |

### 2. PHOENIX-2014T (Pre-trained Model Dataset)
| Property | Value |
|----------|-------|
| Language | DGS — Deutsche Gebärdensprache (German Sign Language) |
| Content | ~8,000 sentences (weather forecasts) |
| Used for | Pre-training SignJoey |
| Link | [phoenix.ira.uka.de](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/) |

---

## 🛠️ Tech Stack

| Domain | Technology | Role |
|--------|-----------|------|
| Deep Learning | **PyTorch** | Main framework |
| Architecture | **SignJoey** (`neccam/slt`) | Pre-trained Seq2Seq model |
| Keypoint Extraction | **Google MediaPipe Holistic** | Replaces the CNN |
| Vision | **OpenCV** | Reading videos frame by frame |
| Data | **NumPy** | Saving key-points (.npy files) |
| Evaluation | **SacreBLEU** | Translation metric |
| Environment | **Python 3.9+** | Development language |

---

## 📁 Project Structure

```
sign-language-translation/
│
├── 📄 README.md                             ← This file
├── 📄 requirements.txt                      ← Python dependencies
├── 📄 .gitignore
├── 🔬 sanity_check.py                       ← Validate architecture (no dataset needed)
│
├── 📂 data_pipeline/    ── PHASE A : Dataset Pipeline
│   ├── preprocessing.py                     ← Extract MediaPipe keypoints from videos
│   ├── tokenizer.py                         ← Build vocabulary & encode sentences
│   ├── dataset.py                           ← PyTorch Dataset (loads .npy + labels)
│   └── dataloader.py                        ← Train / Val / Test DataLoaders
│
├── 📂 model/            ── PHASE B : Model Architecture
│   ├── positional_encoding.py               ← Sin/cos positional encoding
│   ├── encoder_wrapper.py                   ← SignJoey Encoder ❄️ FROZEN
│   ├── decoder_wrapper.py                   ← SignJoey Decoder 🔥 + classifier 🆕
│   └── cslt_model.py                        ← Full CSLT model (main class)
│
├── 📂 training/         ── PHASE C : Training & Evaluation
│   ├── train.py                             ← Main fine-tuning loop
│   ├── loss.py                              ← CrossEntropyLoss (ignores PAD tokens)
│   ├── metrics.py                           ← BLEU-1 and BLEU-4 score computation
│   └── inference.py                         ← Translate a new video (production)
│
├── 📂 notebooks/        ── Exploration & Analysis
│   ├── 01_data_exploration.ipynb            ← Explore How2Sign dataset
│   ├── 02_keypoint_extraction_demo.ipynb    ← Visualize MediaPipe skeleton
│   ├── 03_sanity_check.ipynb               ← Interactive architecture validation
│   └── 04_evaluation.ipynb                 ← Training curves & BLEU analysis
│
├── 📂 data/             ── Raw Data (not versioned in Git)
│   ├── raw/                                 ← How2Sign videos (.mp4)
│   ├── keypoints/                           ← Extracted .npy files (one per video)
│   └── annotations/                         ← English translation .csv files
│
├── 📂 checkpoints/      ── Model Weights (not versioned in Git)
│   ├── phoenix14t.ckpt                      ← Pre-trained SignJoey (PHOENIX-2014T)
│   └── best_model_how2sign.pth             ← Best fine-tuned model on How2Sign
│
└── 📂 outputs/          ── Results
    └── predictions.txt                      ← Generated translations (for BLEU)
```

---

## ⚙️ Phase 1 — Data Preprocessing (PHASE A)

> **Package:** `data_pipeline/`  
> **Data required:** ✅ How2Sign B-F-H 2D Keypoints (test set: 1.6 GB)

We use the **pre-extracted OpenPose keypoints** provided by How2Sign (B-F-H = Body, Face, Hands). This avoids downloading the 290 GB raw videos.

### Files in this phase

| File | Role |
|------|------|
| `data_pipeline/preprocessing.py` | Read OpenPose JSON files, normalize, pad to 150 frames, save as `.npy` |
| `data_pipeline/tokenizer.py` | Build English vocabulary (10,000 words), encode/decode sentences |
| `data_pipeline/dataset.py` | PyTorch Dataset class (loads `.npy` + tokenized labels) |
| `data_pipeline/dataloader.py` | Create Train / Val / Test DataLoaders (80/10/10 split) |

### Keypoint Format (OpenPose)
| Keypoint Group | Count | Features |
|---------------|-------|----------|
| Body (pose) | 25 keypoints × 3 (x, y, conf) | 75 |
| Face | 70 keypoints × 3 | 210 |
| Left hand | 21 keypoints × 3 | 63 |
| Right hand | 21 keypoints × 3 | 63 |
| **Total per frame** | **137 keypoints** | **411 features** |

### Pipeline Steps
1. **Read** OpenPose JSON files from each clip folder
2. **Normalize** with Z-score standardization
3. **Pad/Truncate** all clips to **150 frames** → shape `[150, 411]`
4. **Save** as `.npy` files in `data/processed/keypoints/`
5. **Build vocabulary** from English annotations (special tokens: `<PAD>=0`, `<SOS>=1`, `<EOS>=2`, `<UNK>=3`)

### Execute
```bash
# Step 1: Preprocess keypoints
python -m data_pipeline.preprocessing

# Step 2: Build tokenizer vocabulary
python -m data_pipeline.tokenizer
```

---

## 🔬 Phase 2 — Sanity Check

> **File:** `sanity_check.py`  
> **Data required:** ❌ None — uses randomly generated data

Validates that the architecture works correctly **before** any training.

```bash
python sanity_check.py
```

### Tests Performed

| # | Test | What it checks | Success Condition |
|---|------|---------------|-------------------|
| 1 | **Forward Pass** | Tensor dimensions through all layers | `logits.shape == [4, 19, 500]` |
| 2 | **Overfit Batch** | Backpropagation and gradient flow | Loss decreases > 50% in 50 steps |
| 3 | **Encoder Freeze** | Transfer learning freeze mechanism | Encoder params unchanged after training step |
| 4 | **Greedy Decoding** | Autoregressive inference pipeline | Generated sequence starts with `<SOS>` |
| 5 | **Parameter Count** | Trainable vs frozen parameters | Trainable < total after freeze |

### Expected Output
```
╔═══════════════════════════════════════════════════════╗
║           FINAL SUMMARY                               ║
╠═══════════════════════════════════════════════════════╣
║  ✅ PASS  │  Forward Pass                             ║
║  ✅ PASS  │  Overfit Batch                            ║
║  ✅ PASS  │  Encoder Freeze                           ║
║  ✅ PASS  │  Greedy Decoding                          ║
║  ✅ PASS  │  Parameter Count                          ║
╠═══════════════════════════════════════════════════════╣
║   🎉 ALL TESTS PASSED — Architecture is valid!       ║
╚═══════════════════════════════════════════════════════╝
```

---

## 🚀 Phase 3 — Model Fine-Tuning (PHASE B + C)

> **Package:** `model/` + `training/`  
> **Data required:** ✅ Preprocessed `.npy` files from Phase 1

### Model Architecture (PHASE B)

| File | Component | Role |
|------|-----------|------|
| `model/positional_encoding.py` | Positional Encoding | Sin/cos temporal signals for Transformer |
| `model/encoder_wrapper.py` | Encoder ❄️ | Keypoint embedding + Transformer Encoder (FROZEN) |
| `model/decoder_wrapper.py` | Decoder 🔥 | Token embedding + Transformer Decoder + Output |
| `model/cslt_model.py` | Full CSLT Model | Assembles all components into one module |

### Transfer Learning Strategy

| Component | Strategy | Learning Rate |
|-----------|----------|---------------|
| Keypoint Embedding | 🆕 Trained from scratch | `1e-4` |
| Transformer Encoder | ❄️ Frozen | `0` (not updated) |
| Transformer Decoder | 🔥 Fine-tuned | `1e-4` |
| Output Projection | 🆕 New (10,000 English classes) | `1e-4` |

### Training Pipeline (PHASE C)

| File | Role |
|------|------|
| `training/train.py` | Main training loop with validation + early stopping |
| `training/loss.py` | CrossEntropyLoss (ignores `<PAD>`, label smoothing) |
| `training/metrics.py` | BLEU-1 to BLEU-4 score computation |
| `training/inference.py` | Translate new videos (production/demo mode) |

### Training Configuration
```python
EPOCHS              = 30
BATCH_SIZE          = 8
ENCODER_LR          = 1e-5       # Slow (pre-trained)
DECODER_LR          = 1e-4       # Fast (new layers)
OPTIMIZER           = AdamW
LOSS                = CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
SCHEDULER           = ReduceLROnPlateau(patience=3)
EARLY_STOPPING      = patience=5
GRADIENT_CLIPPING   = max_norm=1.0
```

### Execute
```bash
# Launch training
python -m training.train

# Run inference on a single file
python -m training.inference --input data/processed/keypoints/clip.npy

# Run inference on a directory
python -m training.inference --input data/processed/keypoints/
```

---

## 📊 Phase 4 — Evaluation

The model is evaluated automatically at the end of training using the **BLEU score**:

| BLEU-4 Score | Interpretation |
|-------------|----------------|
| < 10 | Incomprehensible translation |
| 10 – 20 | Partial translation, idea understood |
| 20 – 30 | Correct translation (CSLT research standard) |
| > 30 | High-quality translation |

> 📌 State-of-the-art CSLT models on PHOENIX-2014T achieve ~25 BLEU-4.

---

## 💻 Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/amina-dourdi/sign-language-translation.git
cd sign-language-translation
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Data
Download from [How2Sign](https://how2sign.github.io):
- **B-F-H 2D Keypoints clips (frontal view)** → TEST (1.6 GB) → place in `data/keypoints/`
- **English Translation (manually re-aligned)** → TEST (424K) → place in `data/annotations/`

### 5. Complete Execution Pipeline
```bash
# STEP 0: Validate architecture (no data needed)
python sanity_check.py

# STEP 1: Preprocess keypoints (JSON → .npy)
python -m data_pipeline.preprocessing

# STEP 2: Build vocabulary
python -m data_pipeline.tokenizer

# STEP 3: Train the model
python -m training.train

# STEP 4: Translate new videos
python -m training.inference --input data/processed/keypoints/
```

---

## 🎯 Expected Results

By the end of the project, the system should be able to:

1. **Take as input** pre-extracted keypoints from an ASL video
2. **Process** the skeleton key-points through a Transformer encoder
3. **Translate** the sequence of movements into a coherent English sentence
4. **Evaluate** translation quality using BLEU-4 score

### Example
```
Input  : [150 frames of OpenPose keypoints — person signing "walk to school"]
Output : "the girl is walking to school"
BLEU-4 : ~22 (project target)
```

---

## 📚 References

- **Camgoz et al. (2020)** — *Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation* — [arxiv.org/abs/2003.13830](https://arxiv.org/abs/2003.13830)
- **How2Sign Dataset** — [how2sign.github.io](https://how2sign.github.io)
- **SignJoey (neccam/slt)** — [github.com/neccam/slt](https://github.com/neccam/slt)
- **OpenPose** — [github.com/CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- **PHOENIX-2014T** — [RWTH Aachen University](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/)

---

<div align="center">

**🤟 Continuous Sign Language Translation — Deep Learning Project 2025/2026**

</div>
