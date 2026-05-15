# 🤟 Continuous Sign Language Translation (CSLT)
### Academic Project — Deep Learning Module | Data Engineering

> **Automatic translation of American Sign Language (ASL) into English text**  
> using pre-extracted **OpenPose keypoints** from the **How2Sign** dataset,
> processed through a custom **Transformer Encoder-Decoder** model with Transfer Learning.

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
7. [Phase 2 — Training from Scratch](#-phase-2--training-from-scratch)
8. [Phase 3 — Fine-Tuning](#-phase-3--fine-tuning-transfer-learning)
9. [Phase 4 — Evaluation](#-phase-4--evaluation)
10. [Phase 5 — Real-Time Web Demo](#-phase-5--real-time-web-demo)
11. [Installation & Usage](#-installation--usage)
12. [Expected Results](#-expected-results)

---

## 🎯 Project Overview

This project implements an end-to-end **Continuous Sign Language Translation (CSLT)** system. The goal is to take a video of a person signing in **ASL (American Sign Language)** and automatically generate the **corresponding English sentence**.

### Key Technical Choice: Keypoints Instead of Raw Videos
Instead of feeding raw video frames (very heavy) into a CNN, we use **pre-extracted OpenPose keypoints** (hands, body, face) from each frame. This allows us to:
- ✅ Reduce storage from ~50 GB to ~500 MB
- ✅ Train on a regular CPU / free Google Colab
- ✅ Deploy in real-time on a web platform (no GPU required)
- ✅ Make the model insensitive to clothing color or background

### Training Strategy
1. **Train from scratch** on How2Sign train set → produces `best_model_v3.pth`
2. **Fine-tune** with frozen encoder (Transfer Learning) → produces `finetuned_model.pth`
3. **Evaluate** on the official How2Sign test set → BLEU scores

---

## 🏗️ Full Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      FULL PROJECT PIPELINE                       │
└─────────────────────────────────────────────────────────────────┘

  [INPUT]  OpenPose JSON keypoints (How2Sign — ASL — B-F-H 2D)
      │
      ▼
  ┌─────────────────────────────────────┐
  │  PHASE 1 : DATA PREPROCESSING       │  (preprocess_all_splits.py)
  │                                     │
  │  1. Read OpenPose JSON files        │
  │     → 137 keypoints × 3 coords     │
  │     → 411 features per frame       │
  │  2. Z-score normalization           │
  │  3. Padding/Truncation → [150, 411]│
  │  4. Save as .npy files              │
  │  5. Text tokenization               │
  │     → 10,000-word vocabulary       │
  │  6. Split: Train / Val / Test       │
  └─────────────────────────────────────┘
      │
      ▼  .npy files + token indices
  ┌─────────────────────────────────────┐
  │  PHASE 2+3 : MODEL                  │  (train_colab.py / finetune_colab.py)
  │                                     │
  │  ┌─────────────────────────────┐    │
  │  │ Keypoint Projection Layer   │    │  Linear(411 → 256)
  │  │ + LayerNorm + Dropout       │    │  + Temporal CNN
  │  └────────────┬────────────────┘    │
  │               ▼                     │
  │  ┌─────────────────────────────┐    │
  │  │ 1D Temporal CNN             │    │  Captures local motion
  │  │ (2 conv layers + residual)  │    │  patterns between frames
  │  └────────────┬────────────────┘    │
  │               ▼                     │
  │  ┌─────────────────────────────┐    │
  │  │ Transformer Encoder         │    │  ❄️ FROZEN (fine-tuning)
  │  │ 3 layers, 4 heads, d=256   │    │  or trainable (scratch)
  │  └────────────┬────────────────┘    │
  │               ▼                     │
  │  ┌─────────────────────────────┐    │
  │  │ Transformer Decoder         │    │  🔥 FINE-TUNED
  │  │ 3 layers, 4 heads, d=256   │    │  lr = 5e-5
  │  │ + Cross-Attention           │    │
  │  └────────────┬────────────────┘    │
  │               ▼                     │
  │  ┌─────────────────────────────┐    │
  │  │ Output Projection           │    │  Linear(256 → vocab_size)
  │  │ → English word prediction  │    │  Greedy decoding + rep penalty
  │  └─────────────────────────────┘    │
  └─────────────────────────────────────┘
      │
      ▼
  [OUTPUT]  English sentence: "the woman is walking to school"
```

### Legend
| Symbol | Meaning |
|--------|---------|
| ❄️ FROZEN | Encoder weights frozen during fine-tuning |
| 🔥 FINE-TUNED | Decoder weights updated with small lr (5e-5) |

---

## 📦 Datasets Used

### How2Sign (Main Dataset)
| Property | Value |
|----------|-------|
| Language | ASL — American Sign Language |
| Content | ~35,000 signed sentences |
| Splits | Train (~16K), Val (~2K), Test (~2.3K) |
| Keypoints | B-F-H 2D OpenPose (pre-extracted) |
| Annotations | English translations (TSV/CSV) |
| Link | [how2sign.github.io](https://how2sign.github.io) |

---

## 🛠️ Tech Stack

| Domain | Technology | Role |
|--------|-----------|------|
| Deep Learning | **PyTorch** | Main framework |
| Architecture | Custom **Transformer Encoder-Decoder** | Seq2Seq translation |
| Temporal Feature | **1D Temporal CNN** | Local motion pattern extraction |
| Keypoint Format | **OpenPose** (B-F-H 2D) | Pre-extracted from How2Sign |
| Data | **NumPy** | Keypoint storage (.npy files) |
| Evaluation | **BLEU** (custom) | Translation quality metric |
| Web Backend | **FastAPI** + **WebSocket** | Real-time inference server |
| Web Frontend | **MediaPipe Holistic** | Browser keypoint extraction |
| Environment | **Python 3.9+** | Development language |

---

## 📁 Project Structure

```
sign-language-translation/
│
├── 📄 README.md                        ← This file
├── 📄 requirements.txt                 ← Python dependencies
├── 📄 .gitignore
│
├── 📄 train_colab.py                   ← PHASE 2: Training from scratch
│                                         (Model + Tokenizer + Dataset + BLEU)
├── 📄 finetune_colab.py                ← PHASE 3: Fine-tuning with frozen encoder
│                                         (Transfer Learning + official splits)
├── 📄 preprocess_all_splits.py         ← PHASE 1: Preprocessing (JSON → .npy)
│                                         (Handles train/val/test separately)
│
├── 📂 data_pipeline/                   ← Reusable preprocessing modules
│   ├── __init__.py
│   ├── preprocessing.py                ← OpenPose JSON reader + normalization
│   └── tokenizer.py                    ← Word-level tokenizer (modular version)
│
├── 📂 data/                            ← Data (not tracked in Git)
│   ├── annotations/                    ← How2Sign CSV files
│   │   ├── how2sign_realigned_train.csv
│   │   ├── how2sign_realigned_val.csv
│   │   └── how2sign_realigned_test.csv
│   ├── keypoints/                      ← OpenPose JSON keypoints
│   │   ├── json/                       ← Train clips
│   │   ├── json_val/                   ← Validation clips
│   │   └── json_test/                  ← Test clips
│   ├── raw/                            ← Original videos (.mp4)
│   │   ├── video/                      ← Train videos
│   │   ├── video_val/                  ← Validation videos
│   │   └── video_test/                 ← Test videos
│   └── processed/                      ← Generated files
│       ├── metadata.json               ← Clip metadata with split labels
│       ├── tokenizer.json              ← Saved vocabulary
│       └── keypoints/                  ← Preprocessed .npy files
│
├── 📂 checkpoints/                     ← Model weights (not tracked in Git)
│   ├── best_model_v3.pth              ← Pre-trained model (training from scratch)
│   ├── finetuned_model.pth            ← Fine-tuned model (final)
│   └── history_v3.json                ← Training metrics history
│
├── 📂 webapp/                          ← Real-time web demo
│   ├── app.py                          ← FastAPI backend (WebSocket)
│   └── static/
│       ├── index.html                  ← Frontend interface
│       ├── app.js                      ← MediaPipe + WebSocket client
│       └── style.css                   ← Glassmorphism design
│
└── 📂 docs/                            ← Documentation & diagrams
    ├── architecture.png
    └── arch-1.png
```

---

## ⚙️ Phase 1 — Data Preprocessing

> **Script:** `preprocess_all_splits.py`  
> **Input:** OpenPose JSON keypoints from How2Sign (B-F-H 2D)

### Keypoint Format (OpenPose)
| Keypoint Group | Count | Features |
|---------------|-------|----------|
| Body (pose) | 25 keypoints × 3 (x, y, conf) | 75 |
| Face | 70 keypoints × 3 | 210 |
| Left hand | 21 keypoints × 3 | 63 |
| Right hand | 21 keypoints × 3 | 63 |
| **Total per frame** | **137 keypoints** | **411 features** |

### Pipeline Steps
1. **Read** OpenPose JSON files from each clip folder (train/val/test)
2. **Normalize** with Z-score standardization
3. **Pad/Truncate** all clips to **150 frames** → shape `[150, 411]`
4. **Save** as `.npy` files in `data/processed/keypoints/`
5. **Build vocabulary** from English annotations
6. **Generate** `metadata.json` with split labels (train/val/test)

### Execute
```bash
python preprocess_all_splits.py
```

---

## 🏋️ Phase 2 — Training from Scratch

> **Script:** `train_colab.py`  
> **Output:** `checkpoints/best_model_v3.pth`

Trains the full Transformer model from random initialization.

### Model Architecture

| Component | Details |
|-----------|---------|
| Input Projection | Linear(411 → 256) + LayerNorm + Dropout |
| Temporal CNN | 2× Conv1D (kernel=3) + GELU + Residual |
| Encoder | 3 layers, 4 heads, d_model=256, ff=512 |
| Decoder | 3 layers, 4 heads, d_model=256, ff=512 |
| Output | Linear(256 → vocab_size) |

### Training Configuration
```python
EPOCHS              = 60
BATCH_SIZE          = 8
ACCUMULATION_STEPS  = 4        # Effective batch = 32
LEARNING_RATE       = 5e-4
OPTIMIZER           = AdamW (weight_decay=5e-4)
SCHEDULER           = WarmupCosine (300 warmup steps)
LOSS                = CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
GRADIENT_CLIPPING   = max_norm=0.5
```

### Execute
```bash
python train_colab.py            # Start training
python train_colab.py --resume   # Resume from checkpoint
```

---

## 🎯 Phase 3 — Fine-Tuning (Transfer Learning)

> **Script:** `finetune_colab.py`  
> **Input:** `checkpoints/best_model_v3.pth`  
> **Output:** `checkpoints/finetuned_model.pth`

Fine-tunes the pre-trained model by **freezing the encoder** and only updating the decoder with a small learning rate. Uses official How2Sign splits (train/val/test).

### Transfer Learning Strategy

| Component | Strategy | Learning Rate |
|-----------|----------|---------------|
| Input Projection | ❄️ Frozen | 0 |
| Temporal CNN | ❄️ Frozen | 0 |
| Transformer Encoder | ❄️ Frozen | 0 |
| Transformer Decoder | 🔥 Fine-tuned | 5e-5 |
| Output Projection | 🔥 Fine-tuned | 5e-5 |

### Fine-Tuning Configuration
```python
EPOCHS              = 15
BATCH_SIZE          = 8
ACCUMULATION_STEPS  = 4
LEARNING_RATE       = 5e-5      # 10× smaller than training
OPTIMIZER           = AdamW (weight_decay=1e-4)
LOSS                = CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
GRADIENT_CLIPPING   = max_norm=0.5
```

### Execute
```bash
python finetune_colab.py
```

---

## 📊 Phase 4 — Evaluation

The model is evaluated automatically at the end of both training and fine-tuning using **BLEU scores** and **accuracy** on the test set.

### Metrics
| Metric | Description |
|--------|-------------|
| Test Loss | CrossEntropy loss on test set |
| Test Accuracy | Token-level prediction accuracy |
| BLEU-1 to BLEU-4 | Translation quality (n-gram overlap) |

### BLEU Score Interpretation
| BLEU-4 Score | Interpretation |
|-------------|----------------|
| < 10 | Incomprehensible translation |
| 10 – 20 | Partial translation, idea understood |
| 20 – 30 | Correct translation (CSLT research standard) |
| > 30 | High-quality translation |

> 📌 State-of-the-art CSLT models on PHOENIX-2014T achieve ~25 BLEU-4.

---

## 🌐 Phase 5 — Real-Time Web Demo

> **Backend:** `webapp/app.py` (FastAPI + WebSocket)  
> **Frontend:** `webapp/static/` (HTML + JS + CSS)

The web application provides real-time ASL translation using the webcam:
1. **Browser** captures video via webcam
2. **MediaPipe Holistic** extracts 411 keypoints per frame (client-side)
3. **WebSocket** sends keypoints to FastAPI backend
4. **Model** translates accumulated frames into English
5. **Translation** is displayed in real-time

### Execute
```bash
# Start the web server
uvicorn webapp.app:app --reload --host 0.0.0.0 --port 8000

# Open in browser
# http://localhost:8000
```

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
- **B-F-H 2D Keypoints** (Train/Val/Test) → place in `data/keypoints/json/`, `json_val/`, `json_test/`
- **English Translation CSV** (Train/Val/Test) → place in `data/annotations/`

### 5. Complete Execution Pipeline
```bash
# STEP 1: Preprocess keypoints (JSON → .npy) for all splits
python preprocess_all_splits.py

# STEP 2: Train the model from scratch (on Colab recommended)
python train_colab.py

# STEP 3: Fine-tune with frozen encoder
python finetune_colab.py

# STEP 4: Launch the real-time web demo
uvicorn webapp.app:app --reload --port 8000
```

---

## 🎯 Expected Results

By the end of the project, the system should be able to:

1. **Take as input** pre-extracted keypoints from an ASL video
2. **Process** the skeleton keypoints through Temporal CNN + Transformer Encoder
3. **Translate** the sequence of movements into a coherent English sentence
4. **Evaluate** translation quality using BLEU scores
5. **Demonstrate** real-time translation via webcam in a web browser

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
- **OpenPose** — [github.com/CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- **MediaPipe Holistic** — [google.github.io/mediapipe](https://google.github.io/mediapipe/)
- **PyTorch** — [pytorch.org](https://pytorch.org)

---

<div align="center">

**🤟 Continuous Sign Language Translation — Deep Learning Project 2025/2026**

</div>
