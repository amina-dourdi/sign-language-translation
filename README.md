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

## ⚙️ Phase 1 — Data Preprocessing

> **File:** `preprocessing.py`  
> **Dataset required:** ✅ How2Sign videos

This phase is run **once** on the entire dataset. It transforms raw `.mp4` videos into numerical `.npy` arrays representing the skeleton movements of the signer.

### Detailed Steps

**Step 1 — Video Cleaning**
- Check that each video can be opened (not corrupted)
- Reject videos that are too short (< 10 frames)
- Convert BGR → RGB (OpenCV to MediaPipe format)

**Step 2 — Keypoint Extraction (MediaPipe Holistic)**
- For each frame: extract 543 anatomical keypoints
  - 33 body pose points (shoulders, arms, hips...)
  - 21 left hand points (all finger joints)
  - 21 right hand points (all finger joints)
  - 468 face mesh points (mouth, eyes, eyebrows)
  - Each point has 3 values: (x, y, z) coordinates
- If a keypoint is not detected → replace with zeros `(0, 0, 0)`
- Output per frame: 543 × 3 = **1,629 numerical values**
- Output for a full video: tensor of shape **[T, 1629]**

**Step 3 — Normalization**
- Method: Z-score standardization (subtract mean, divide by std)
- Goal: make the model invariant to the signer's height and camera distance

**Step 4 — Sequence Length Uniformization**
- Fixed length: **200 frames** for all videos
- If too long → truncate to 200 frames
- If too short → zero-pad at the end
- Create a **padding mask** (1 = real data, 0 = padding)

**Step 5 — Save**
- One `.npy` file per video in `data/keypoints/`
- Final shape: array of `(200, 1629)` per video

**Step 6 — Text Tokenization**
- Build a vocabulary of the **15,000 most frequent words** from How2Sign
- Special tokens: `<PAD>=0`, `<SOS>=1`, `<EOS>=2`, `<UNK>=3`
- Convert each English sentence into a list of integer indices

---

## 🔬 Phase 2 — Sanity Check

> **File:** `sanity_check.py`  
> **Dataset required:** ❌ None — uses randomly generated data

This phase validates that the architecture is correct **before** launching real training. It must be run first.

```bash
python sanity_check.py
```

### Tests Performed

| Test | Description | Success Condition |
|------|-------------|------------------|
| **Test 1** — Forward Pass | Checks all tensor dimensions across layers | `logits.shape == [4, 50, 15000]` |
| **Test 2** — Overfit 1 Batch | Checks that backpropagation works | Loss reduction > 80% in 100 epochs |
| **Test 3** — Layer Freezing | Checks that encoder freezing is active | `frozen params > 0` |
| **Test 4** — Classifier Replacement | Checks the new How2Sign classifier | `out_features == 15,000` |

### Expected Output
```
✅ PASSED  →  Test 1 — Forward Pass
✅ PASSED  →  Test 2 — Overfit 1 Batch
✅ PASSED  →  Test 3 — Layer Freezing
✅ PASSED  →  Test 4 — Classifier Replacement

🎉 ARCHITECTURE VALIDATED — You can now launch Fine-Tuning!
```

---

## 🚀 Phase 3 — Model Fine-Tuning

> **File:** `fine_tune_cslt.py`  
> **Dataset required:** ✅ `.npy` files generated by preprocessing.py

### Fine-Tuning Strategy

```
Source model : SignJoey pre-trained on PHOENIX-2014T (DGS)
Target model : SignJoey fine-tuned on How2Sign (ASL)
```

| Component | Strategy | Learning Rate |
|-----------|----------|---------------|
| Keypoint Embedding | Trained from scratch | `1e-3` |
| Transformer Encoder | ❄️ Frozen | `0` (not updated) |
| Transformer Decoder | 🔥 Fine-tuned | `1e-5` |
| Vocabulary Classifier | 🆕 Replaced + trained | `1e-3` |

### Recommended Training Parameters
```python
EPOCHS          = 30
BATCH_SIZE      = 16
LEARNING_RATE   = 1e-4    # For unfrozen layers
OPTIMIZER       = Adam
LOSS            = CrossEntropyLoss(ignore_index=PAD_IDX)
SCHEDULER       = ReduceLROnPlateau(patience=3)
```

### Launch
```bash
python fine_tune_cslt.py
```

---

## 📊 Phase 4 — Evaluation

The model is evaluated on a held-out test set (videos not seen during training) using the standard machine translation metric:

### BLEU Score (Bilingual Evaluation Understudy)
- **BLEU-1**: Precision on individual words
- **BLEU-4**: Precision on sequences of 4 consecutive words
- Score between 0 and 100

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

### 4. Execution Order

```bash
# STEP 1: Validate the architecture (no data needed)
python sanity_check.py

# STEP 2: Extract keypoints from How2Sign videos
python preprocessing.py

# STEP 3: Launch Fine-Tuning
python fine_tune_cslt.py
```

---

## 🎯 Expected Results

By the end of the project, the system should be able to:

1. **Take as input** a `.mp4` video of a person signing in ASL
2. **Automatically extract** skeleton key-points using MediaPipe
3. **Translate** the sequence of movements into a coherent English sentence
4. **Display** the translation on a user interface (web platform)

### Example
```
Input  : [3-second video — person signing "walk to school"]
Output : "The girl is walking to school"
BLEU-4 : ~22 (project target)
```

---

## 📚 References

- **Camgoz et al. (2020)** — *Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation* — [arxiv.org/abs/2003.13830](https://arxiv.org/abs/2003.13830)
- **How2Sign Dataset** — [how2sign.github.io](https://how2sign.github.io)
- **SignJoey (neccam/slt)** — [github.com/neccam/slt](https://github.com/neccam/slt)
- **MediaPipe Holistic** — [google.github.io/mediapipe](https://google.github.io/mediapipe/solutions/holistic.html)
- **PHOENIX-2014T** — [RWTH Aachen University](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/)

---

<div align="center">

**🤟 Continuous Sign Language Translation — Deep Learning Project 2025/2026**

</div>
