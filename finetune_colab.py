"""
=============================================================
finetune_colab.py — FINE-TUNING Script (Official Splits)
=============================================================
Pipeline de Fine-Tuning pour le modèle CSLT :
  1. Charge les poids pré-entraînés (best_model_v3.pth)
  2. Gèle l'Encoder (Transfer Learning)
  3. Fine-tune le Decoder avec un learning rate faible
  4. Utilise les splits officiels How2Sign (train/val/test)
  5. Évalue avec Loss, Accuracy et BLEU scores

Usage:
  python finetune_colab.py
=============================================================
"""

import os, sys, json, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from train_colab import (
    CSLTModel, Tokenizer, KeypointAugmentation,
    CSLTDataset, SubsetByIndices, collate_fn, compute_bleu
)

PROJECT_ROOT = Path(__file__).resolve().parent

# ─── Configuration ─────────────────────────────────────────
CONFIG = {
    # Paths
    "metadata_path":        str(PROJECT_ROOT / "data" / "processed" / "metadata.json"),
    "tokenizer_path":       str(PROJECT_ROOT / "data" / "processed" / "tokenizer.json"),
    "pretrained_model_path": str(PROJECT_ROOT / "checkpoints" / "best_model_v3.pth"),
    "checkpoint_dir":       str(PROJECT_ROOT / "checkpoints"),

    # Model architecture (must match pre-trained model)
    "input_dim": 411, "d_model": 256, "nhead": 4,
    "num_encoder_layers": 3, "num_decoder_layers": 3,
    "dim_feedforward": 512, "dropout": 0.25,
    "max_frames": 150, "max_seq_len": 80, "max_vocab_size": 10000,

    # Fine-tuning hyperparameters
    "freeze_encoder": True,
    "epochs": 15,
    "batch_size": 8,
    "accumulation_steps": 4,
    "lr": 5e-5,
    "weight_decay": 1e-4,
    "num_workers": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def finetune():
    print("=" * 60)
    print("   CSLT FINE-TUNING PIPELINE")
    print("   (Transfer Learning with Frozen Encoder)")
    print("=" * 60)
    device = torch.device(CONFIG["device"])
    print(f"  Device: {device}")

    # ═══════════════════════════════════════════════════════
    # 1. Load Tokenizer
    # ═══════════════════════════════════════════════════════
    print("\n[1/7] Loading Tokenizer...")
    tok = Tokenizer(CONFIG["max_vocab_size"], CONFIG["max_seq_len"])
    tok.load(CONFIG["tokenizer_path"])
    print(f"  Vocabulary size: {tok.vocab_size}")

    # ═══════════════════════════════════════════════════════
    # 2. Load Data — Use official splits from metadata.json
    # ═══════════════════════════════════════════════════════
    print("\n[2/7] Loading datasets...")
    aug = KeypointAugmentation({
        "noise_std": 0.01, "frame_drop_prob": 0.05,
        "scale_range": (0.9, 1.1), "joint_mask_prob": 0.05,
        "time_warp_prob": 0.1
    })
    train_ds_full = CSLTDataset(CONFIG["metadata_path"], tok, CONFIG["max_frames"], augment=aug)
    eval_ds_full  = CSLTDataset(CONFIG["metadata_path"], tok, CONFIG["max_frames"], augment=None)

    # Check if metadata has official split labels
    with open(CONFIG["metadata_path"], "r") as f:
        meta = json.load(f)

    has_official_splits = any("split" in v for v in meta.values())
    clip_ids = list(meta.keys())

    if has_official_splits:
        # ── Official splits from preprocess_all_splits.py ──
        tr_idx = [i for i, cid in enumerate(clip_ids) if meta[cid].get("split") == "train"]
        va_idx = [i for i, cid in enumerate(clip_ids) if meta[cid].get("split") == "val"]
        te_idx = [i for i, cid in enumerate(clip_ids) if meta[cid].get("split") == "test"]
        print(f"  ✅ Using OFFICIAL How2Sign splits:")
    else:
        # ── Fallback: random 80/10/10 split ──
        n = len(train_ds_full)
        indices = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
        tr_n = int(n * 0.8); va_n = int(n * 0.1)
        tr_idx = indices[:tr_n]
        va_idx = indices[tr_n:tr_n + va_n]
        te_idx = indices[tr_n + va_n:]
        print(f"  ⚠️  No official splits found. Using random 80/10/10 split.")
        print(f"  → Run preprocess_all_splits.py to use official How2Sign splits.")

    print(f"     Train: {len(tr_idx)} | Val: {len(va_idx)} | Test: {len(te_idx)}")

    tr_loader = DataLoader(SubsetByIndices(train_ds_full, tr_idx), CONFIG["batch_size"], shuffle=True,  collate_fn=collate_fn)
    va_loader = DataLoader(SubsetByIndices(eval_ds_full,  va_idx), CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn)
    te_loader = DataLoader(SubsetByIndices(eval_ds_full,  te_idx), CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn)

    # ═══════════════════════════════════════════════════════
    # 3. Initialize Model & Load Pre-trained Weights
    # ═══════════════════════════════════════════════════════
    print("\n[3/7] Loading pre-trained model...")
    model = CSLTModel(
        CONFIG["input_dim"], tok.vocab_size, CONFIG["d_model"], CONFIG["nhead"],
        CONFIG["num_encoder_layers"], CONFIG["num_decoder_layers"],
        CONFIG["dim_feedforward"], CONFIG["dropout"], CONFIG["max_frames"], CONFIG["max_seq_len"]
    ).to(device)

    if not os.path.exists(CONFIG["pretrained_model_path"]):
        print(f"  ❌ ERROR: Pre-trained model not found: {CONFIG['pretrained_model_path']}")
        print(f"  → Run train_colab.py first to train from scratch.")
        return

    print(f"  📥 Loading weights from: {os.path.basename(CONFIG['pretrained_model_path'])}")
    model.load_state_dict(torch.load(CONFIG["pretrained_model_path"], map_location=device))
    print(f"  ✅ Pre-trained weights loaded successfully")

    # ═══════════════════════════════════════════════════════
    # 4. FREEZE THE ENCODER (Transfer Learning)
    # ═══════════════════════════════════════════════════════
    print("\n[4/7] Configuring Transfer Learning...")
    if CONFIG["freeze_encoder"]:
        print("  ❄️  Freezing Encoder parameters...")
        for param in model.src_proj.parameters():    param.requires_grad = False
        for param in model.temporal_cnn.parameters(): param.requires_grad = False
        for param in model.encoder.parameters():      param.requires_grad = False

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params    = total_params - trainable_params
    print(f"  Total parameters:     {total_params:>10,}")
    print(f"  Trainable (Decoder):  {trainable_params:>10,}")
    print(f"  Frozen (Encoder):     {frozen_params:>10,}")

    # ═══════════════════════════════════════════════════════
    # 5. Setup Optimizer & Loss
    # ═══════════════════════════════════════════════════════
    print("\n[5/7] Setting up optimizer...")
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
    )
    print(f"  Optimizer: AdamW (lr={CONFIG['lr']}, wd={CONFIG['weight_decay']})")
    print(f"  Loss: CrossEntropy (label_smoothing=0.1)")

    best_path = os.path.join(CONFIG["checkpoint_dir"], "finetuned_model.pth")
    best_val = float('inf')
    accum = CONFIG["accumulation_steps"]
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # ═══════════════════════════════════════════════════════
    # 6. Fine-Tuning Loop
    # ═══════════════════════════════════════════════════════
    print(f"\n[6/7] Starting Fine-Tuning ({CONFIG['epochs']} epochs)...")
    print("─" * 75)
    print(f"  {'Epoch':>5} │ {'TrLoss':>7} {'VaLoss':>7} │ {'TrAcc':>7} {'VaAcc':>7} │ {'Time':>5} │ {'Status'}")
    print("─" * 75)

    for epoch in range(1, CONFIG["epochs"] + 1):
        # ── Train ──
        model.train()
        t0 = time.time()
        ep_loss = ep_acc = ep_n = 0
        optimizer.zero_grad()

        for i, (kp, tgt, _, _) in enumerate(tr_loader):
            kp, tgt = kp.to(device), tgt.to(device)
            logits = model(kp, tgt)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1)) / accum
            loss.backward()

            if (i + 1) % accum == 0 or (i + 1) == len(tr_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                optimizer.zero_grad()

            ep_loss += loss.item() * accum
            mask = (tgt[:, 1:] != 0)
            ep_acc += ((logits.argmax(-1) == tgt[:, 1:]) & mask).sum().item()
            ep_n += mask.sum().item()

        tr_loss = ep_loss / max(len(tr_loader), 1)
        tr_acc  = ep_acc / max(ep_n, 1) * 100

        # ── Validate ──
        model.eval()
        vl = va = vn = 0
        with torch.no_grad():
            for kp, tgt, _, _ in va_loader:
                kp, tgt = kp.to(device), tgt.to(device)
                logits = model(kp, tgt)
                vl += criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1)).item()
                mask = (tgt[:, 1:] != 0)
                va += ((logits.argmax(-1) == tgt[:, 1:]) & mask).sum().item()
                vn += mask.sum().item()

        val_loss = vl / max(len(va_loader), 1)
        val_acc  = va / max(vn, 1) * 100
        elapsed  = time.time() - t0

        # Save history
        history["train_loss"].append(round(tr_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["train_acc"].append(round(tr_acc, 2))
        history["val_acc"].append(round(val_acc, 2))

        # Check for best model
        status = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            status = "✅ SAVED"

        print(f"  {epoch:>3}/{CONFIG['epochs']} │ {tr_loss:>7.4f} {val_loss:>7.4f} │ {tr_acc:>6.1f}% {val_acc:>6.1f}% │ {elapsed:>4.0f}s │ {status}")

    print("─" * 75)

    # Save training history
    hist_path = os.path.join(CONFIG["checkpoint_dir"], "finetune_history.json")
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  📊 Training history saved → {hist_path}")

    # ═══════════════════════════════════════════════════════
    # 7. Final Evaluation on Test Set (BLEU Scores)
    # ═══════════════════════════════════════════════════════
    print(f"\n[7/7] Final Evaluation on Test Set ({len(te_idx)} clips)...")
    print("─" * 50)

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    # Test loss & accuracy
    te_loss = te_acc = te_n = 0
    all_preds, all_refs = [], []

    with torch.no_grad():
        for kp, tgt, _, _ in te_loader:
            kp, tgt = kp.to(device), tgt.to(device)
            logits = model(kp, tgt)
            te_loss += criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1)).item()
            mask = (tgt[:, 1:] != 0)
            te_acc += ((logits.argmax(-1) == tgt[:, 1:]) & mask).sum().item()
            te_n += mask.sum().item()

            # Generate translations for BLEU
            preds = model.translate(kp, CONFIG["max_seq_len"], 1.3, tok)
            all_preds.extend(preds)
            for i in range(tgt.size(0)):
                all_refs.append(tok.decode(tgt[i].tolist()))

    test_loss = te_loss / max(len(te_loader), 1)
    test_acc  = te_acc / max(te_n, 1) * 100

    print(f"\n  📊 TEST RESULTS:")
    print(f"     Test Loss:     {test_loss:.4f}")
    print(f"     Test Accuracy: {test_acc:.2f}%")

    # BLEU Scores
    scores = compute_bleu(all_preds, all_refs)
    print(f"\n  📊 BLEU SCORES:")
    print(f"     BLEU-1: {scores['bleu1']}")
    print(f"     BLEU-2: {scores['bleu2']}")
    print(f"     BLEU-3: {scores['bleu3']}")
    print(f"     BLEU-4: {scores['bleu4']}")

    # Sample translations
    print(f"\n  📝 Sample Translations:")
    for i in range(min(5, len(all_preds))):
        print(f"\n     [{i+1}] Reference: {all_refs[i]}")
        print(f"         Predicted: {all_preds[i]}")

    # Summary
    print("\n" + "=" * 60)
    print("   ✅ FINE-TUNING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"   Best model saved → {best_path}")
    print(f"   History saved    → {hist_path}")
    print(f"   Best Val Loss    → {best_val:.4f}")
    print(f"   Test Accuracy    → {test_acc:.2f}%")
    print(f"   BLEU-4           → {scores['bleu4']}")
    print("=" * 60)


if __name__ == "__main__":
    finetune()
