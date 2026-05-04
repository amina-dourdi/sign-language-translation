"""
=============================================================
train.py — PHASE C : Training
=============================================================
Role: Main training script that orchestrates the entire
      fine-tuning pipeline for the CSLT model.

Pipeline:
    1. Load preprocessed data (metadata.json)
    2. Build tokenizer vocabulary
    3. Create DataLoaders (train / val / test)
    4. Initialize CSLT model
    5. Freeze encoder (transfer learning)
    6. Training loop with validation
    7. Save best model checkpoint
    8. Evaluate on test set (BLEU score)

USAGE:
    python -m training.train

Configuration:
    All hyperparameters are defined in the CONFIG dictionary
    at the top of this file for easy modification.
=============================================================
"""

import os
import sys
import time
import torch
import torch.optim as optim
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.tokenizer import Tokenizer, build_tokenizer_from_metadata
from data_pipeline.dataloader import get_dataloaders
from model.cslt_model import CSLTModel
from training.loss import CSLTLoss
from training.metrics import compute_bleu, print_bleu_report


# ═════════════════════════════════════════════
# CONFIGURATION (edit these values as needed)
# ═════════════════════════════════════════════
CONFIG = {
    # ── Data ──
    "metadata_path": str(PROJECT_ROOT / "data" / "processed" / "metadata.json"),
    "tokenizer_path": str(PROJECT_ROOT / "data" / "processed" / "tokenizer.json"),
    "checkpoint_dir": str(PROJECT_ROOT / "checkpoints"),

    # ── Model Architecture ──
    "input_dim": 411,           # OpenPose keypoint features (auto-detected)
    "d_model": 512,             # Transformer hidden dimension
    "nhead": 8,                 # Number of attention heads
    "num_encoder_layers": 4,    # Encoder layers (reduced for efficiency)
    "num_decoder_layers": 4,    # Decoder layers (reduced for efficiency)
    "dim_feedforward": 1024,    # FFN inner dimension
    "dropout": 0.1,             # Dropout rate
    "max_frames": 150,          # Max source sequence length
    "max_seq_len": 80,          # Max target sequence length

    # ── Training ──
    "epochs": 30,               # Number of training epochs
    "batch_size": 8,            # Samples per batch (reduce if out of memory)
    "encoder_lr": 1e-5,         # Learning rate for frozen/slow encoder
    "decoder_lr": 1e-4,         # Learning rate for decoder
    "weight_decay": 1e-4,       # L2 regularization
    "label_smoothing": 0.1,     # Label smoothing factor
    "max_vocab_size": 10000,    # Maximum vocabulary size
    "patience": 5,              # Early stopping patience (epochs)
    "freeze_encoder": True,     # Whether to freeze encoder weights

    # ── System ──
    "num_workers": 0,           # DataLoader workers (0 for Windows)
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model: CSLT model.
        train_loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: torch device.

    Returns:
        avg_loss (float): Average training loss for this epoch.
        avg_acc (float): Average token accuracy for this epoch.
    """
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    for batch_idx, (keypoints, targets, src_lens, tgt_lens) in enumerate(train_loader):
        # Move to device
        keypoints = keypoints.to(device)  # [B, T, D]
        targets = targets.to(device)      # [B, S]

        # Forward pass
        logits = model(keypoints, targets)  # [B, S-1, vocab_size]

        # Compute loss (compare with shifted targets)
        loss = criterion(logits, targets[:, 1:])

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_acc += criterion.compute_accuracy(logits, targets[:, 1:])
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_acc = total_acc / max(num_batches, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """
    Validate the model on the validation set.

    Args:
        model: CSLT model.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        device: torch device.

    Returns:
        avg_loss (float): Average validation loss.
        avg_acc (float): Average validation accuracy.
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    for keypoints, targets, src_lens, tgt_lens in val_loader:
        keypoints = keypoints.to(device)
        targets = targets.to(device)

        logits = model(keypoints, targets)
        loss = criterion(logits, targets[:, 1:])

        total_loss += loss.item()
        total_acc += criterion.compute_accuracy(logits, targets[:, 1:])
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_acc = total_acc / max(num_batches, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate_bleu(model, test_loader, tokenizer, device):
    """
    Evaluate the model on the test set using BLEU score.

    Generates translations using greedy decoding and compares
    them against the ground truth references.

    Args:
        model: CSLT model.
        test_loader: Test DataLoader.
        tokenizer: Tokenizer for decoding.
        device: torch device.

    Returns:
        dict: BLEU scores.
    """
    model.eval()
    all_predictions = []
    all_references = []

    for keypoints, targets, src_lens, tgt_lens in test_loader:
        keypoints = keypoints.to(device)

        # Generate translations (greedy decoding)
        pred_sentences = model.translate(
            keypoints,
            max_len=CONFIG["max_seq_len"],
            tokenizer=tokenizer,
        )
        all_predictions.extend(pred_sentences)

        # Decode reference sentences
        for i in range(targets.size(0)):
            ref_sentence = tokenizer.decode(targets[i].tolist())
            all_references.append(ref_sentence)

    # Compute BLEU scores
    scores = print_bleu_report(all_predictions, all_references)
    return scores


def train():
    """
    Main training function. Runs the complete training pipeline.
    """
    print("=" * 60)
    print("   CSLT MODEL TRAINING")
    print("   Continuous Sign Language Translation")
    print("=" * 60)
    print(f"  Device: {CONFIG['device']}")
    print(f"  Epochs: {CONFIG['epochs']}")
    print(f"  Batch size: {CONFIG['batch_size']}")

    device = torch.device(CONFIG["device"])

    # ── Step 1: Build or load tokenizer ──
    print("\n" + "─" * 40)
    print("  STEP 1: Tokenizer")
    print("─" * 40)

    tokenizer_path = CONFIG["tokenizer_path"]
    if os.path.exists(tokenizer_path):
        tokenizer = Tokenizer(
            max_vocab_size=CONFIG["max_vocab_size"],
            max_seq_len=CONFIG["max_seq_len"],
        )
        tokenizer.load(tokenizer_path)
    else:
        tokenizer = build_tokenizer_from_metadata(
            CONFIG["metadata_path"],
            save_path=tokenizer_path,
            max_vocab_size=CONFIG["max_vocab_size"],
            max_seq_len=CONFIG["max_seq_len"],
        )

    # ── Step 2: Create DataLoaders ──
    print("\n" + "─" * 40)
    print("  STEP 2: DataLoaders")
    print("─" * 40)

    train_loader, val_loader, test_loader = get_dataloaders(
        metadata_path=CONFIG["metadata_path"],
        tokenizer=tokenizer,
        batch_size=CONFIG["batch_size"],
        max_frames=CONFIG["max_frames"],
        num_workers=CONFIG["num_workers"],
    )

    # Auto-detect input dimension from first batch
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0].shape[2]  # [B, T, D] → D
    CONFIG["input_dim"] = input_dim
    print(f"  Input dimension: {input_dim}")

    # ── Step 3: Initialize Model ──
    print("\n" + "─" * 40)
    print("  STEP 3: Model Initialization")
    print("─" * 40)

    model = CSLTModel(
        input_dim=input_dim,
        vocab_size=tokenizer.vocab_size,
        d_model=CONFIG["d_model"],
        nhead=CONFIG["nhead"],
        num_encoder_layers=CONFIG["num_encoder_layers"],
        num_decoder_layers=CONFIG["num_decoder_layers"],
        dim_feedforward=CONFIG["dim_feedforward"],
        dropout=CONFIG["dropout"],
        max_src_len=CONFIG["max_frames"],
        max_tgt_len=CONFIG["max_seq_len"],
    ).to(device)

    # Freeze encoder if using transfer learning
    if CONFIG["freeze_encoder"]:
        model.freeze_encoder()

    model.count_parameters()

    # ── Step 4: Loss, Optimizer, Scheduler ──
    criterion = CSLTLoss(
        vocab_size=tokenizer.vocab_size,
        pad_idx=0,
        label_smoothing=CONFIG["label_smoothing"],
    )

    optimizer = optim.AdamW(
        model.get_parameter_groups(
            encoder_lr=CONFIG["encoder_lr"],
            decoder_lr=CONFIG["decoder_lr"],
        ),
        weight_decay=CONFIG["weight_decay"],
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, 
    )

    # ── Step 5: Training Loop ──
    print("\n" + "─" * 40)
    print("  STEP 5: Training Loop")
    print("─" * 40)

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    best_val_loss = float('inf')
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, CONFIG["epochs"] + 1):
        start_time = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Track history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start_time

        # Print epoch summary
        print(f"  Epoch {epoch:3d}/{CONFIG['epochs']} │ "
              f"Train Loss: {train_loss:.4f} │ "
              f"Val Loss: {val_loss:.4f} │ "
              f"Train Acc: {train_acc:.1f}% │ "
              f"Val Acc: {val_acc:.1f}% │ "
              f"Time: {elapsed:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = os.path.join(
                CONFIG["checkpoint_dir"], "best_model_how2sign.pth"
            )
            model.save(best_path)
            print(f"  ✅ Best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"\n  ⏹️  Early stopping at epoch {epoch} "
                      f"(no improvement for {CONFIG['patience']} epochs)")
                break

    # ── Step 6: Final Evaluation ──
    print("\n" + "─" * 40)
    print("  STEP 6: Final Evaluation (BLEU)")
    print("─" * 40)

    # Load best model for evaluation
    best_path = os.path.join(CONFIG["checkpoint_dir"], "best_model_how2sign.pth")
    if os.path.exists(best_path):
        model.load(best_path, device=CONFIG["device"])

    scores = evaluate_bleu(model, test_loader, tokenizer, device)

    # Save training history
    import json
    history_path = os.path.join(CONFIG["checkpoint_dir"], "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n  Training history saved to: {history_path}")

    print("\n" + "=" * 60)
    print("   TRAINING COMPLETE ✅")
    print("=" * 60)

    return model, tokenizer, scores


if __name__ == "__main__":
    train()
