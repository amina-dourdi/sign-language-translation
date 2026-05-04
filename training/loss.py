"""
=============================================================
loss.py — PHASE C : Training
=============================================================
Role: Define the loss function used during CSLT model training.

Loss Function: CrossEntropyLoss
    - Computes the difference between predicted and true tokens
    - Ignores <PAD> tokens (index 0) in the loss calculation
    - Optional label smoothing to improve generalization

Formula:
    loss = -sum( y_true * log(y_pred) ) / num_non_pad_tokens

Input:  logits  [B, S, vocab_size]  (model predictions)
        targets [B, S]              (ground truth token indices)
Output: loss    (scalar)
=============================================================
"""

import torch
import torch.nn as nn


class CSLTLoss(nn.Module):
    """
    Cross-Entropy Loss for sequence-to-sequence translation.

    Features:
    - Ignores padding tokens (<PAD> = index 0)
    - Optional label smoothing for better generalization
    - Reports both total loss and per-token loss

    Args:
        vocab_size (int): Size of the target vocabulary.
        pad_idx (int): Index of the <PAD> token (default: 0).
        label_smoothing (float): Label smoothing factor (0.0 = off).
            Distributes a small probability to non-target tokens,
            preventing the model from becoming overconfident.
    """

    def __init__(self, vocab_size, pad_idx=0, label_smoothing=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_idx,
            label_smoothing=label_smoothing,
            reduction='mean',
        )

    def forward(self, logits, targets):
        """
        Compute the cross-entropy loss.

        Args:
            logits (Tensor): Model predictions [B, S, vocab_size].
            targets (LongTensor): Ground truth tokens [B, S].

        Returns:
            loss (Tensor): Scalar loss value.
        """
        # Reshape for CrossEntropyLoss:
        # logits:  [B * S, vocab_size]
        # targets: [B * S]
        B, S, V = logits.shape
        logits_flat = logits.reshape(B * S, V)
        targets_flat = targets.reshape(B * S)

        loss = self.criterion(logits_flat, targets_flat)
        return loss

    def compute_accuracy(self, logits, targets):
        """
        Compute token-level accuracy (ignoring padding).

        Args:
            logits (Tensor): [B, S, vocab_size]
            targets (LongTensor): [B, S]

        Returns:
            float: Accuracy as a percentage (0-100).
        """
        predictions = logits.argmax(dim=-1)  # [B, S]

        # Mask: only count non-padding tokens
        non_pad_mask = (targets != self.pad_idx)
        correct = (predictions == targets) & non_pad_mask
        accuracy = correct.sum().float() / non_pad_mask.sum().float()

        return accuracy.item() * 100  # Convert to percentage
