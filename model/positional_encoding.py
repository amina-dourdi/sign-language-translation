"""
=============================================================
positional_encoding.py — PHASE B : Model Architecture
=============================================================
Role: Inject temporal position information into the input
      embeddings so the Transformer knows the ORDER of frames.

Why needed?
    Unlike RNNs, Transformers process all positions in parallel
    and have no built-in notion of sequence order. Positional
    Encoding adds unique sin/cos signals to each position so
    the model can distinguish frame 1 from frame 100.

Formula:
    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

Input:  tensor [Batch, SeqLen, d_model]
Output: tensor [Batch, SeqLen, d_model]  (with PE added)
=============================================================
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal Positional Encoding from
    "Attention Is All You Need" (Vaswani et al., 2017).

    Creates a fixed (non-learnable) encoding matrix and adds it
    to the input embeddings at each forward pass.

    Args:
        d_model (int): Dimension of the model embeddings.
        max_len (int): Maximum sequence length supported.
        dropout (float): Dropout rate applied after adding PE.
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create the positional encoding matrix: shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # Position indices: shape (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Division term: 10000^(2i/d_model) computed in log space for stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions

        # Add batch dimension: shape (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as buffer (saved with model but not trained)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input embeddings.

        Args:
            x (Tensor): Input embeddings, shape [Batch, SeqLen, d_model].

        Returns:
            Tensor: Embeddings with PE added, same shape as input.
        """
        seq_len = x.size(1)
        # Add the positional encoding (broadcasting over batch dimension)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)
