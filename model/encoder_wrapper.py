"""
=============================================================
encoder_wrapper.py — PHASE B : Model Architecture
=============================================================
Role: Encode the input keypoint sequences into rich contextual
      representations using a Transformer Encoder.

Architecture:
    1. KeypointEmbedding: Linear(input_dim → d_model) + LayerNorm
       → Projects raw keypoints (411 features) into model dimension
    2. PositionalEncoding: sin/cos temporal position signals
    3. TransformerEncoder: 6 layers of Multi-Head Self-Attention
       → Captures temporal relationships between frames

Transfer Learning Strategy:
    ❄️ FROZEN: After loading pre-trained weights (PHOENIX-2014T),
    the encoder parameters are frozen to preserve learned temporal
    patterns from German Sign Language.

Input:  keypoints  [Batch, MaxFrames, input_dim]
Output: memory     [Batch, MaxFrames, d_model]
=============================================================
"""

import torch
import torch.nn as nn
from model.positional_encoding import PositionalEncoding


class KeypointEmbedding(nn.Module):
    """
    Project raw keypoint features into the model's hidden dimension.

    This replaces the CNN visual encoder used in traditional
    sign language models. Instead of extracting features from
    raw pixels, we embed pre-extracted keypoint coordinates.

    Args:
        input_dim (int): Raw keypoint feature size (e.g., 411).
        d_model (int): Target embedding dimension (e.g., 512).
        dropout (float): Dropout rate for regularization.
    """

    def __init__(self, input_dim, d_model, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(input_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (Tensor): Raw keypoints [Batch, SeqLen, input_dim].

        Returns:
            Tensor: Projected embeddings [Batch, SeqLen, d_model].
        """
        x = self.projection(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class EncoderWrapper(nn.Module):
    """
    Full encoder pipeline: Embedding → Positional Encoding → Transformer Encoder.

    This encoder processes sequences of keypoint vectors and outputs
    contextual representations that capture temporal dependencies
    between sign language gestures across frames.

    Args:
        input_dim (int): Keypoint feature dimension (e.g., 411).
        d_model (int): Hidden dimension of the Transformer.
        nhead (int): Number of attention heads.
        num_layers (int): Number of Transformer encoder layers.
        dim_feedforward (int): Dimension of the FFN inside each layer.
        dropout (float): Dropout rate.
        max_seq_len (int): Maximum sequence length for PE.
    """

    def __init__(self, input_dim=411, d_model=512, nhead=8,
                 num_layers=6, dim_feedforward=2048, dropout=0.1,
                 max_seq_len=500):
        super().__init__()

        # Step 1: Project keypoints into model dimension
        self.embedding = KeypointEmbedding(input_dim, d_model, dropout)

        # Step 2: Add positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Step 3: Transformer Encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Input shape: [Batch, SeqLen, d_model]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, keypoints, src_key_padding_mask=None):
        """
        Encode a batch of keypoint sequences.

        Args:
            keypoints (Tensor): [Batch, MaxFrames, input_dim]
            src_key_padding_mask (BoolTensor, optional): [Batch, MaxFrames]
                True for padded positions (to be ignored by attention).

        Returns:
            memory (Tensor): [Batch, MaxFrames, d_model]
                Contextual encoder representations.
        """
        # Project keypoints to d_model dimension
        x = self.embedding(keypoints)         # [B, T, d_model]

        # Add positional encoding
        x = self.pos_encoding(x)              # [B, T, d_model]

        # Pass through Transformer encoder layers
        memory = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask,
        )  # [B, T, d_model]

        return memory

    def freeze(self):
        """
        Freeze all encoder parameters (no gradient updates).
        Used for transfer learning: preserve pre-trained knowledge.
        """
        for param in self.parameters():
            param.requires_grad = False
        print("  [ENCODER] ❄️  All parameters FROZEN")

    def unfreeze(self):
        """
        Unfreeze all encoder parameters (allow gradient updates).
        Used when we want to fine-tune the encoder too.
        """
        for param in self.parameters():
            param.requires_grad = True
        print("  [ENCODER] 🔥 All parameters UNFROZEN")

    def count_parameters(self):
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total
