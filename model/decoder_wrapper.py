"""
=============================================================
decoder_wrapper.py — PHASE B : Model Architecture
=============================================================
Role: Decode the encoder's memory representations into English
      text tokens using a Transformer Decoder with Cross-Attention.

Architecture:
    1. TokenEmbedding: nn.Embedding(vocab_size → d_model)
       → Converts token indices into dense vectors
    2. PositionalEncoding: sin/cos position signals
    3. TransformerDecoder: 6 layers with:
       - Masked Self-Attention (causal: can't see future tokens)
       - Cross-Attention (attends to encoder memory)
    4. OutputProjection: Linear(d_model → vocab_size)
       → Produces logits over the entire vocabulary

Transfer Learning Strategy:
    🔥 FINE-TUNED: The decoder layers are trained with a small
    learning rate (1e-5) to adapt from German to English.
    🆕 NEW: The OutputProjection layer is randomly initialized
    (new vocabulary of ~10,000 English words).

Training Mode:
    Teacher Forcing — the true target tokens are fed as input.
Inference Mode:
    Autoregressive — generates one token at a time.

Input:  memory [B, T, d_model] + target [B, S]
Output: logits [B, S, vocab_size]
=============================================================
"""

import torch
import torch.nn as nn
from model.positional_encoding import PositionalEncoding


class DecoderWrapper(nn.Module):
    """
    Full decoder pipeline: Token Embedding → PE → Transformer Decoder → Output.

    Args:
        vocab_size (int): Size of the target vocabulary.
        d_model (int): Hidden dimension of the Transformer.
        nhead (int): Number of attention heads.
        num_layers (int): Number of Transformer decoder layers.
        dim_feedforward (int): FFN inner dimension.
        dropout (float): Dropout rate.
        max_seq_len (int): Maximum target sentence length.
    """

    def __init__(self, vocab_size, d_model=512, nhead=8,
                 num_layers=6, dim_feedforward=2048, dropout=0.1,
                 max_seq_len=200):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Step 1: Token embedding (word index → dense vector)
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=0,  # <PAD> token index
        )

        # Step 2: Positional encoding for target sequence
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Step 3: Transformer Decoder stack
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # [Batch, SeqLen, d_model]
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        # Step 4: Output projection (d_model → vocab_size)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def generate_causal_mask(self, seq_len, device):
        """
        Generate a causal (autoregressive) mask for the decoder.

        This prevents the decoder from attending to future tokens
        during training. Position i can only attend to positions ≤ i.

        Args:
            seq_len (int): Length of the target sequence.
            device: torch device.

        Returns:
            Tensor: Upper triangular mask [seq_len, seq_len].
                    True values are BLOCKED from attention.
        """
        # Create upper triangular matrix (True = blocked)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device),
            diagonal=1
        ).bool()
        return mask

    def forward(self, target, memory,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        """
        Decode encoder memory into vocabulary logits.

        During training, this uses Teacher Forcing: the true
        target tokens (shifted right) are fed as decoder input.

        Args:
            target (LongTensor): Target token indices [B, S].
            memory (Tensor): Encoder output [B, T, d_model].
            tgt_key_padding_mask (BoolTensor, optional): [B, S]
                True for padded target positions.
            memory_key_padding_mask (BoolTensor, optional): [B, T]
                True for padded source positions.

        Returns:
            logits (Tensor): [B, S, vocab_size] — raw predictions.
        """
        seq_len = target.size(1)
        device = target.device

        # Embed target tokens
        tgt_embedded = self.token_embedding(target)    # [B, S, d_model]

        # Scale embeddings (as in original Transformer paper)
        tgt_embedded = tgt_embedded * (self.d_model ** 0.5)

        # Add positional encoding
        tgt_embedded = self.pos_encoding(tgt_embedded)  # [B, S, d_model]

        # Generate causal mask (prevent looking at future tokens)
        tgt_mask = self.generate_causal_mask(seq_len, device)

        # Pass through Transformer decoder
        decoder_output = self.transformer_decoder(
            tgt=tgt_embedded,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # [B, S, d_model]

        # Project to vocabulary size
        logits = self.output_projection(decoder_output)  # [B, S, vocab_size]

        return logits

    def count_parameters(self):
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total
