"""
=============================================================
cslt_model.py — PHASE B : Model Architecture
=============================================================
Role: Assemble ALL model components into one unified PyTorch
      module for Continuous Sign Language Translation.

Full Architecture Pipeline:
    Raw Keypoints [B, 150, 411]
         ↓
    KeypointEmbedding  (Linear 411→512 + LayerNorm)
         ↓
    PositionalEncoding (sin/cos temporal signals)
         ↓
    TransformerEncoder ❄️ FROZEN  (6 layers, 8 heads)
         ↓  memory [B, 150, 512]
    TransformerDecoder 🔥 FINE-TUNED (6 layers, cross-attention)
         ↓
    OutputProjection   🆕 NEW  (Linear 512→vocab_size)
         ↓
    Logits [B, S, vocab_size]

Methods:
    forward()    → Training (teacher forcing)
    translate()  → Inference (autoregressive greedy decoding)
    save()       → Save model weights
    load()       → Load model weights
=============================================================
"""

import torch
import torch.nn as nn
from model.encoder_wrapper import EncoderWrapper
from model.decoder_wrapper import DecoderWrapper


# Import special token indices from tokenizer
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2


class CSLTModel(nn.Module):
    """
    Continuous Sign Language Translation Model.

    End-to-end Transformer that takes keypoint sequences as input
    and produces English text translations as output.

    Args:
        input_dim (int): Keypoint feature dimension (411 for OpenPose).
        vocab_size (int): Target vocabulary size.
        d_model (int): Transformer hidden dimension.
        nhead (int): Number of attention heads.
        num_encoder_layers (int): Number of encoder layers.
        num_decoder_layers (int): Number of decoder layers.
        dim_feedforward (int): FFN inner dimension.
        dropout (float): Dropout rate.
        max_src_len (int): Max source sequence length (frames).
        max_tgt_len (int): Max target sequence length (tokens).
    """

    def __init__(self, input_dim=411, vocab_size=10000, d_model=512,
                 nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 max_src_len=500, max_tgt_len=200):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # ── Encoder (processes sign language keypoints) ──
        self.encoder = EncoderWrapper(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_src_len,
        )

        # ── Decoder (generates English text) ──
        self.decoder = DecoderWrapper(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_tgt_len,
        )

    def create_padding_mask(self, seq, pad_idx=PAD_IDX):
        """
        Create a boolean mask where True = padded position.

        Args:
            seq (Tensor): Input tensor (any shape with last dim = seq_len).
            pad_idx (int): Index used for padding.

        Returns:
            BoolTensor: True for padded positions.
        """
        if seq.dim() == 3:
            # Source keypoints: padded frames are all zeros
            return (seq.sum(dim=-1) == 0)  # [B, T]
        else:
            # Target tokens: padded tokens equal pad_idx
            return (seq == pad_idx)  # [B, S]

    def forward(self, keypoints, target):
        """
        Forward pass for training (with Teacher Forcing).

        The decoder receives the target sequence shifted right:
        input  to decoder: <SOS> w1 w2 w3 ... wN
        expected output:    w1  w2 w3 w4 ... <EOS>

        Args:
            keypoints (FloatTensor): [B, MaxFrames, input_dim]
            target (LongTensor): [B, MaxSeqLen] — full target with <SOS> and <EOS>

        Returns:
            logits (Tensor): [B, MaxSeqLen-1, vocab_size]
                Predictions for each target position.
        """
        # Create padding masks
        src_padding_mask = self.create_padding_mask(keypoints)  # [B, T]
        tgt_padding_mask = self.create_padding_mask(target[:, :-1])  # [B, S-1]

        # ── Encode ──
        memory = self.encoder(
            keypoints,
            src_key_padding_mask=src_padding_mask,
        )  # [B, T, d_model]

        # ── Decode ──
        # Feed target[:-1] as input (shift right: remove last token)
        # The model predicts target[1:] (shift left: remove <SOS>)
        logits = self.decoder(
            target=target[:, :-1],
            memory=memory,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )  # [B, S-1, vocab_size]

        return logits

    @torch.no_grad()
    def translate(self, keypoints, max_len=80, tokenizer=None):
        """
        Translate keypoint sequences into English text (inference).

        Uses greedy autoregressive decoding: at each step, the model
        predicts the most likely next token and feeds it back as input.

        Args:
            keypoints (FloatTensor): [B, MaxFrames, input_dim]
            max_len (int): Maximum number of tokens to generate.
            tokenizer (Tokenizer, optional): For decoding indices to text.

        Returns:
            If tokenizer is provided:
                list[str]: Translated sentences.
            Otherwise:
                LongTensor: [B, generated_len] — predicted token indices.
        """
        self.eval()
        device = keypoints.device
        batch_size = keypoints.size(0)

        # Create source padding mask
        src_padding_mask = self.create_padding_mask(keypoints)

        # Encode the source keypoints
        memory = self.encoder(
            keypoints,
            src_key_padding_mask=src_padding_mask,
        )  # [B, T, d_model]

        # Start with <SOS> token for all samples in the batch
        generated = torch.full(
            (batch_size, 1), SOS_IDX, dtype=torch.long, device=device
        )

        # Track which sequences have finished (produced <EOS>)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Autoregressive generation loop
        for step in range(max_len):
            # Decode current sequence
            logits = self.decoder(
                target=generated,
                memory=memory,
                memory_key_padding_mask=src_padding_mask,
            )  # [B, current_len, vocab_size]

            # Take the prediction for the last position
            next_token_logits = logits[:, -1, :]  # [B, vocab_size]

            # Greedy: pick the token with highest probability
            next_token = next_token_logits.argmax(dim=-1)  # [B]

            # Replace predictions for finished sequences with <PAD>
            next_token[finished] = PAD_IDX

            # Append predicted token
            generated = torch.cat(
                [generated, next_token.unsqueeze(1)], dim=1
            )

            # Check if any sequence produced <EOS>
            finished = finished | (next_token == EOS_IDX)

            # Stop if ALL sequences have finished
            if finished.all():
                break

        # Convert to text if tokenizer is provided
        if tokenizer is not None:
            sentences = []
            for i in range(batch_size):
                token_ids = generated[i].cpu().tolist()
                sentence = tokenizer.decode(token_ids)
                sentences.append(sentence)
            return sentences

        return generated

    def freeze_encoder(self):
        """Freeze encoder parameters for transfer learning."""
        self.encoder.freeze()

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        self.encoder.unfreeze()

    def get_parameter_groups(self, encoder_lr=1e-5, decoder_lr=1e-4):
        """
        Create parameter groups with different learning rates.

        This is essential for fine-tuning: the encoder (pre-trained)
        gets a smaller learning rate, while the decoder (new) gets
        a larger one.

        Args:
            encoder_lr (float): Learning rate for encoder params.
            decoder_lr (float): Learning rate for decoder params.

        Returns:
            list[dict]: Parameter groups for the optimizer.
        """
        return [
            {"params": self.encoder.parameters(), "lr": encoder_lr},
            {"params": self.decoder.parameters(), "lr": decoder_lr},
        ]

    def count_parameters(self):
        """Print a summary of model parameters."""
        enc_trainable, enc_total = self.encoder.count_parameters()
        dec_trainable, dec_total = self.decoder.count_parameters()
        total = enc_total + dec_total
        trainable = enc_trainable + dec_trainable

        print("\n" + "=" * 50)
        print("   MODEL PARAMETER SUMMARY")
        print("=" * 50)
        print(f"  Encoder  : {enc_trainable:>10,} / {enc_total:>10,}  "
              f"({'frozen' if enc_trainable == 0 else 'trainable'})")
        print(f"  Decoder  : {dec_trainable:>10,} / {dec_total:>10,}  (trainable)")
        print(f"  ─────────────────────────────────")
        print(f"  Total    : {trainable:>10,} / {total:>10,}  trainable")
        print("=" * 50)
        return trainable, total

    def save(self, filepath):
        """Save model weights to a file."""
        torch.save(self.state_dict(), filepath)
        print(f"  [MODEL] Saved to: {filepath}")

    def load(self, filepath, device='cpu'):
        """Load model weights from a file."""
        state_dict = torch.load(filepath, map_location=device)
        self.load_state_dict(state_dict)
        print(f"  [MODEL] Loaded from: {filepath}")
