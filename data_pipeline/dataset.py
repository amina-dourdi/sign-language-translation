"""
=============================================================
dataset.py — PHASE A : Dataset Pipeline
=============================================================
Role: PyTorch Dataset class that loads preprocessed .npy
      keypoint files and their corresponding tokenized
      English sentences for training.

Usage:
    dataset = CSLTDataset(metadata_path, tokenizer)
    keypoints, target, src_len, tgt_len = dataset[0]

Returns per sample:
    keypoints : torch.FloatTensor  [MAX_FRAMES, input_dim]
    target    : torch.LongTensor   [max_seq_len]
    src_len   : int  (original number of frames before padding)
    tgt_len   : int  (original number of tokens before padding)
=============================================================
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class CSLTDataset(Dataset):
    """
    PyTorch Dataset for Continuous Sign Language Translation.

    Loads pre-processed .npy keypoint files and tokenized
    English sentence targets for the CSLT model.

    Args:
        metadata_path (str): Path to metadata.json (from preprocessing).
        tokenizer (Tokenizer): Fitted tokenizer for encoding sentences.
        max_frames (int): Fixed number of frames per sample.
        transform (callable, optional): Optional transform on keypoints.
    """

    def __init__(self, metadata_path, tokenizer, max_frames=150, transform=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_frames = max_frames
        self.transform = transform

        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Filter: keep only samples that have BOTH keypoints and annotations
        self.samples = []
        skipped = 0
        for clip_id, info in metadata.items():
            npy_path = info.get("npy_path", "")
            sentence = info.get("sentence", "")

            # Skip samples without annotation or without .npy file
            if not sentence or not os.path.exists(npy_path):
                skipped += 1
                continue

            self.samples.append({
                "clip_id": clip_id,
                "npy_path": npy_path,
                "sentence": sentence,
                "original_frames": info.get("original_frames", max_frames),
            })

        print(f"  [CSLTDataset] Loaded {len(self.samples)} samples "
              f"(skipped {skipped} without annotations or keypoints)")

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load a single sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            keypoints (FloatTensor): Shape [max_frames, input_dim]
            target (LongTensor): Shape [max_seq_len] — tokenized sentence
            src_len (int): Original number of frames (before padding)
            tgt_len (int): Original number of tokens (before padding)
        """
        sample = self.samples[idx]

        # ── Load keypoints from .npy ──
        keypoints = np.load(sample["npy_path"])  # (max_frames, input_dim)
        keypoints = torch.FloatTensor(keypoints)

        # Apply optional transform (e.g., data augmentation)
        if self.transform is not None:
            keypoints = self.transform(keypoints)

        # ── Encode the target sentence ──
        token_indices = self.tokenizer.encode(sample["sentence"])
        tgt_len = len(token_indices)

        # Pad the target sequence to fixed length
        token_indices = self.tokenizer.pad_sequence(token_indices)
        target = torch.LongTensor(token_indices)

        # Source length (original frames before padding)
        src_len = sample["original_frames"]

        return keypoints, target, src_len, tgt_len

    def get_sentence(self, idx):
        """
        Get the raw English sentence for a given index.
        Useful for evaluation and debugging.

        Args:
            idx (int): Sample index.

        Returns:
            str: Original English sentence.
        """
        return self.samples[idx]["sentence"]

    def get_clip_id(self, idx):
        """
        Get the clip ID for a given index.

        Args:
            idx (int): Sample index.

        Returns:
            str: Clip ID string.
        """
        return self.samples[idx]["clip_id"]

    @property
    def input_dim(self):
        """
        Auto-detect the input feature dimension from the first sample.

        Returns:
            int: Number of features per frame (e.g., 411 for OpenPose).
        """
        if len(self.samples) == 0:
            return 411  # Default OpenPose dimension
        first_npy = np.load(self.samples[0]["npy_path"])
        return first_npy.shape[1]
