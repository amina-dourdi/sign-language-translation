"""
=============================================================
dataloader.py — PHASE A : Dataset Pipeline
=============================================================
Role: Create PyTorch DataLoaders for train / val / test splits.
      Handles batching and collation of variable-length data.

Usage:
    train_loader, val_loader, test_loader = get_dataloaders(
        metadata_path, tokenizer, batch_size=16
    )

Split Strategy:
    - If separate metadata files exist for train/val/test → use them
    - Otherwise → split the single dataset 80/10/10
=============================================================
"""

import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path

# Import from our own modules
from data_pipeline.dataset import CSLTDataset


def collate_fn(batch):
    """
    Custom collate function to handle batching of CSLT samples.

    Takes a list of (keypoints, target, src_len, tgt_len) tuples
    and stacks them into proper batch tensors.

    Args:
        batch (list): List of tuples from CSLTDataset.__getitem__().

    Returns:
        keypoints (FloatTensor): Shape [B, max_frames, input_dim]
        targets (LongTensor): Shape [B, max_seq_len]
        src_lengths (LongTensor): Shape [B] — original frame counts
        tgt_lengths (LongTensor): Shape [B] — original token counts
    """
    keypoints_list, targets_list, src_lens, tgt_lens = zip(*batch)

    # Stack into batch tensors
    keypoints = torch.stack(keypoints_list, dim=0)   # [B, T, D]
    targets = torch.stack(targets_list, dim=0)        # [B, S]
    src_lengths = torch.LongTensor(src_lens)          # [B]
    tgt_lengths = torch.LongTensor(tgt_lens)          # [B]

    return keypoints, targets, src_lengths, tgt_lengths


def get_dataloaders(metadata_path, tokenizer, batch_size=16,
                    max_frames=150, num_workers=0,
                    train_ratio=0.8, val_ratio=0.1):
    """
    Create DataLoaders for training, validation, and testing.

    If only one metadata file is provided, it will be split
    into train/val/test according to the given ratios.

    Args:
        metadata_path (str): Path to metadata.json.
        tokenizer (Tokenizer): Fitted tokenizer instance.
        batch_size (int): Number of samples per batch.
        max_frames (int): Fixed number of frames per sample.
        num_workers (int): Number of parallel data loading workers.
        train_ratio (float): Fraction of data for training.
        val_ratio (float): Fraction of data for validation.

    Returns:
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        test_loader (DataLoader): Test data loader.
    """
    # Create the full dataset
    full_dataset = CSLTDataset(
        metadata_path=metadata_path,
        tokenizer=tokenizer,
        max_frames=max_frames,
    )

    total = len(full_dataset)
    if total == 0:
        raise ValueError("Dataset is empty! Check your metadata.json file.")

    # Calculate split sizes
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    # Ensure minimum sizes
    if test_size <= 0:
        test_size = 1
        train_size = total - val_size - test_size
    if val_size <= 0:
        val_size = 1
        train_size = total - val_size - test_size

    print(f"\n[DATALOADERS] Splitting dataset:")
    print(f"  Total samples : {total}")
    print(f"  Train         : {train_size} ({train_size/total*100:.1f}%)")
    print(f"  Validation    : {val_size} ({val_size/total*100:.1f}%)")
    print(f"  Test          : {test_size} ({test_size/total*100:.1f}%)")
    print(f"  Batch size    : {batch_size}")

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible splits
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # Shuffle training data each epoch
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,          # No shuffle for validation
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,          # No shuffle for testing
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=False,
    )

    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")
    print(f"  Test batches  : {len(test_loader)}")

    return train_loader, val_loader, test_loader


def get_single_loader(metadata_path, tokenizer, batch_size=16,
                      max_frames=150, shuffle=False, num_workers=0):
    """
    Create a single DataLoader for all data (useful for inference
    or when using a pre-defined test set).

    Args:
        metadata_path (str): Path to metadata.json.
        tokenizer (Tokenizer): Fitted tokenizer instance.
        batch_size (int): Batch size.
        max_frames (int): Fixed number of frames.
        shuffle (bool): Whether to shuffle data.
        num_workers (int): Number of data loading workers.

    Returns:
        DataLoader: Single data loader for the entire dataset.
    """
    dataset = CSLTDataset(
        metadata_path=metadata_path,
        tokenizer=tokenizer,
        max_frames=max_frames,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=False,
    )

    print(f"  [LOADER] {len(dataset)} samples, {len(loader)} batches")
    return loader
