"""
=============================================================
preprocessing.py — PHASE A : Dataset Pipeline
=============================================================
Role: Read the pre-extracted OpenPose JSON keypoints from
      How2Sign (B-F-H 2D Keypoints), normalize them, pad or
      truncate to a fixed number of frames, and save as .npy

INPUT:
    data/keypoints/<clip_id>/  (folders with JSON files per frame)
    data/annotations/*.csv     (English translations)

OUTPUT:
    data/processed/keypoints/  (.npy files, one per clip)
    data/processed/annotations.json  (cleaned mapping)

OpenPose JSON format per frame:
    - pose_keypoints_2d:       25 body keypoints  × 3 (x, y, conf) = 75
    - face_keypoints_2d:       70 face keypoints  × 3              = 210
    - hand_left_keypoints_2d:  21 left hand       × 3              = 63
    - hand_right_keypoints_2d: 21 right hand      × 3              = 63
    Total per frame: 137 keypoints × 3 = 411 features
=============================================================
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MAX_FRAMES = 150          # Pad or truncate all clips to this length
KEYPOINT_DIM = 411        # 137 keypoints × 3 coordinates (x, y, confidence)

# Project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def read_openpose_json(json_path):
    """
    Read a single OpenPose JSON file and extract all keypoints
    into a flat 1D array.

    Args:
        json_path (str): Path to an OpenPose JSON file.

    Returns:
        np.ndarray: Shape (411,) — all keypoints concatenated.
                    Returns zeros if no person is detected.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # If no person detected in this frame, return zeros
    if len(data.get("people", [])) == 0:
        return np.zeros(KEYPOINT_DIM, dtype=np.float32)

    # Take the first (main) person detected
    person = data["people"][0]

    # Extract each keypoint group (default to zeros if missing)
    body = person.get("pose_keypoints_2d", [0.0] * 75)
    face = person.get("face_keypoints_2d", [0.0] * 210)
    hand_left = person.get("hand_left_keypoints_2d", [0.0] * 63)
    hand_right = person.get("hand_right_keypoints_2d", [0.0] * 63)

    # Concatenate all keypoints into a single vector
    all_keypoints = body + face + hand_left + hand_right
    return np.array(all_keypoints, dtype=np.float32)


def process_single_clip(clip_dir):
    """
    Process all JSON files in a single clip directory.
    Each JSON file represents one video frame.

    Args:
        clip_dir (str): Path to directory containing frame JSONs.

    Returns:
        np.ndarray: Shape (num_frames, 411) — keypoints for all frames.
                    Returns None if the clip directory is empty.
    """
    # Find all JSON files, sorted by frame number
    json_files = sorted(glob.glob(os.path.join(clip_dir, "*_keypoints.json")))

    if len(json_files) == 0:
        # Try alternative naming patterns
        json_files = sorted(glob.glob(os.path.join(clip_dir, "*.json")))

    if len(json_files) == 0:
        return None

    # Read all frames
    frames = []
    for jf in json_files:
        frame_keypoints = read_openpose_json(jf)
        frames.append(frame_keypoints)

    return np.array(frames, dtype=np.float32)  # (T, 411)


def pad_or_truncate(sequence, max_len):
    """
    Pad (with zeros) or truncate a sequence to a fixed length.

    Args:
        sequence (np.ndarray): Shape (T, D) — variable-length sequence.
        max_len (int): Desired fixed length.

    Returns:
        np.ndarray: Shape (max_len, D) — fixed-length sequence.
        int: Original length before padding (useful for masking).
    """
    T, D = sequence.shape
    original_length = min(T, max_len)

    if T >= max_len:
        # Truncate: take the first max_len frames
        return sequence[:max_len], original_length
    else:
        # Pad: add zeros at the end
        padding = np.zeros((max_len - T, D), dtype=np.float32)
        padded = np.concatenate([sequence, padding], axis=0)
        return padded, original_length


def normalize_keypoints(keypoints):
    """
    Apply Z-score normalization to keypoints.
    Each feature (column) is normalized independently:
        x_normalized = (x - mean) / (std + epsilon)

    Args:
        keypoints (np.ndarray): Shape (T, D) — raw keypoints.

    Returns:
        np.ndarray: Shape (T, D) — normalized keypoints.
    """
    epsilon = 1e-8  # Avoid division by zero
    mean = np.mean(keypoints, axis=0, keepdims=True)
    std = np.std(keypoints, axis=0, keepdims=True)
    normalized = (keypoints - mean) / (std + epsilon)
    return normalized.astype(np.float32)


def load_annotations(annotations_dir):
    """
    Load and parse the How2Sign annotation CSV file.
    Tries multiple common column-naming conventions.

    Args:
        annotations_dir (str): Path to directory with CSV files.

    Returns:
        dict: Mapping {clip_id: english_sentence}
    """
    csv_files = glob.glob(os.path.join(annotations_dir, "*.csv"))
    if len(csv_files) == 0:
        print("[WARNING] No CSV files found in annotations directory.")
        return {}

    annotations = {}
    for csv_file in csv_files:
        print(f"  Loading annotations from: {os.path.basename(csv_file)}")
        # Try different separators (tab or comma)
        try:
            df = pd.read_csv(csv_file, sep='\t')
        except Exception:
            df = pd.read_csv(csv_file, sep=',')

        # Identify the correct columns
        # How2Sign uses: SENTENCE_NAME, SENTENCE
        id_col = None
        text_col = None

        for col in df.columns:
            col_lower = col.strip().lower()
            if col_lower in ['sentence_name', 'video_id', 'id', 'name', 'clip_id']:
                id_col = col
            if col_lower in ['sentence', 'translation', 'text', 'english']:
                text_col = col

        if id_col is None or text_col is None:
            print(f"  [WARNING] Could not identify columns in {csv_file}")
            print(f"  Available columns: {list(df.columns)}")
            # Fall back to first and last columns
            id_col = df.columns[0]
            text_col = df.columns[-1]
            print(f"  Using: id_col='{id_col}', text_col='{text_col}'")

        for _, row in df.iterrows():
            clip_id = str(row[id_col]).strip()
            sentence = str(row[text_col]).strip()
            if sentence and sentence != 'nan':
                annotations[clip_id] = sentence

    print(f"  Total annotations loaded: {len(annotations)}")
    return annotations


def run_preprocessing(keypoints_dir=None, annotations_dir=None, output_dir=None):
    """
    Main preprocessing pipeline:
    1. Discover all clip directories
    2. Read JSON keypoints for each clip
    3. Normalize with Z-score
    4. Pad/truncate to MAX_FRAMES
    5. Save as .npy files
    6. Save cleaned annotations mapping

    Args:
        keypoints_dir (str): Path to raw keypoints directory.
        annotations_dir (str): Path to annotations CSV directory.
        output_dir (str): Path to save processed files.
    """
    # Default paths
    if keypoints_dir is None:
        keypoints_dir = str(PROJECT_ROOT / "data" / "keypoints")
    if annotations_dir is None:
        annotations_dir = str(PROJECT_ROOT / "data" / "annotations")
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "data" / "processed")

    # Create output directories
    out_kp_dir = os.path.join(output_dir, "keypoints")
    os.makedirs(out_kp_dir, exist_ok=True)

    print("=" * 60)
    print("   HOW2SIGN PREPROCESSING PIPELINE")
    print("=" * 60)

    # ── Step 1: Load annotations ──
    print("\n[STEP 1/4] Loading annotations...")
    annotations = load_annotations(annotations_dir)

    # ── Step 2: Discover clip directories ──
    print("\n[STEP 2/4] Discovering clip directories...")
    # The keypoints directory might have subdirectories, or it might
    # directly contain clip folders. We search recursively.
    clip_dirs = []
    for root, dirs, files in os.walk(keypoints_dir):
        json_files = [f for f in files if f.endswith('.json') and f != '.gitkeep']
        if len(json_files) > 0 and len(dirs) == 0:
            clip_dirs.append(root)

    # If no nested structure found, treat each subdirectory as a clip
    if len(clip_dirs) == 0:
        clip_dirs = [
            os.path.join(keypoints_dir, d)
            for d in os.listdir(keypoints_dir)
            if os.path.isdir(os.path.join(keypoints_dir, d))
        ]

    print(f"  Found {len(clip_dirs)} clips to process")

    if len(clip_dirs) == 0:
        print("[ERROR] No clip directories found!")
        print(f"  Looked in: {keypoints_dir}")
        print("  Make sure to place the How2Sign keypoint folders in data/keypoints/")
        return

    # ── Step 3: Process each clip ──
    print(f"\n[STEP 3/4] Processing clips (max_frames={MAX_FRAMES})...")
    processed_data = {}
    skipped = 0
    detected_dim = None

    for clip_dir in tqdm(clip_dirs, desc="Processing"):
        clip_id = os.path.basename(clip_dir)

        # Read keypoints from JSON files
        keypoints = process_single_clip(clip_dir)
        if keypoints is None:
            skipped += 1
            continue

        # Auto-detect feature dimension from first clip
        if detected_dim is None:
            detected_dim = keypoints.shape[1]
            print(f"  Auto-detected feature dimension: {detected_dim}")

        # Normalize
        keypoints = normalize_keypoints(keypoints)

        # Pad or truncate
        keypoints, original_length = pad_or_truncate(keypoints, MAX_FRAMES)

        # Save as .npy
        save_path = os.path.join(out_kp_dir, f"{clip_id}.npy")
        np.save(save_path, keypoints)

        # Store metadata
        processed_data[clip_id] = {
            "npy_path": save_path,
            "original_frames": original_length,
            "sentence": annotations.get(clip_id, ""),
        }

    # ── Step 4: Save metadata ──
    print(f"\n[STEP 4/4] Saving metadata...")
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

    # Summary
    total = len(processed_data)
    with_text = sum(1 for v in processed_data.values() if v["sentence"])
    print("\n" + "=" * 60)
    print("   PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Clips processed : {total}")
    print(f"  Clips skipped   : {skipped}")
    print(f"  With annotations: {with_text}")
    print(f"  Feature dim     : {detected_dim}")
    print(f"  Fixed length    : {MAX_FRAMES} frames")
    print(f"  Output dir      : {output_dir}")
    print("=" * 60)

    return processed_data


if __name__ == "__main__":
    run_preprocessing()
