"""
=============================================================
preprocess_all_splits.py — FAST Preprocessing (Multiprocessing)
=============================================================
Reads 3 separate JSON keypoint folders and produces a unified
metadata.json with a "split" field per clip (train/val/test).

OPTIMIZATIONS:
  - Skips clips that are already processed (.npy exists)
  - Uses multiprocessing to parallelize across CPU cores
  - Faster JSON reading

Usage:
    python preprocess_all_splits.py
    python preprocess_all_splits.py --force   (reprocess everything)
=============================================================
"""

import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

PROJECT_ROOT = Path(__file__).resolve().parent
MAX_FRAMES   = 150
KEYPOINT_DIM = 411

# ─── Paths per split ──────────────────────────────────────
SPLITS = {
    "train": {
        "json_dir": str(PROJECT_ROOT / "data" / "keypoints" / "json"),
        "csv_file": str(PROJECT_ROOT / "data" / "annotations" / "how2sign_realigned_train.csv"),
    },
    "val": {
        "json_dir": str(PROJECT_ROOT / "data" / "keypoints" / "json_val"),
        "csv_file": str(PROJECT_ROOT / "data" / "annotations" / "how2sign_realigned_val.csv"),
    },
    "test": {
        "json_dir": str(PROJECT_ROOT / "data" / "keypoints" / "json_test"),
        "csv_file": str(PROJECT_ROOT / "data" / "annotations" / "how2sign_realigned_test.csv"),
    },
}

OUT_KP_DIR = str(PROJECT_ROOT / "data" / "processed" / "keypoints")
OUT_META   = str(PROJECT_ROOT / "data" / "processed" / "metadata.json")
os.makedirs(OUT_KP_DIR, exist_ok=True)


# ─── Helper functions ────────────────────────────────────
def read_openpose_json_fast(json_path):
    """Read a single OpenPose JSON file → flat 1D array (411,). Optimized."""
    try:
        with open(json_path, 'rb') as f:  # rb is faster than r
            data = json.loads(f.read())
    except Exception:
        return np.zeros(KEYPOINT_DIM, dtype=np.float32)

    people = data.get("people")
    if not people:
        return np.zeros(KEYPOINT_DIM, dtype=np.float32)

    person = people[0]
    body       = person.get("pose_keypoints_2d",       [0.0] * 75)
    face       = person.get("face_keypoints_2d",       [0.0] * 210)
    hand_left  = person.get("hand_left_keypoints_2d",  [0.0] * 63)
    hand_right = person.get("hand_right_keypoints_2d", [0.0] * 63)
    return np.array(body + face + hand_left + hand_right, dtype=np.float32)


def process_single_clip(args):
    """Process ONE clip — used by multiprocessing Pool.
    Returns (clip_id, npy_path, orig_len) or None if failed."""
    clip_dir, out_dir, force = args
    clip_id = os.path.basename(clip_dir)
    npy_path = os.path.join(out_dir, f"{clip_id}.npy")

    # Skip if already processed
    if not force and os.path.exists(npy_path):
        try:
            kp = np.load(npy_path)
            if kp.shape == (MAX_FRAMES, KEYPOINT_DIM):
                return (clip_id, npy_path, MAX_FRAMES, True)  # True = skipped
        except Exception:
            pass  # Corrupted file, reprocess

    # Find JSON files
    json_files = sorted(glob.glob(os.path.join(clip_dir, "*_keypoints.json")))
    if not json_files:
        json_files = sorted(glob.glob(os.path.join(clip_dir, "*.json")))
    if not json_files:
        return None

    # Read all frames
    frames = [read_openpose_json_fast(jf) for jf in json_files]
    kp = np.array(frames, dtype=np.float32)

    # Normalize (Z-score)
    eps = 1e-8
    mean = np.mean(kp, axis=0, keepdims=True)
    std = np.std(kp, axis=0, keepdims=True)
    kp = ((kp - mean) / (std + eps)).astype(np.float32)

    # Pad or truncate
    T = kp.shape[0]
    orig_len = min(T, MAX_FRAMES)
    if T >= MAX_FRAMES:
        kp = kp[:MAX_FRAMES]
    else:
        pad = np.zeros((MAX_FRAMES - T, KEYPOINT_DIM), dtype=np.float32)
        kp = np.concatenate([kp, pad], axis=0)

    # Save
    np.save(npy_path, kp)
    return (clip_id, npy_path, orig_len, False)  # False = newly processed


def load_csv_annotations(csv_file):
    """Load {clip_id: sentence} from one How2Sign CSV file."""
    if not os.path.exists(csv_file):
        print(f"  [WARNING] CSV not found: {csv_file}")
        return {}
    try:
        df = pd.read_csv(csv_file, sep='\t')
    except Exception:
        df = pd.read_csv(csv_file, sep=',')

    id_col, text_col = None, None
    for col in df.columns:
        cl = col.strip().lower()
        if cl in ['sentence_name', 'video_id', 'id', 'name', 'clip_id']:
            id_col = col
        if cl in ['sentence', 'translation', 'text', 'english']:
            text_col = col

    if id_col is None or text_col is None:
        id_col   = df.columns[0]
        text_col = df.columns[-1]
        print(f"  [WARNING] Auto-selected columns: id='{id_col}', text='{text_col}'")

    result = {}
    for _, row in df.iterrows():
        cid  = str(row[id_col]).strip()
        sent = str(row[text_col]).strip()
        if sent and sent != 'nan':
            result[cid] = sent
    return result


# ─── Main Pipeline ────────────────────────────────────────
def run(force=False):
    print("=" * 60)
    print("   PREPROCESSING — FAST MODE (Multiprocessing)")
    print("=" * 60)

    num_workers = max(1, cpu_count() - 1)  # Leave 1 core free
    print(f"  CPU cores available: {cpu_count()}")
    print(f"  Workers used: {num_workers}")
    if not force:
        print(f"  ⚡ Skip mode ON — already processed clips will be skipped")
    print()

    all_metadata = {}

    for split_name, cfg in SPLITS.items():
        json_dir = cfg["json_dir"]
        csv_file = cfg["csv_file"]

        print(f"{'─' * 55}")
        print(f"  Processing split: [{split_name.upper()}]")
        print(f"  JSON dir : {json_dir}")
        print(f"{'─' * 55}")

        if not os.path.exists(json_dir):
            print(f"  ⚠️  Folder not found, skipping: {json_dir}")
            continue

        # Load annotations
        annotations = load_csv_annotations(csv_file)
        print(f"  Annotations loaded: {len(annotations)} clips")

        # Find clip directories
        clip_dirs = []
        for entry in os.scandir(json_dir):
            if entry.is_dir():
                clip_dirs.append(entry.path)
        if not clip_dirs:
            # Maybe JSONs are directly in json_dir subfolders
            for root, dirs, files in os.walk(json_dir):
                jsons = [f for f in files if f.endswith('.json') and f != '.gitkeep']
                if jsons and not dirs:
                    clip_dirs.append(root)

        print(f"  Clip directories found: {len(clip_dirs)}")

        # Prepare args for multiprocessing
        mp_args = [(cd, OUT_KP_DIR, force) for cd in clip_dirs]

        # Process with multiprocessing
        skipped = 0
        processed = 0
        failed = 0

        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(process_single_clip, mp_args, chunksize=32),
                total=len(clip_dirs),
                desc=f"  [{split_name}]"
            ))

        for result in results:
            if result is None:
                failed += 1
                continue

            clip_id, npy_path, orig_len, was_skipped = result
            if was_skipped:
                skipped += 1
            else:
                processed += 1

            sentence = annotations.get(clip_id, "")
            all_metadata[clip_id] = {
                "npy_path":        npy_path,
                "original_frames": orig_len,
                "sentence":        sentence,
                "split":           split_name,
            }

        print(f"  ✅ Done: {processed} new + {skipped} skipped + {failed} failed")

    # ── Save metadata ──
    print(f"\n[SAVING] metadata.json → {OUT_META}")
    with open(OUT_META, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    # Summary
    from collections import Counter
    counts = Counter(v["split"] for v in all_metadata.values())
    print("\n  Split summary:")
    for sp, cnt in sorted(counts.items()):
        print(f"    {sp:8s}: {cnt} clips")
    print(f"    {'TOTAL':8s}: {len(all_metadata)} clips")

    with_sent = sum(1 for v in all_metadata.values() if v["sentence"])
    print(f"    With English text: {with_sent} clips")

    print("\n" + "=" * 60)
    print("   ✅ PREPROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Reprocess all clips even if .npy already exists")
    args = parser.parse_args()
    run(force=args.force)
