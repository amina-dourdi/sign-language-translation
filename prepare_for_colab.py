"""
prepare_for_colab.py — Zip les fichiers nécessaires pour Colab
Usage: python prepare_for_colab.py
"""
import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "colab_upload"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Zip all .npy keypoints
kp_dir = PROJECT_ROOT / "data" / "processed" / "keypoints"
zip_path = OUTPUT_DIR / "keypoints.zip"

if zip_path.exists():
    print(f"  ⚠️  {zip_path.name} already exists, skipping...")
else:
    npy_files = list(kp_dir.glob("*.npy"))
    print(f"  📦 Zipping {len(npy_files)} .npy files...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
        for i, f in enumerate(npy_files):
            zf.write(f, f"keypoints/{f.name}")
            if (i + 1) % 5000 == 0:
                print(f"     {i+1}/{len(npy_files)}...")
    size_gb = os.path.getsize(zip_path) / (1024**3)
    print(f"  ✅ Created: {zip_path} ({size_gb:.2f} GB)")

# 2. Copy small files
import shutil
files_to_copy = [
    ("train_colab.py", PROJECT_ROOT / "train_colab.py"),
    ("finetune_colab.py", PROJECT_ROOT / "finetune_colab.py"),
    ("metadata.json", PROJECT_ROOT / "data" / "processed" / "metadata.json"),
    ("tokenizer.json", PROJECT_ROOT / "data" / "processed" / "tokenizer.json"),
    ("best_model_v3.pth", PROJECT_ROOT / "checkpoints" / "best_model_v3.pth"),
]

print("\n  📁 Copying files to colab_upload/:")
for name, src in files_to_copy:
    dst = OUTPUT_DIR / name
    if src.exists():
        shutil.copy2(src, dst)
        size = os.path.getsize(dst) / (1024*1024)
        print(f"     ✅ {name} ({size:.1f} MB)")
    else:
        print(f"     ❌ {name} — NOT FOUND: {src}")

print(f"\n{'='*55}")
print(f"  ✅ Tout est prêt dans: {OUTPUT_DIR}")
print(f"  → Upload ce dossier sur Google Drive")
print(f"{'='*55}")
