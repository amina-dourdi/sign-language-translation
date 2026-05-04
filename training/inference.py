"""
=============================================================
inference.py — PHASE C : Training
=============================================================
Role: Translate sign language videos or keypoint files into
      English text using a trained CSLT model.

This is the production/demo script. It can:
    1. Load a single .npy keypoint file and translate it
    2. Translate all files in a directory (batch mode)
    3. Save predictions to outputs/predictions.txt

USAGE:
    # Translate a single .npy file
    python -m training.inference --input data/processed/keypoints/clip1.npy

    # Translate all .npy files in a directory
    python -m training.inference --input data/processed/keypoints/

    # Specify a custom model checkpoint
    python -m training.inference --input file.npy --model checkpoints/best.pth
=============================================================
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.cslt_model import CSLTModel
from data_pipeline.tokenizer import Tokenizer


# Default paths
DEFAULT_MODEL = str(PROJECT_ROOT / "checkpoints" / "best_model_how2sign.pth")
DEFAULT_TOKENIZER = str(PROJECT_ROOT / "data" / "processed" / "tokenizer.json")
DEFAULT_OUTPUT = str(PROJECT_ROOT / "outputs" / "predictions.txt")


def load_model(model_path, tokenizer, device="cpu",
               input_dim=411, d_model=512, nhead=8,
               num_encoder_layers=4, num_decoder_layers=4):
    """
    Load a trained CSLT model from a checkpoint file.

    Args:
        model_path (str): Path to the .pth checkpoint.
        tokenizer (Tokenizer): Fitted tokenizer (for vocab_size).
        device (str): Device to load model on.
        input_dim (int): Keypoint feature dimension.
        d_model (int): Model hidden dimension.
        nhead (int): Number of attention heads.
        num_encoder_layers (int): Encoder layer count.
        num_decoder_layers (int): Decoder layer count.

    Returns:
        CSLTModel: Loaded model in eval mode.
    """
    model = CSLTModel(
        input_dim=input_dim,
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
    )
    model.load(model_path, device=device)
    model = model.to(device)
    model.eval()
    return model


def translate_single_file(model, tokenizer, npy_path, device="cpu", max_len=80):
    """
    Translate a single .npy keypoint file into English text.

    Args:
        model (CSLTModel): Trained model.
        tokenizer (Tokenizer): Fitted tokenizer.
        npy_path (str): Path to the .npy keypoint file.
        device (str): Device for inference.
        max_len (int): Maximum tokens to generate.

    Returns:
        str: Translated English sentence.
    """
    # Load keypoints
    keypoints = np.load(npy_path)  # (T, D)
    keypoints = torch.FloatTensor(keypoints).unsqueeze(0).to(device)  # (1, T, D)

    # Generate translation
    sentences = model.translate(keypoints, max_len=max_len, tokenizer=tokenizer)
    return sentences[0]


def translate_directory(model, tokenizer, input_dir, device="cpu",
                        max_len=80, output_path=None):
    """
    Translate all .npy files in a directory.

    Args:
        model (CSLTModel): Trained model.
        tokenizer (Tokenizer): Fitted tokenizer.
        input_dir (str): Directory containing .npy files.
        device (str): Device for inference.
        max_len (int): Maximum tokens to generate.
        output_path (str): Path to save predictions.

    Returns:
        dict: Mapping {filename: translation}.
    """
    npy_files = sorted([
        f for f in os.listdir(input_dir) if f.endswith('.npy')
    ])

    if len(npy_files) == 0:
        print(f"[ERROR] No .npy files found in: {input_dir}")
        return {}

    print(f"\n  Translating {len(npy_files)} files...")
    translations = {}

    for filename in npy_files:
        filepath = os.path.join(input_dir, filename)
        translation = translate_single_file(
            model, tokenizer, filepath, device, max_len
        )
        clip_id = filename.replace('.npy', '')
        translations[clip_id] = translation
        print(f"  [{clip_id}] → {translation}")

    # Save predictions to file
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for clip_id, sentence in translations.items():
                f.write(f"{clip_id}\t{sentence}\n")
        print(f"\n  Predictions saved to: {output_path}")

    return translations


def main():
    """Main entry point for inference."""
    parser = argparse.ArgumentParser(
        description="CSLT Inference — Translate sign language to English"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to a .npy file or a directory of .npy files"
    )
    parser.add_argument(
        "--model", "-m", default=DEFAULT_MODEL,
        help="Path to model checkpoint (.pth)"
    )
    parser.add_argument(
        "--tokenizer", "-t", default=DEFAULT_TOKENIZER,
        help="Path to tokenizer JSON file"
    )
    parser.add_argument(
        "--output", "-o", default=DEFAULT_OUTPUT,
        help="Path to save predictions"
    )
    parser.add_argument(
        "--device", "-d", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cpu or cuda)"
    )
    parser.add_argument(
        "--max-len", type=int, default=80,
        help="Maximum number of tokens to generate"
    )

    args = parser.parse_args()

    print("=" * 50)
    print("   CSLT INFERENCE")
    print("=" * 50)

    # Load tokenizer
    print("\n  Loading tokenizer...")
    tokenizer = Tokenizer()
    tokenizer.load(args.tokenizer)

    # Load model
    print("  Loading model...")
    model = load_model(args.model, tokenizer, device=args.device)

    # Run inference
    if os.path.isfile(args.input):
        # Single file
        translation = translate_single_file(
            model, tokenizer, args.input, args.device, args.max_len
        )
        print(f"\n  ┌─────────────────────────────────────┐")
        print(f"  │ File: {os.path.basename(args.input)}")
        print(f"  │ Translation: {translation}")
        print(f"  └─────────────────────────────────────┘")

    elif os.path.isdir(args.input):
        # Directory of files
        translate_directory(
            model, tokenizer, args.input, args.device,
            args.max_len, args.output
        )
    else:
        print(f"[ERROR] Input not found: {args.input}")
        sys.exit(1)

    print("\n  Inference complete ✅")


if __name__ == "__main__":
    main()
