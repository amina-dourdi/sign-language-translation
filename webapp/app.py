"""
=============================================================
webapp/app.py — FastAPI Backend for Real-Time CSLT Translation
=============================================================
Serves a WebSocket endpoint that receives MediaPipe keypoints
from the browser and returns English translations in real-time.

Usage:
    cd sign-language-translation
    uvicorn webapp.app:app --reload --host 0.0.0.0 --port 8000
=============================================================
"""

import os
import sys
import json
import torch
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

# ─── Add project root to PATH so we can import from train_colab.py ───
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train_colab import CSLTModel, Tokenizer

# ─── FastAPI App ──────────────────────────────────────────
app = FastAPI(
    title="CSLT Live Translation",
    description="Real-time American Sign Language to English Translation",
    version="1.0.0"
)

# ─── Configuration ────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model paths — try finetuned first, fallback to pre-trained
FINETUNED_PATH  = str(PROJECT_ROOT / "best_finetuned_model.pth")
PRETRAINED_PATH = str(PROJECT_ROOT / "checkpoints" / "best_model_v3.pth")
TOKENIZER_PATH  = str(PROJECT_ROOT / "data" / "processed" / "tokenizer.json")

# Model architecture — MUST match train_colab.py exactly
MODEL_CONFIG = {
    "input_dim": 411,
    "d_model": 256,
    "nhead": 4,
    "num_encoder_layers": 3,
    "num_decoder_layers": 3,
    "dim_feedforward": 512,
    "dropout": 0.25,
    "max_frames": 150,
    "max_seq_len": 80,
    "max_vocab_size": 10000,
}

# ─── Load Tokenizer ──────────────────────────────────────
print("=" * 50)
print("  CSLT Live Translation — Server Starting")
print("=" * 50)

print("\n[1/2] Loading tokenizer...")
tokenizer = Tokenizer(MODEL_CONFIG["max_vocab_size"], MODEL_CONFIG["max_seq_len"])
if os.path.exists(TOKENIZER_PATH):
    tokenizer.load(TOKENIZER_PATH)
    print(f"  ✅ Tokenizer loaded (vocab_size={tokenizer.vocab_size})")
else:
    print(f"  ⚠️  Tokenizer not found at: {TOKENIZER_PATH}")
    print(f"  → Run preprocess_all_splits.py or train_colab.py first.")

# ─── Load Model ──────────────────────────────────────────
print("\n[2/2] Loading model...")
model = CSLTModel(
    input_dim=MODEL_CONFIG["input_dim"],
    vocab_size=tokenizer.vocab_size,
    d_model=MODEL_CONFIG["d_model"],
    nhead=MODEL_CONFIG["nhead"],
    enc_layers=MODEL_CONFIG["num_encoder_layers"],
    dec_layers=MODEL_CONFIG["num_decoder_layers"],
    dim_ff=MODEL_CONFIG["dim_feedforward"],
    dropout=MODEL_CONFIG["dropout"],
    max_src=MODEL_CONFIG["max_frames"],
    max_tgt=MODEL_CONFIG["max_seq_len"],
).to(DEVICE)

# Try finetuned model first, then pre-trained
model_loaded = False
for path, label in [(FINETUNED_PATH, "finetuned"), (PRETRAINED_PATH, "pre-trained")]:
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            print(f"  ✅ Model loaded ({label}): {os.path.basename(path)}")
            model_loaded = True
            break
        except Exception as e:
            print(f"  ⚠️  Failed to load {label} model: {e}")

if not model_loaded:
    print("  ⚠️  No trained model found — using random weights (for demo only)")
    print(f"  → Train with: python train_colab.py")
    print(f"  → Or fine-tune: python finetune_colab.py")

model.eval()
print(f"\n  Device: {DEVICE}")
print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print("=" * 50)

# ─── Serve Frontend ──────────────────────────────────────
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def get_index():
    """Serve the main HTML page."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "CSLT API is running. Place index.html in webapp/static/"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "device": DEVICE,
        "vocab_size": tokenizer.vocab_size,
    }


# ─── WebSocket for Real-Time Translation ─────────────────
@app.websocket("/ws/translate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔗 New WebSocket connection established")

    frames_buffer = []
    MAX_FRAMES = MODEL_CONFIG["max_frames"]
    KEYPOINT_DIM = MODEL_CONFIG["input_dim"]
    MIN_FRAMES_TO_TRANSLATE = 30  # Minimum frames before attempting translation

    try:
        while True:
            # Receive keypoints (411 values) from the browser
            data = await websocket.receive_text()
            keypoints = json.loads(data)

            if len(keypoints) == KEYPOINT_DIM:
                frames_buffer.append(keypoints)

                # Translate once we have enough frames
                if len(frames_buffer) >= MIN_FRAMES_TO_TRANSLATE:
                    # Build input tensor
                    input_seq = np.array(frames_buffer, dtype=np.float32)

                    # Z-score normalization (same as preprocessing)
                    mean = np.mean(input_seq, axis=0, keepdims=True)
                    std = np.std(input_seq, axis=0, keepdims=True)
                    input_seq = (input_seq - mean) / (std + 1e-8)

                    # Pad or truncate to MAX_FRAMES
                    if len(input_seq) < MAX_FRAMES:
                        padding = np.zeros((MAX_FRAMES - len(input_seq), KEYPOINT_DIM), dtype=np.float32)
                        input_seq = np.concatenate([input_seq, padding], axis=0)
                    else:
                        input_seq = input_seq[:MAX_FRAMES]

                    # Run inference
                    tensor_input = torch.FloatTensor(input_seq).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        try:
                            translations = model.translate(
                                tensor_input,
                                max_len=MODEL_CONFIG["max_seq_len"],
                                rep_penalty=1.3,
                                tokenizer=tokenizer
                            )
                            translation = translations[0] if isinstance(translations, list) else str(translations)

                            if translation.strip():
                                await websocket.send_json({"translation": translation})
                        except Exception as e:
                            print(f"  Inference error: {e}")

                    # Sliding window — advance by 5 frames
                    frames_buffer = frames_buffer[5:]

    except WebSocketDisconnect:
        print("🔌 Client disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
