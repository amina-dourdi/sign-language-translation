"""
=============================================================
demo_webcam.py — Real-Time Sign Language Translation Demo
=============================================================
Opens the webcam, extracts keypoints with MediaPipe Holistic,
and translates sign language to English using the trained model.

No server needed — just run:
    python demo_webcam.py

Requirements:
    pip install mediapipe opencv-python torch numpy
=============================================================
"""

import sys
import os
import time
import numpy as np
import torch
import cv2
import mediapipe as mp
from pathlib import Path

# ─── Add project root to PATH ────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from train_colab import CSLTModel, Tokenizer

# ─── Configuration ────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FINETUNED_PATH  = str(PROJECT_ROOT / "checkpoints" / "finetuned_model.pth")
PRETRAINED_PATH = str(PROJECT_ROOT / "checkpoints" / "best_model_v3.pth")
TOKENIZER_PATH  = str(PROJECT_ROOT / "data" / "processed" / "tokenizer.json")

MAX_FRAMES   = 150
KEYPOINT_DIM = 411
MIN_FRAMES   = 30   # Minimum frames before translating
WINDOW_STEP  = 10   # Sliding window step


# ─── Load Model ──────────────────────────────────────────
def load_model():
    print("=" * 55)
    print("  CSLT Real-Time Demo — Loading Model...")
    print("=" * 55)

    # Tokenizer
    tok = Tokenizer(10000, 80)
    if os.path.exists(TOKENIZER_PATH):
        tok.load(TOKENIZER_PATH)
    else:
        print(f"  ❌ Tokenizer not found: {TOKENIZER_PATH}")
        sys.exit(1)

    # Model
    model = CSLTModel(
        input_dim=KEYPOINT_DIM, vocab_size=tok.vocab_size,
        d_model=256, nhead=4, enc_layers=3, dec_layers=3,
        dim_ff=512, dropout=0.25, max_src=MAX_FRAMES, max_tgt=80
    ).to(DEVICE)

    # Load weights
    for path, label in [(FINETUNED_PATH, "finetuned"), (PRETRAINED_PATH, "pre-trained")]:
        if os.path.exists(path):
            try:
                model.load_state_dict(torch.load(path, map_location=DEVICE))
                print(f"  ✅ Loaded {label} model: {os.path.basename(path)}")
                break
            except Exception as e:
                print(f"  ⚠️  Failed: {e}")
    else:
        print("  ⚠️  No trained model found — using random weights")

    model.eval()
    print(f"  Device: {DEVICE}")
    print(f"  Vocab: {tok.vocab_size} words")
    print("=" * 55)
    return model, tok


# ─── Extract 411 keypoints from MediaPipe results ────────
def extract_keypoints(results):
    """Convert MediaPipe Holistic results → 411 flat array (OpenPose format)."""
    kp = []

    # Body: 25 keypoints × 3 = 75
    if results.pose_landmarks:
        for i in range(25):
            if i < len(results.pose_landmarks.landmark):
                lm = results.pose_landmarks.landmark[i]
                kp.extend([lm.x, lm.y, lm.visibility])
            else:
                kp.extend([0, 0, 0])
    else:
        kp.extend([0] * 75)

    # Face: 70 keypoints × 3 = 210
    if results.face_landmarks:
        for i in range(70):
            if i < len(results.face_landmarks.landmark):
                lm = results.face_landmarks.landmark[i]
                kp.extend([lm.x, lm.y, 1.0])
            else:
                kp.extend([0, 0, 0])
    else:
        kp.extend([0] * 210)

    # Left Hand: 21 keypoints × 3 = 63
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            kp.extend([lm.x, lm.y, 1.0])
    else:
        kp.extend([0] * 63)

    # Right Hand: 21 keypoints × 3 = 63
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            kp.extend([lm.x, lm.y, 1.0])
    else:
        kp.extend([0] * 63)

    return np.array(kp[:KEYPOINT_DIM], dtype=np.float32)


# ─── Draw skeleton on frame ──────────────────────────────
def draw_landmarks(frame, results):
    """Draw hand/body landmarks on the frame."""
    mp_draw = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    # Body
    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(128, 0, 255), thickness=2, circle_radius=2),
            mp_draw.DrawingSpec(color=(200, 130, 255), thickness=2))

    # Left Hand
    if results.left_hand_landmarks:
        mp_draw.draw_landmarks(
            frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(255, 100, 200), thickness=2, circle_radius=2),
            mp_draw.DrawingSpec(color=(255, 180, 230), thickness=1))

    # Right Hand
    if results.right_hand_landmarks:
        mp_draw.draw_landmarks(
            frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(255, 100, 200), thickness=2, circle_radius=2),
            mp_draw.DrawingSpec(color=(255, 180, 230), thickness=1))


# ─── Translate keypoint buffer ────────────────────────────
def translate_buffer(model, tok, buffer):
    """Run translation on accumulated keypoint frames."""
    seq = np.array(buffer, dtype=np.float32)

    # Z-score normalize
    mean = np.mean(seq, axis=0, keepdims=True)
    std = np.std(seq, axis=0, keepdims=True)
    seq = (seq - mean) / (std + 1e-8)

    # Pad to MAX_FRAMES
    if len(seq) < MAX_FRAMES:
        pad = np.zeros((MAX_FRAMES - len(seq), KEYPOINT_DIM), dtype=np.float32)
        seq = np.concatenate([seq, pad], axis=0)
    else:
        seq = seq[:MAX_FRAMES]

    # Inference
    tensor = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        result = model.translate(tensor, max_len=80, rep_penalty=1.3, tokenizer=tok)
    return result[0] if isinstance(result, list) else str(result)


# ─── Main Loop ────────────────────────────────────────────
def main():
    model, tok = load_model()

    # Setup MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        smooth_landmarks=True,
        refine_face_landmarks=True
    )

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frames_buffer = []
    current_translation = "Waiting for signs..."
    frame_count = 0
    fps = 0
    last_time = time.time()

    print("\n  🎥 Webcam started!")
    print("  Press 'Q' to quit")
    print("  Press 'C' to clear buffer")
    print("  Press 'T' to force translate now\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        # MediaPipe processing
        results = holistic.process(frame_rgb)
        frame_rgb.flags.writeable = True

        # Draw skeleton
        draw_landmarks(frame, results)

        # Extract keypoints
        kp = extract_keypoints(results)
        is_person = np.any(kp != 0)

        if is_person:
            frames_buffer.append(kp)

            # Auto-translate when buffer is big enough
            if len(frames_buffer) >= MIN_FRAMES and len(frames_buffer) % WINDOW_STEP == 0:
                current_translation = translate_buffer(model, tok, frames_buffer)
                # Sliding window
                if len(frames_buffer) > MAX_FRAMES:
                    frames_buffer = frames_buffer[WINDOW_STEP:]

        # ── Draw UI overlay ──
        h, w = frame.shape[:2]

        # Top bar
        cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 20), -1)
        cv2.putText(frame, "CSLT Live Translation", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 130, 255), 2)
        cv2.putText(frame, f"Frames: {len(frames_buffer)} | FPS: {fps}",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        # Status indicator
        color = (0, 255, 100) if is_person else (0, 0, 255)
        cv2.circle(frame, (w - 30, 40), 10, color, -1)
        status = "Tracking" if is_person else "No person"
        cv2.putText(frame, status, (w - 140, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Translation box (bottom)
        cv2.rectangle(frame, (0, h - 90), (w, h), (20, 20, 20), -1)
        cv2.rectangle(frame, (0, h - 90), (w, h - 88), (200, 130, 255), -1)
        cv2.putText(frame, "Translation:", (20, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(frame, current_translation, (20, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # FPS counter
        frame_count += 1
        if frame_count % 10 == 0:
            now = time.time()
            fps = int(10 / (now - last_time + 1e-8))
            last_time = now

        # Show
        cv2.imshow("CSLT Live Translation", frame)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            frames_buffer = []
            current_translation = "Buffer cleared"
        elif key == ord('t') and len(frames_buffer) >= 5:
            current_translation = translate_buffer(model, tok, frames_buffer)

    cap.release()
    cv2.destroyAllWindows()
    holistic.close()
    print("\n  ✅ Demo closed")


if __name__ == "__main__":
    main()
