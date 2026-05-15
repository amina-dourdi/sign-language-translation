"""
=============================================================
demo_external_model.py — ASL Word Recognition (Pre-trained)
=============================================================
Uses a PRE-TRAINED VideoMAE model fine-tuned on WLASL-100
(Word-Level ASL) to recognize 100 common ASL sign words
in real-time from your webcam.

No training needed — the model downloads automatically.

The 100 words it recognizes:
  accident, africa, all, apple, basketball, bed, before,
  bird, birthday, black, blue, book, bowling, brown, but,
  can, candy, chair, change, cheat, city, clothes, color,
  computer, cook, cool, corn, cousin, cow, dance, dark,
  deaf, decide, doctor, dog, drink, eat, enjoy, family,
  fine, finish, fish, forget, full, give, go, graduate,
  hat, hearing, help, hot, how, jacket, kiss, language,
  last, later, letter, like, man, many, medicine, meet,
  mother, need, no, now, orange, paint, paper, pink,
  pizza, play, pull, purple, right, same, school, secretary,
  shirt, short, son, study, table, tall, tell, thanksgiving,
  thin, thursday, time, walk, want, what, white, who,
  woman, work, wrong, year, yes

Requirements:
    pip install transformers torch torchvision opencv-python

Usage:
    python demo_external_model.py

Controls:
    Q     = Quit
    C     = Clear translation history
    H     = Toggle help guide (shows how to sign)
    SPACE = Add space to translation
=============================================================
"""

import cv2
import numpy as np
import time
import torch
from collections import deque
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification


# ─── Configuration ────────────────────────────────────────
MODEL_NAME = "Shawon16/videoMAE_base_wlasl_100_50ep_coR_p10"
NUM_FRAMES = 16
BUFFER_SIZE = 32
PREDICT_EVERY = 16
CONFIDENCE_THRESHOLD = 0.15


# ─── Easy signs guide (how to perform them) ──────────────
SIGN_GUIDE = [
    ("BOOK",     "Open palms together, then open like a book"),
    ("DRINK",    "Cup hand, bring to mouth (drinking motion)"),
    ("EAT",      "Pinch fingers, tap mouth repeatedly"),
    ("HELP",     "Fist on flat palm, lift both hands up"),
    ("NO",       "Snap index+middle finger to thumb"),
    ("YES",      "Make fist, nod it up and down (like head nod)"),
    ("WANT",     "Both hands clawed, pull toward yourself"),
    ("LIKE",     "Hand on chest, pull away pinching fingers"),
    ("WALK",     "Both flat hands, alternate flipping forward"),
    ("WORK",     "Fist on fist, tap top fist on bottom"),
    ("DOG",      "Snap fingers + pat thigh"),
    ("COOK",     "Flat hand on flat hand, flip top hand"),
    ("FAMILY",   "Both F-hands, circle outward from you"),
    ("SCHOOL",   "Clap hands twice"),
    ("GIVE",     "Flat-O hand, push away from body"),
    ("PLAY",     "Both Y-hands, twist wrists"),
    ("FINISH",   "Open hands facing you, flip outward"),
    ("MEET",     "Index fingers pointing up, bring together"),
    ("DANCE",    "V-hand swaying over flat palm"),
    ("MOTHER",   "Open hand, thumb taps chin"),
]


# ─── Load Model ──────────────────────────────────────────
def load_model():
    """Load the pre-trained VideoMAE model from HuggingFace."""
    print(f"  📥 Loading model: {MODEL_NAME}")
    print("     (First run downloads ~350 MB, then cached)")

    processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
    model = VideoMAEForVideoClassification.from_pretrained(MODEL_NAME)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  ✅ Model loaded on {device}")
    print(f"  📋 Recognizes {model.config.num_labels} ASL words")

    return processor, model, device


def sample_frames(frame_buffer, num_frames=16):
    """Uniformly sample num_frames from the buffer."""
    buf_len = len(frame_buffer)
    indices = [i * buf_len // num_frames for i in range(num_frames)]
    return [frame_buffer[i] for i in indices]


def draw_help_overlay(frame, h, w):
    """Draw the sign guide overlay on the frame."""
    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (40, 40), (w - 40, h - 40), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)

    # Title
    cv2.putText(frame, "HOW TO SIGN — Quick Reference Guide",
                (70, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
    cv2.putText(frame, "Press H to close | Try these signs in front of your camera!",
                (70, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    # Draw sign instructions in two columns
    col1_x, col2_x = 70, w // 2 + 20
    start_y = 155
    line_h = 28

    for i, (word, desc) in enumerate(SIGN_GUIDE):
        col_x = col1_x if i < 10 else col2_x
        y = start_y + (i % 10) * line_h

        # Word label (colored)
        cv2.putText(frame, f"{word}:", (col_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 2)
        # Description
        cv2.putText(frame, desc, (col_x + 95, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    # Tips at bottom
    tip_y = start_y + 11 * line_h
    cv2.line(frame, (70, tip_y - 15), (w - 70, tip_y - 15), (60, 60, 60), 1)
    tips = [
        "TIP: Face the camera, good lighting, sign in center of frame",
        "TIP: Perform each sign for ~1-2 seconds with clear movements",
        "TIP: The model sees VIDEO (motion matters!), not just hand position",
    ]
    for j, tip in enumerate(tips):
        cv2.putText(frame, tip, (70, tip_y + j * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)


# ─── Main ─────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  ASL WORD RECOGNITION — Pre-trained VideoMAE (WLASL-100)")
    print("  Recognizes 100 ASL sign WORDS from webcam")
    print("=" * 60)

    processor, model, device = load_model()

    id2label = model.config.id2label
    labels = sorted(id2label.values())
    print(f"\n  All 100 words: {', '.join(labels)}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ❌ Cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_buffer = deque(maxlen=BUFFER_SIZE)
    frame_count = 0
    fps = 0
    last_time = time.time()
    prediction = ""
    confidence = 0.0
    top3 = []
    word_history = []
    show_help = True  # Start with help shown

    print("\n  🎥 Webcam started!")
    print("  Press H for sign guide | Q to quit | C to clear\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_buffer.append(frame_rgb)

        # Run prediction
        frame_count += 1
        if frame_count % PREDICT_EVERY == 0 and len(frame_buffer) >= NUM_FRAMES:
            try:
                sampled = sample_frames(frame_buffer, NUM_FRAMES)
                inputs = processor(sampled, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)[0]

                    top_k = torch.topk(probs, k=3)
                    top3 = [(id2label[i.item()], p.item())
                            for i, p in zip(top_k.indices, top_k.values)]

                    if top3[0][1] > CONFIDENCE_THRESHOLD:
                        prediction = top3[0][0]
                        confidence = top3[0][1]
                        if not word_history or word_history[-1] != prediction:
                            word_history.append(prediction)
                            if len(word_history) > 20:
                                word_history = word_history[-20:]
                    else:
                        prediction = ""
                        confidence = 0.0
            except Exception as e:
                print(f"  ⚠ Error: {e}")

        # ── Draw UI (only when help is NOT shown) ──
        if not show_help:
            # Top bar
            cv2.rectangle(frame, (0, 0), (w, 90), (20, 20, 20), -1)
            cv2.putText(frame, "ASL Word Recognition — VideoMAE (WLASL-100)",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (100, 200, 255), 2)
            cv2.putText(frame, "Pre-trained model | 100 ASL words | Press H for sign guide",
                        (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (150, 150, 150), 1)
            cv2.putText(frame, f"FPS: {fps}  |  Buffer: {len(frame_buffer)}/{BUFFER_SIZE}",
                        (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (100, 100, 100), 1)

            # Recording indicator
            pulse = int(abs((frame_count % 30) - 15) * 17)
            cv2.circle(frame, (w - 30, 20), 8, (0, 0, 150 + pulse), -1)

            # Prediction display
            if prediction:
                cv2.rectangle(frame, (w - 400, 100), (w - 10, 270),
                              (30, 30, 30), -1)
                cv2.rectangle(frame, (w - 400, 100), (w - 10, 104),
                              (0, 255, 100), -1)
                cv2.putText(frame, "DETECTED SIGN:", (w - 390, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (150, 150, 150), 1)
                cv2.putText(frame, prediction.upper(), (w - 390, 185),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 100), 3)
                cv2.putText(frame, f"Confidence: {confidence:.1%}",
                            (w - 390, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (180, 180, 180), 1)
                # Top 3
                for i, (lbl, prob) in enumerate(top3[1:3]):
                    cv2.putText(frame, f"{lbl}: {prob:.1%}",
                                (w - 390 + i * 150, 255),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (100, 100, 100), 1)

            # Translation history (bottom)
            cv2.rectangle(frame, (0, h - 80), (w, h), (20, 20, 20), -1)
            cv2.rectangle(frame, (0, h - 80), (w, h - 78),
                          (100, 200, 255), -1)
            cv2.putText(frame, "Translation:", (15, h - 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
            hist_text = " ".join(word_history[-10:])
            cv2.putText(frame, hist_text, (15, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Help overlay
        if show_help:
            draw_help_overlay(frame, h, w)

        # FPS
        if frame_count % 10 == 0:
            now = time.time()
            fps = int(10 / (now - last_time + 1e-8))
            last_time = now

        cv2.imshow("ASL Word Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            word_history = []
        elif key == ord('h'):
            show_help = not show_help
        elif key == ord(' '):
            word_history.append(" ")

    cap.release()
    cv2.destroyAllWindows()
    print("\n  ✅ Demo closed")


if __name__ == "__main__":
    main()
