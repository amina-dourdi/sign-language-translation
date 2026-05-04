"""
preprocessing.py — PHASE A : Dataset Pipeline
================================================
Rôle : Charger les vidéos brutes How2Sign et extraire
       les key-points avec Google MediaPipe Holistic.

INPUT  : data/raw/*.mp4
OUTPUT : data/keypoints/*.npy

Étapes :
  1. Valider et nettoyer les vidéos
  2. Extraire 543 key-points par frame (MediaPipe)
  3. Normalisation Z-score
  4. Padding/Truncation → [200, 1629]
  5. Sauvegarder en .npy
"""

# TODO: import cv2, mediapipe, numpy, pathlib
