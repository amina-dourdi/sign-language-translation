"""
inference.py — PHASE C : Training
====================================
Rôle : Générer des traductions anglaises à partir
       de nouvelles vidéos (mode production/démo).

C'est le script utilisé par la plateforme web quand
un utilisateur upload une vidéo.

Pipeline d'inférence :
  1. Charger la vidéo ou le flux webcam
  2. Extraire les key-points avec MediaPipe (temps réel)
  3. Normaliser et formater en [1, 200, 1629]
  4. Passer dans le modèle fine-tuné
  5. Décodage autorégressif (générer mot par mot)
     → s'arrête quand le token <EOS> est prédit
  6. Retourner la phrase anglaise traduite

USAGE :
  python training/inference.py --video path/to/video.mp4
"""

# TODO: import torch, cv2, mediapipe, argparse
# TODO: from model.cslt_model import CSLTModel
# TODO: from data_pipeline.tokenizer import Tokenizer
