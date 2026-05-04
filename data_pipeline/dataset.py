"""
dataset.py — PHASE A : Dataset Pipeline
=========================================
Rôle : Classe PyTorch Dataset qui charge les fichiers
       .npy (key-points) et les indices de tokens (labels)
       pour les passer au DataLoader.

Hérite de : torch.utils.data.Dataset
Méthodes  :
  - __len__()         → nombre total d'exemples
  - __getitem__(idx)  → (keypoints_tensor, label_tensor)

INPUT  : data/keypoints/*.npy  +  data/vocabulary.json
OUTPUT : tenseurs PyTorch prêts pour l'entraînement
         keypoints : [200, 1629]  (float32)
         label     : [50]         (int64)
"""

# TODO: import torch, numpy, json
# TODO: from torch.utils.data import Dataset
