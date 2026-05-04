"""
loss.py — PHASE C : Training
================================
Rôle : Définir la fonction de perte utilisée
       pendant l'entraînement du modèle CSLT.

Fonction principale : CrossEntropyLoss
  - Ignore les tokens <PAD> (index 0) dans le calcul
  - Label smoothing optionnel (améliore la généralisation)

Formule :
  loss = CrossEntropyLoss(logits, targets, ignore_index=0)

INPUT  : logits  [B, SeqLen, VocabSize]  (prédictions)
         targets [B, SeqLen]             (vérité terrain)
OUTPUT : loss (scalaire Python float)
"""

# TODO: import torch, torch.nn as nn
