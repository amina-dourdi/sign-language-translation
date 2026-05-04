"""
positional_encoding.py — PHASE B : Model
==========================================
Rôle : Ajouter l'information de position aux vecteurs
       d'embedding pour que le Transformer comprenne
       l'ordre temporel des frames (des signes).

Pourquoi ? Le Transformer n'a pas de notion d'ordre
par défaut. Le Positional Encoding injecte cette info
via des fonctions sinus/cosinus.

Formule :
  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

INPUT  : tensor [Batch, SeqLen, d_model]
OUTPUT : tensor [Batch, SeqLen, d_model]  (+ encoding)
"""

# TODO: import torch, torch.nn as nn, math
