"""
encoder_wrapper.py — PHASE B : Model
=======================================
Rôle : Wrapper autour de l'encodeur SignJoey pré-entraîné.
       Gère le chargement des poids PHOENIX-2014T et
       le gel (freezing) des paramètres.

Ce fichier contient :
  1. KeypointEmbedding  : Linear(1629 → 512) + LayerNorm
  2. SignJoeyEncoder    : Transformer Encoder (6 couches, 8 têtes)
                          chargé depuis neccam/slt

Méthodes importantes :
  - load_pretrained(path) → charge les poids .ckpt
  - freeze()              → gèle tous les paramètres
  - forward(keypoints)    → [B, T, 512]

INPUT  : keypoints [B, 200, 1629]
OUTPUT : memory    [B, 200, 512]
"""

# TODO: import torch, torch.nn as nn
