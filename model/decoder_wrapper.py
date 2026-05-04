"""
decoder_wrapper.py — PHASE B : Model
=======================================
Rôle : Wrapper autour du décodeur SignJoey pré-entraîné.
       Gère le chargement des poids PHOENIX-2014T et
       le fine-tuning avec un learning rate réduit (1e-5).

Ce fichier contient :
  1. SignJoeyDecoder    : Transformer Decoder (6 couches)
                          avec Cross-Attention vers l'encodeur
                          chargé depuis neccam/slt
  2. VocabularyClassifier : nn.Linear(512 → 15 000)
                             REMPLACÉE pour le vocabulaire How2Sign

Génération :
  - Mode entraînement : Teacher Forcing (target connu)
  - Mode inférence    : Autorégressif (mot par mot)

INPUT  : memory [B, 200, 512]  (sortie encodeur)
         target [B, 50]        (indices des tokens)
OUTPUT : logits [B, 50, 15000]
"""

# TODO: import torch, torch.nn as nn
