"""
train.py — PHASE C : Training
================================
Rôle : Script principal d'entraînement (Fine-Tuning).
       Orchestre toutes les phases : chargement des données,
       configuration du modèle, boucle d'entraînement,
       validation, et sauvegarde du meilleur modèle.

USAGE : python training/train.py

Configuration :
  EPOCHS          = 30
  BATCH_SIZE      = 16
  LEARNING_RATE   = 1e-4   (couches non gelées)
  OPTIMIZER       = Adam
  SCHEDULER       = ReduceLROnPlateau(patience=3)

Sauvegarde :
  - Meilleur modèle → models/finetuned/best_model_how2sign.pth
  - Checkpoint complet (reprise possible)
"""

# TODO: import torch, torch.optim as optim
# TODO: from data_pipeline.dataloader import get_dataloaders
# TODO: from model.cslt_model import CSLTModel
# TODO: from training.loss import CSLTLoss
# TODO: from training.metrics import compute_bleu
