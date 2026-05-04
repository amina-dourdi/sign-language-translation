"""
dataloader.py — PHASE A : Dataset Pipeline
============================================
Rôle : Créer les DataLoaders PyTorch pour les splits
       train / validation / test du dataset How2Sign.

Retourne :
  - train_loader  : DataLoader (shuffle=True)
  - val_loader    : DataLoader (shuffle=False)
  - test_loader   : DataLoader (shuffle=False)

Splits recommandés (How2Sign) :
  - Train : 80%
  - Val   : 10%
  - Test  : 10%
"""

# TODO: from torch.utils.data import DataLoader, random_split
# TODO: from data_pipeline.dataset import CSLTDataset
