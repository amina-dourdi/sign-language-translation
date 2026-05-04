"""
metrics.py — PHASE C : Training
==================================
Rôle : Calculer les métriques d'évaluation du modèle.

Métriques implémentées :
  1. BLEU-1  : Précision sur les mots individuels
  2. BLEU-4  : Précision sur les séquences de 4 mots
               (métrique standard en traduction CSLT)

Score BLEU cible du projet : ~20-25

Usage :
  score = compute_bleu(predictions, references)
  → retourne un dictionnaire :
    {'bleu1': 45.2, 'bleu4': 22.1}
"""

# TODO: import sacrebleu
