"""
tokenizer.py — PHASE A : Dataset Pipeline
==========================================
Rôle : Construire le vocabulaire anglais à partir des
       annotations How2Sign et convertir les phrases
       en séquences d'indices numériques.

INPUT  : data/annotations/*.csv (phrases anglaises)
OUTPUT : data/vocabulary.json   (dictionnaire mot → index)

Éléments :
  - Taille du vocabulaire : 15 000 mots
  - Tokens spéciaux : <PAD>=0, <SOS>=1, <EOS>=2, <UNK>=3
  - Méthodes : encode(sentence) → list[int]
               decode(indices)  → str
"""

# TODO: from collections import Counter
