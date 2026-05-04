# 🤟 Continuous Sign Language Translation (CSLT)
### Projet de Fin d'Études (PFA) — Data Engineering & Deep Learning

> **Traduction automatique de la Langue des Signes Américaine (ASL) vers le texte anglais**  
> à partir de vidéos brutes du dataset **How2Sign**, en utilisant le Transfer Learning depuis **PHOENIX-2014T**.

---

## 👥 Équipe & Encadrement

| Rôle | Nom | Filière | École |
|------|-----|---------|-------|
| 🎓 Étudiante | **Amina Dourdi** | Data Engineering | ENSAH |
| 🎓 Étudiante | **Firdawss El Haddouchi** | Data Engineering | ENSAH |
| 🎓 Étudiante | **Oumaima El Ghalbouni** | Data Engineering | ENSAH |

> 🏫 **ENSAH** — École Nationale des Sciences Appliquées d'Al Hoceima  
> 📅 **Année universitaire** : 2025 – 2026  
> 📌 **Module** : Deep Learning — Projet de Fin d'Études (PFA)

---

## 📋 Table des Matières

1. [Vue d'ensemble du projet](#-vue-densemble-du-projet)
2. [Architecture complète](#-architecture-complète)
3. [Datasets utilisés](#-datasets-utilisés)
4. [Stack Technique](#-stack-technique)
5. [Structure du projet](#-structure-du-projet)
6. [Phase 1 — Data Preprocessing](#-phase-1--data-preprocessing)
7. [Phase 2 — Sanity Check](#-phase-2--sanity-check)
8. [Phase 3 — Fine-Tuning du Modèle](#-phase-3--fine-tuning-du-modèle)
9. [Phase 4 — Évaluation](#-phase-4--évaluation)
10. [Installation & Lancement](#-installation--lancement)
11. [Résultats attendus](#-résultats-attendus)

---

## 🎯 Vue d'ensemble du projet

Ce projet implémente un système de **Traduction Continue de la Langue des Signes** (Continuous Sign Language Translation — CSLT) de bout en bout. L'objectif est de prendre une vidéo d'une personne pratiquant l'**ASL (American Sign Language)** et de générer automatiquement la **phrase en anglais correspondante**.

### Contrainte principale
> ❌ Pas d'assemblage manuel d'encodeur et décodeur provenant de dépôts différents.  
> ✅ Utilisation d'un modèle **End-to-End pré-entraîné** (SignJoey sur PHOENIX-2014T) adapté par **Fine-Tuning** à How2Sign.

### Choix technique clé : Key-points plutôt que vidéos brutes
Au lieu d'utiliser des frames vidéo brutes (très lourdes) dans un CNN, nous utilisons **Google MediaPipe Holistic** pour extraire les **key-points du squelette** (mains, corps, visage) de chaque frame. Cela nous permet de :
- ✅ Réduire le stockage de ~50 Go à ~500 Mo
- ✅ Entraîner sur un simple CPU / Google Colab gratuit
- ✅ Déployer en temps réel sur une plateforme web (pas de GPU nécessaire)
- ✅ Rendre le modèle insensible à la couleur de vêtements ou à l'arrière-plan

---

## 🏗️ Architecture Complète

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE COMPLET DU PROJET                    │
└─────────────────────────────────────────────────────────────────┘

  [INPUT]  Vidéo brute .mp4 (How2Sign — ASL)
      │
      ▼
  ┌─────────────────────────────────────┐
  │  PHASE 1 : DATA PREPROCESSING       │  (preprocessing.py)
  │                                     │
  │  1. Nettoyage & validation vidéos   │
  │  2. MediaPipe Holistic              │  ← REMPLACE le CNN
  │     → 543 keypoints × 3 coords     │
  │     → Format [T, 1629]             │
  │  3. Normalisation Z-score           │
  │  4. Padding/Truncation → [200,1629] │
  │  5. Sauvegarde en .npy              │
  │  6. Tokenisation du texte           │
  │     → Vocabulaire 15 000 mots      │
  └─────────────────────────────────────┘
      │
      ▼  Fichiers .npy + indices de tokens
  ┌─────────────────────────────────────┐
  │  PHASE 2 : MODÈLE (Fine-Tuning)     │  (fine_tune_cslt.py)
  │                                     │
  │  ┌─────────────────────────────┐    │
  │  │ Keypoint Embedding Layer    │    │  🆕 Nouvelle couche
  │  │ Linear(1629 → 512)          │    │  Entraînée from scratch
  │  └────────────┬────────────────┘    │
  │               ▼                     │
  │  ┌─────────────────────────────┐    │
  │  │ Transformer Encoder         │    │  ❄️ GELÉ
  │  │ (SignJoey — PHOENIX-2014T)  │    │  Poids pré-entraînés
  │  │ 6 couches, 8 têtes          │    │  conservés intacts
  │  └────────────┬────────────────┘    │
  │               ▼                     │
  │  ┌─────────────────────────────┐    │
  │  │ Transformer Decoder         │    │  🔥 FINE-TUNÉ
  │  │ (SignJoey — PHOENIX-2014T)  │    │  lr = 1e-5
  │  │ Cross-Attention             │    │
  │  └────────────┬────────────────┘    │
  │               ▼                     │
  │  ┌─────────────────────────────┐    │
  │  │ Vocabulary Classifier       │    │  🆕 REMPLACÉE
  │  │ Linear(512 → 15 000)        │    │  Vocabulaire How2Sign
  │  └─────────────────────────────┘    │
  └─────────────────────────────────────┘
      │
      ▼
  [OUTPUT]  Phrase en anglais : "The woman is walking to school"
```

### Légende
| Symbole | Signification |
|---------|--------------|
| ❄️ GELÉ | Poids pré-entraînés sur PHOENIX-2014T, jamais modifiés |
| 🔥 FINE-TUNÉ | Poids pré-entraînés mis à jour lentement (lr = 1e-5) |
| 🆕 NOUVEAU | Couche aléatoire, entièrement entraînée sur How2Sign |

---

## 📦 Datasets Utilisés

### 1. How2Sign (Dataset principal — Fine-Tuning)
| Propriété | Valeur |
|-----------|--------|
| Langue | ASL — American Sign Language |
| Contenu | ~35 000 phrases signées |
| Stockage vidéos | ~50 Go |
| Format annotations | `.csv` (phrases anglaises) |
| Lien | [how2sign.github.io](https://how2sign.github.io) |

### 2. PHOENIX-2014T (Dataset du modèle pré-entraîné)
| Propriété | Valeur |
|-----------|--------|
| Langue | DGS — Deutsche Gebärdensprache (Langue des Signes Allemande) |
| Contenu | ~8 000 phrases (bulletins météo) |
| Utilisé pour | Pré-entraînement de SignJoey |
| Lien | [phoenix.ira.uka.de](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/) |

---

## 🛠️ Stack Technique

| Domaine | Technologie | Rôle |
|---------|------------|------|
| Deep Learning | **PyTorch** | Framework principal |
| Architecture | **SignJoey** (`neccam/slt`) | Modèle Seq2Seq pré-entraîné |
| Extraction Keypoints | **Google MediaPipe Holistic** | Remplace le CNN |
| Vision | **OpenCV** | Lecture des vidéos frame par frame |
| Data | **NumPy** | Sauvegarde des key-points (.npy) |
| Évaluation | **SacreBLEU** | Métrique de traduction |
| Environnement | **Python 3.9+** | Langage de développement |

---

## 📁 Structure du Projet

```
sign-language-translation/
│
├── 📄 README.md                  ← Ce fichier
├── 📄 requirements.txt           ← Dépendances Python
├── 📄 .gitignore
│
├── 🔬 sanity_check.py            ← ÉTAPE 1 : Valider l'architecture (aucun dataset requis)
├── ⚙️  preprocessing.py           ← ÉTAPE 2 : Extraire les keypoints des vidéos
├── 🚀 fine_tune_cslt.py          ← ÉTAPE 3 : Fine-Tuning du modèle
│
├── 📂 data/
│   ├── raw/                      ← Vidéos brutes How2Sign (.mp4)
│   ├── keypoints/                ← Keypoints extraits (.npy) — générés par preprocessing.py
│   └── annotations/              ← Fichiers .csv des traductions anglaises
│
├── 📂 models/
│   ├── pretrained/               ← Poids SignJoey pré-entraîné sur PHOENIX-2014T (.ckpt)
│   └── finetuned/                ← Poids sauvegardés après le Fine-Tuning
│
└── 📂 outputs/
    └── predictions.txt           ← Traductions générées (pour l'évaluation BLEU)
```

---

## ⚙️ Phase 1 — Data Preprocessing

> **Fichier :** `preprocessing.py`  
> **Dataset requis :** ✅ Vidéos How2Sign

Cette phase est exécutée **une seule fois** sur tout le dataset. Elle transforme les vidéos brutes `.mp4` en tableaux numériques `.npy` qui représentent les mouvements du squelette de la personne.

### Étapes détaillées

**Étape 1 — Nettoyage des vidéos**
- Vérifier que chaque vidéo peut être ouverte (pas corrompue)
- Rejeter les vidéos trop courtes (< 10 frames)
- Convertir BGR → RGB (OpenCV vers MediaPipe)

**Étape 2 — Extraction des Key-points (MediaPipe Holistic)**
- Pour chaque frame : extraire 543 points anatomiques
  - 33 points du corps (épaules, bras, hanches...)
  - 21 points de la main gauche (toutes les articulations des doigts)
  - 21 points de la main droite
  - 468 points du maillage facial (bouche, yeux, sourcils)
- Si un point n'est pas détecté → remplacer par zéros `(0, 0, 0)`
- Format de sortie : `[T, 543, 3]` → aplati en `[T, 1629]`

**Étape 3 — Normalisation**
- Méthode : Z-score (soustraction de la moyenne, division par l'écart-type)
- Objectif : rendre le modèle invariant à la taille du signeur et à la distance caméra

**Étape 4 — Uniformisation des longueurs**
- Longueur fixe : **200 frames** pour toutes les vidéos
- Si trop longue → tronquée à 200 frames
- Si trop courte → complétée avec des zéros (zero-padding)
- Création d'un **masque de padding** (1=donnée réelle, 0=padding)

**Étape 5 — Sauvegarde**
- Un fichier `.npy` par vidéo dans `data/keypoints/`
- Format final : tableau de forme `(200, 1629)` par vidéo

**Étape 6 — Tokenisation du texte**
- Construction d'un vocabulaire de **15 000 mots** les plus fréquents
- Tokens spéciaux : `<PAD>=0`, `<SOS>=1`, `<EOS>=2`, `<UNK>=3`
- Conversion de chaque phrase anglaise en liste d'indices entiers

---

## 🔬 Phase 2 — Sanity Check

> **Fichier :** `sanity_check.py`  
> **Dataset requis :** ❌ Aucun — utilise des données aléatoires

Cette phase valide que l'architecture est correcte **avant** de lancer le vrai entraînement. Elle doit être exécutée en premier.

```bash
python sanity_check.py
```

### Tests effectués

| Test | Description | Condition de succès |
|------|-------------|-------------------|
| **Test 1** — Forward Pass | Vérifie les dimensions de tous les tenseurs | `logits.shape == [4, 50, 15000]` |
| **Test 2** — Overfit 1 Batch | Vérifie que la backpropagation fonctionne | Réduction de loss > 80% en 100 epochs |
| **Test 3** — Freeze des couches | Vérifie que le gel de l'encodeur est actif | `paramètres gelés > 0` |
| **Test 4** — Remplacement classifieur | Vérifie le nouveau classifieur How2Sign | `out_features == 15 000` |

### Résultat attendu
```
✅ RÉUSSI  →  Test 1 — Forward Pass
✅ RÉUSSI  →  Test 2 — Overfit 1 Batch
✅ RÉUSSI  →  Test 3 — Freeze des couches
✅ RÉUSSI  →  Test 4 — Remplacement couche

🎉 ARCHITECTURE VALIDÉE — Vous pouvez lancer le Fine-Tuning !
```

---

## 🚀 Phase 3 — Fine-Tuning du Modèle

> **Fichier :** `fine_tune_cslt.py`  
> **Dataset requis :** ✅ Fichiers `.npy` générés par preprocessing.py

### Stratégie de Fine-Tuning

```
Modèle source  : SignJoey pré-entraîné sur PHOENIX-2014T (DGS)
Modèle cible   : SignJoey fine-tuné sur How2Sign (ASL)
```

| Composant | Stratégie | Learning Rate |
|-----------|-----------|---------------|
| Keypoint Embedding | Entraîné from scratch | `1e-3` |
| Transformer Encoder | ❄️ Gelé (FROZEN) | `0` (pas mis à jour) |
| Transformer Decoder | 🔥 Fine-tuné | `1e-5` |
| Vocabulary Classifier | 🆕 Remplacé + entraîné | `1e-3` |

### Paramètres d'entraînement recommandés
```python
EPOCHS          = 30
BATCH_SIZE      = 16
LEARNING_RATE   = 1e-4    # Pour les couches non gelées
OPTIMIZER       = Adam
LOSS            = CrossEntropyLoss(ignore_index=PAD_IDX)
SCHEDULER       = ReduceLROnPlateau(patience=3)
```

### Lancement
```bash
python fine_tune_cslt.py
```

---

## 📊 Phase 4 — Évaluation

Le modèle est évalué sur un jeu de test (vidéos non vues pendant l'entraînement) à l'aide de la métrique standard en traduction automatique :

### BLEU Score (Bilingual Evaluation Understudy)
- **BLEU-1** : Précision des mots individuels
- **BLEU-4** : Précision des séquences de 4 mots consécutifs
- Score entre 0 et 1 (ou 0 à 100)

| Score BLEU-4 | Interprétation |
|-------------|----------------|
| < 10 | Traduction incompréhensible |
| 10 – 20 | Traduction partielle, idée comprise |
| 20 – 30 | Traduction correcte (standard recherche CSLT) |
| > 30 | Traduction de haute qualité |

> 📌 Les modèles CSLT état de l'art sur PHOENIX-2014T atteignent ~25 BLEU-4.

---

## 💻 Installation & Lancement

### 1. Cloner le dépôt
```bash
git clone https://github.com/amina-dourdi/sign-language-translation.git
cd sign-language-translation
```

### 2. Créer l'environnement virtuel
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / Mac
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Ordre d'exécution des scripts

```bash
# ÉTAPE 1 : Valider l'architecture (sans données)
python sanity_check.py

# ÉTAPE 2 : Extraire les keypoints des vidéos How2Sign
python preprocessing.py

# ÉTAPE 3 : Lancer le Fine-Tuning
python fine_tune_cslt.py
```

---

## 🎯 Résultats Attendus

À la fin du projet, le système doit être capable de :

1. **Prendre en entrée** une vidéo `.mp4` d'une personne signant en ASL
2. **Extraire automatiquement** les key-points du squelette avec MediaPipe
3. **Traduire** la séquence de mouvements en une phrase anglaise cohérente
4. **Afficher** la traduction sur une interface utilisateur (plateforme web)

### Exemple
```
Entrée  : [Vidéo de 3 secondes — personne signant "walk to school"]
Sortie  : "The girl is walking to school"
BLEU-4  : ~22 (objectif du projet)
```

---

## 📚 Références

- **Camgoz et al. (2020)** — *Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation* — [arxiv.org/abs/2003.13830](https://arxiv.org/abs/2003.13830)
- **How2Sign Dataset** — [how2sign.github.io](https://how2sign.github.io)
- **SignJoey (neccam/slt)** — [github.com/neccam/slt](https://github.com/neccam/slt)
- **MediaPipe Holistic** — [google.github.io/mediapipe](https://google.github.io/mediapipe/solutions/holistic.html)
- **PHOENIX-2014T** — [RWTH Aachen University](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/)

---

<div align="center">

**🤟 Continuous Sign Language Translation — PFA 2025/2026**

</div>
