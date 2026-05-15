# 🤟 Rapport de Projet : Traducteur Automatique de Langue des Signes (CSLT)
### Master Ingénierie des Données - Amina Dourdi

---

## 1. 🌟 Le Concept : Qu'avons-nous fait ?
Le but est de transformer une vidéo de quelqu'un qui parle la langue des signes en une phrase écrite en anglais. Pour faire cela, nous avons suivi 3 grandes étapes :

1.  **Simplifier l'image :** Au lieu de donner toute la vidéo à l'ordinateur (trop lourd), on utilise une technologie (MediaPipe) qui ne garde que les **points du squelette** (mains, visage, corps). C'est beaucoup plus léger.
2.  **Créer un Cerveau (Le Modèle) :** On a construit un programme d'Intelligence Artificielle capable de "lire" ces points.
3.  **Entraîner le Cerveau :** On lui a montré des milliers d'exemples pour qu'il apprenne.

---

### A. Les étapes de transformation du signal
Pour que les points du squelette deviennent des mots, ils passent par plusieurs couches spécialisées :

1.  **La Projection Linéaire :** On reçoit 411 coordonnées par frame. Cette couche les transforme en un "vecteur de caractéristiques" (dimension 256). C'est comme si on traduisait les positions brutes en un langage que l'IA comprend mieux.
2.  **Le CNN Temporel (1D) :** C'est une couche de convolution. Son rôle est de regarder les frames voisines (le mouvement sur 1 ou 2 secondes) pour détecter la **dynamique**. Elle comprend si une main "accélère" ou "s'arrête".
3.  **L'Encodage Positionnel :** Comme le Transformer regarde toute la vidéo d'un coup, il ne sait pas naturellement quel geste vient avant l'autre. On ajoute une "étiquette de temps" à chaque frame pour qu'il connaisse l'ordre chronologique.

### B. L'ENCODER (L'analyse contextuelle)
L'Encoder utilise le mécanisme de **Multi-Head Attention** (Attention à têtes multiples).
- Il compare chaque geste à tous les autres gestes de la vidéo.
- Il décide quelle partie de la vidéo est la plus importante pour comprendre le sens global.
- À la fin, il produit une "représentation riche" de toute la phrase signée.

### C. LE DECODER (La génération de la phrase)
Le Decoder est "autorégressif" : il génère le texte mot par mot.
- Il regarde ce qu'il a déjà écrit (ex: "The") pour décider du mot suivant (ex: "cat").
- Il utilise une **Attention Croisée (Cross-Attention)** : il jette un coup d'œil permanent aux notes de l'Encoder pour s'assurer qu'il ne s'éloigne pas du sens de la vidéo originale.
- Enfin, une couche **Linear Head** choisit le mot final parmi les 10 000 mots possibles de son vocabulaire.
- Il sait quel mouvement de main est le plus important pour comprendre le mot.

---

## 3. 💾 Les Poids Pré-entraînés (`best_model_v3.pth`) : C'est quoi ?
C'est une question très importante. Pourquoi ne pas commencer de zéro ?

- **Apprendre à marcher avant de courir :** Un modèle d'IA au début est "bébé", il ne sait rien.
- **Le Pre-training :** Nous avons d'abord entraîné notre modèle sur 30 000 vidéos pour qu'il apprenne simplement à reconnaître ce qu'est une "main" ou un "mouvement de bras". C'est ce qu'on appelle les **poids pré-entraînés**.
- **Le fichier `best_model_v3.pth` :** C'est le cerveau de notre modèle après cet entraînement. Il contient déjà une "connaissance de base".
- **Le Fine-Tuning (Colab) :** Aujourd'hui, on prend ce cerveau déjà "intelligent" et on lui apprend spécifiquement à faire des phrases parfaites.

---

## 4. 📈 L'Apprentissage : Comment devient-il meilleur ?
Pendant l'entraînement sur Google Colab, le modèle utilise deux outils mathématiques :

1.  **La fonction de perte (Loss) :** C'est le "professeur". Chaque fois que le modèle traduit un mot de travers, la Loss augmente. Le but du modèle est de rendre cette perte la plus petite possible.
2.  **L'Optimiseur (AdamW) :** C'est le "mécanisme de correction". Dès que le professeur (la Loss) signale une erreur, l'Optimiseur modifie légèrement les connexions du cerveau (les neurones) pour ne plus refaire la même erreur la prochaine fois.

---

## 5. 📏 L'Évaluation : Le Score BLEU
Comment savoir si une traduction est bonne ? On utilise le **Score BLEU** (Bilingual Evaluation Understudy).
- Il compare la phrase générée par l'IA avec la phrase réelle écrite par un humain.
- Plus le score est proche de 100, plus la traduction est parfaite. 
- *Note : En langue des signes, un score de 10 à 15 est déjà considéré comme un exploit technique !*

---

## 6. 🔄 Résumé des étapes de ton projet (Le Workflow)

1.  **Pre-processing (PC Local) :** On transforme les vidéos en fichiers de points (`.npy`). C'est la préparation de la nourriture avant de cuisiner.
2.  **Transfert Cloud (Drive) :** On envoie tout sur Google Drive pour utiliser la puissance des serveurs de Google (GPU).
3.  **Fine-Tuning (Google Colab) :** C'est l'entraînement final. Le modèle devient un expert en traduction.
4.  **Inférence (Webcam) :** C'est le résultat final où tu montres tes mains à la caméra et le texte s'affiche !

---

## 7. 📁 Guide des Fichiers : Où se trouve le code ?

Voici l'organisation de ton projet pour t'y retrouver facilement :

### A. La Gestion des Données
- **`preprocess_all_splits.py`** : C'est ici qu'on transforme les vidéos How2Sign en points clés (`.npy`). C'est le script qui utilise le **Multiprocessing** pour aller vite.
- **`prepare_for_colab.py`** : Ce petit utilitaire zippe tes fichiers pour les envoyer proprement sur Google Drive.

### B. Le "Cerveau" et l'Entraînement
- **`train_colab.py`** : C'est le fichier **le plus important**. Il contient la définition de l'architecture (`CSLTModel`), l'Encoder, le Decoder, et le Tokenizer (le dictionnaire). C'est lui qui a créé les poids de base `v3`.
- **`finetune_colab.py`** : C'est le script que tu lances sur Google Colab. Il charge le modèle de base et le spécialise pour obtenir une traduction parfaite.

### C. La Démonstration et l'Inférence
- **`demo_webcam.py`** : C'est le script que tu lances sur ton PC pour faire la démo en direct. Il connecte ta webcam au modèle entraîné.
- **`demo_external_model.py`** : Ce script sert à montrer la différence avec un modèle externe (MediaPipe Gesture Recognizer) qui ne reconnaît que des mots isolés.

---

## 8. 🛠️ Les technologies clés
- **MediaPipe :** L'outil qui dessine le squelette sur tes mains.
- **PyTorch :** Le moteur qui permet de construire et d'entraîner le cerveau (Encoder/Decoder).
- **FastAPI :** Le serveur qui fait le lien entre ta webcam et l'IA.

---
### 💡 Ce qu'il faut dire à ton oral :
*"Mon projet utilise une architecture **Transformer Encoder-Decoder**. L'Encoder analyse les points clés extraits par **MediaPipe** pour comprendre le mouvement, tandis que le Decoder génère la phrase en anglais. Nous utilisons le **Transfer Learning** à partir de poids pré-entraînés (`best_model_v3`) pour garantir une meilleure précision et un entraînement plus rapide."*

**Réalisé par Amina Dourdi - 2026**
