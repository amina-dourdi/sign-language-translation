# Explication Mathématique de l'Architecture MSKA (Niveau Ingénieur / Master)

Ce document est la version "Mathématiques Pures" de votre diagramme d'architecture. Si votre jury (surtout des professeurs de mathématiques ou d'informatique théorique) vous demande *comment ça marche à l'intérieur*, voici les équations exactes.

---

## 1. Data Pipeline & Spatial Embedding (Blocs 1 et 2 Haut)

### Entrée Initiale (Input)
L'entrée est une matrice (tenseur) représentant la vidéo entière.
Soit $X \in \mathbb{R}^{T \times 399}$, où $T$ est le nombre de frames (images) de la vidéo.

### Découpage (Partitioning)
Le tenseur est scindé en 3 sous-tenseurs selon les points clés :
*   $X_{face} \in \mathbb{R}^{T \times 210}$
*   $X_{hands} \in \mathbb{R}^{T \times 126}$
*   $X_{body} \in \mathbb{R}^{T \times 63}$

### Transformation Spatiale (Spatial Embedding)
Chaque sous-tenseur passe par son propre réseau de neurones (Multi-Layer Perceptron) composé de couches linéaires et d'activations `ReLU` ($\max(0, x)$).
Par exemple, pour les mains :
$$H_{hands} = \text{ReLU}(X_{hands} W_{h1} + b_{h1}) W_{h2} + b_{h2}$$
*(où $W$ représente la matrice de poids et $b$ le biais).*

Les 3 flux sont ensuite concaténés (collés) :
$$F = [H_{face} \oplus H_{hands} \oplus H_{body}] \in \mathbb{R}^{T \times 512}$$
On applique un Dropout final et une couche linéaire pour obtenir le **Spatial-Temporal Fusion** : $S \in \mathbb{R}^{T \times 512}$.

---

## 2. MSKA Transformer Encoder (Bloc 2 Bas)

### Positional Encoding
Le Transformer ne comprenant pas le temps, on ajoute une onde mathématique à la matrice $S$.
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$
L'entrée de l'encodeur devient : $E_{in} = S + PE$.

### Multi-Head Self-Attention
C'est le cœur du système. Pour chaque frame, le modèle crée 3 vecteurs : Query ($Q$), Key ($K$) et Value ($V$) en multipliant $E_{in}$ par des matrices de poids apprenables.
La matrice d'attention se calcule ainsi :
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
Cette équation calcule le "score de corrélation" entre chaque geste de la vidéo (le produit scalaire $QK^T$) et pondère l'information finale ($V$).
La sortie de l'encodeur est une matrice riche en contexte : **$Z_{enc} \in \mathbb{R}^{T \times 512}$**.

---

## 3. CTC Head & Loss (La Branche de Reconnaissance)

L'encodeur doit deviner les glosses (les signes bruts). On projette $Z_{enc}$ vers la taille du vocabulaire :
$$L_{ctc} = Z_{enc} W_{ctc} + b_{ctc} \in \mathbb{R}^{T \times |V|}$$

### La Mathématique du CTC (Connectionist Temporal Classification)
La magie du CTC est de pouvoir s'entraîner sans savoir exactement à quelle frame correspond quel mot. Le CTC introduit un jeton "Blank" ($\epsilon$).
La probabilité de générer la vraie séquence de mots $Y^*$ sachant la vidéo $X$ est la somme des probabilités de tous les chemins d'alignement valides $\pi$ :
$$P(Y^* | X) = \sum_{\pi \in \mathcal{B}^{-1}(Y^*)} \prod_{t=1}^T P(\pi_t | x_t)$$
Où $\mathcal{B}$ est la fonction qui supprime les doublons consécutifs et les "Blanks".
La Loss CTC est simplement l'opposé du logarithme de cette probabilité :
$$\mathcal{L}_{CTC} = -\ln P(Y^* | X)$$

---

## 4. Linguistic Mapping (VL Mapper)

Le VL Mapper est un perceptron avec normalisation de couche (LayerNorm). Il prend la matrice visuelle $Z_{enc}$ et la convertit en **Mémoire ($M$)** pour le décodeur textuel.
$$M = \text{Dropout}\left(\text{ReLU}(\text{LayerNorm}(Z_{enc}) W_{vlm} + b_{vlm})\right)$$

---

## 5. Translation Decoder & Output (Bloc 3)

### Masked Self-Attention (Causal Mask)
Soit $Y_{in}$ la séquence des mots cibles déjà générés. On calcule l'attention entre les mots, mais on ajoute un masque mathématique $-\infty$ sur le futur pour empêcher le modèle de tricher :
$$\text{MaskedAttention} = \text{Softmax}\left(\frac{Q_{text} K_{text}^T}{\sqrt{d_k}} + \text{Mask}\right) V_{text}$$
*(L'exponentielle de $-\infty$ valant $0$, la probabilité de regarder un mot du futur devient strictement $0$).*

### Cross-Attention
C'est ici que l'information vidéo croise l'information texte.
*   La Query ($Q$) vient de la phrase en cours de génération (le texte).
*   Les Keys ($K$) et Values ($V$) viennent de la **Mémoire ($M$)** vidéo du VL Mapper.
Le décodeur se demande : *"Étant donné les mots que j'ai déjà dits ($Q$), quelle partie de la vidéo ($K, V$) dois-je regarder pour deviner le prochain mot ?"*

---

## 6. L'Optimisation Globale (Loss & Backward)

### Label Smoothing Loss
Au lieu d'utiliser la Cross-Entropy classique, on lisse les probabilités cibles. Soit $\alpha$ le facteur de lissage (ex: $0.1$), et $K$ le nombre total de mots dans le dictionnaire. 
La probabilité cible (qui était $1.0$ pour le bon mot et $0$ pour les autres) devient :
$$y_{cible} = 1 - \alpha$$
$$y_{autres} = \frac{\alpha}{K - 1}$$
La perte finale de traduction est :
$$\mathcal{L}_{Trans} = - \sum_{i=1}^K y_i \log(p_i)$$

### Fonction de Perte Multi-Tâches Finale (Total Objective)
Pendant la backpropagation, la carte graphique calcule les gradients par rapport à cette équation finale :
$$\mathcal{L}_{Totale} = \mathcal{L}_{Trans} + (\lambda \times \mathcal{L}_{CTC})$$
Où $\lambda$ (lambda_ctc) est un poids (ex: 0.2) que vous avez défini pour équilibrer l'importance de la reconnaissance par rapport à la traduction.
