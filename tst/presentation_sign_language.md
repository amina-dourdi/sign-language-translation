# Préparation de la Présentation : Projet de Traduction de la Langue des Signes

Ce document suit exactement le plan demandé pour votre présentation. Les explications sont vulgarisées pour être comprises par un jury qui ne serait pas expert en Deep Learning.

---

## 1. Introduction
**La Problématique :**
Aujourd'hui, il existe une grande barrière de communication entre les personnes sourdes ou malentendantes et le reste du monde. La majorité des gens ne comprennent pas la langue des signes, ce qui crée une exclusion au quotidien (hôpitaux, administrations, etc.).

**L'importance du sujet :**
Briser cette barrière est un enjeu d'inclusion sociale majeur. Il ne s'agit pas seulement de comprendre un geste isolé (comme dire "Bonjour"), mais de comprendre de vraies phrases complètes et fluides.

**La nécessité d'une solution :**
Il est physiquement impossible d'avoir un interprète humain disponible partout, 24h/24. Nous avons donc besoin d'un "Google Traduction" visuel : un système capable de "regarder" une personne signer et de traduire immédiatement ses gestes en texte compréhensible par tous.

---

## 2. Solution proposée
Notre solution est un système de **Traduction Continue de la Langue des Signes (CSLT)**. 

Concrètement, plutôt que d'analyser une vidéo très lourde (qui ferait ramer l'ordinateur), notre système ne regarde que le "squelette" de la personne (la position de ses mains, de son corps et de son visage). À partir de ces mouvements en temps réel, notre intelligence artificielle ne va pas juste traduire du "mot-à-mot" comme un robot, mais va générer une phrase grammaticalement correcte dans une langue parlée (comme l'allemand ou l'anglais).

---

## 3. Dataset (La Base de Données)
**Nom :** PHOENIX-2014-T.
C'est l'une des bases de données les plus connues dans ce domaine. Elle est constituée d'enregistrements de la météo en langue des signes allemande.

**La structure de nos données :**
Pour entraîner notre modèle plus efficacement, nous n'utilisons pas les pixels des vidéos brutes. Nous utilisons des **"Keypoints"** (points clés). 
Imaginez une combinaison de motion-capture de cinéma : pour chaque image de la vidéo, nous avons 399 coordonnées mathématiques (X, Y, Z) qui représentent exactement où se trouvent le bout des doigts, les coudes et les expressions du visage du signeur. 
Notre base de données contient donc des séquences de ces points mathématiques, associées à la phrase texte correspondante que le modèle doit apprendre à deviner.

---

## 4. Architecture
Nous avons construit une architecture sur mesure en utilisant la technologie des "Transformers" (la même technologie qui fait fonctionner ChatGPT).

**Notre modèle a deux grands "cerveaux" :**
1. **Un Encodeur (Le module de Reconnaissance) :** C'est la partie qui "regarde". Il analyse les 399 points du squelette qui bougent dans le temps pour comprendre quels gestes sont faits.
2. **Un Décodeur (Le module de Traduction) :** C'est la partie qui "parle". Il prend l'information visuelle comprise par l'encodeur, et construit une phrase mot par mot avec la bonne grammaire. Pour choisir le meilleur mot, il utilise une technique appelée **Beam Search** : au lieu de dire le premier mot qui lui vient à l'esprit, il "réfléchit" à 5 fins de phrases possibles en même temps et garde la plus logique.

**Justification du modèle (Transfer Learning) :**
Plutôt que d'apprendre à notre modèle à lire de zéro, nous avons utilisé une technique appelée le **Transfer Learning (Apprentissage par transfert)**. Nous avons pris les "poids" (les connaissances) d'un ancien modèle qui savait déjà bien reconnaître des gestes isolés, et nous les avons insérés dans notre nouveau modèle. Cela nous permet de gagner un temps précieux et d'avoir un modèle beaucoup plus intelligent dès le départ.

---

## 5. Évaluation
Pour évaluer notre IA, nous ne pouvons pas juste demander "As-tu deviné le bon mot ?". La traduction de phrases entières est plus complexe. Nous utilisons deux métriques principales :

1. **Le Score BLEU (Notre métrique principale) :**
   *C'est quoi ?* C'est la référence mondiale pour évaluer les traductions automatisées. 
   *Pourquoi ce choix ?* Le score BLEU vérifie si les groupes de mots générés par notre IA ressemblent aux phrases traduites par un humain. Il vérifie la grammaire, la fluidité et l'ordre des mots, pas juste le vocabulaire.
2. **Le WER (Word Error Rate - Taux d'erreur par mot) :**
   *C'est quoi ?* Il compte combien de mots le modèle a oublié, rajouté par erreur, ou confondu.
   *Pourquoi ce choix ?* Il nous permet de voir si l'encodeur fait bien son travail de reconnaissance de base.

*(Note technique : Pour que le modèle apprenne de ses erreurs, nous utilisons une perte conjointe "Joint Loss" qui combine le CTC, pour aligner le geste avec le mot, et le Label Smoothing, pour empêcher l'IA d'être trop arrogante sur ses traductions).*

---

## 6. Démonstration
Pour clôturer, nous allons vous montrer le système en action. 
Nous allons envoyer à notre modèle une séquence de mouvements (squelette) qu'il n'a jamais vue pendant son entraînement. Vous verrez comment le système "avale" ces points mathématiques, les fait passer dans ses couches d'attention (Transformer), et affiche instantanément la traduction finale sous forme de texte (en Allemand, puis traduit en Anglais).
