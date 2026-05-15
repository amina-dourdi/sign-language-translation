# 🗣️ Script pour l'oral : Bloc 1 (Data Pipeline & Inputs)

*Ce document contient votre texte exact à prononcer. Les formules mathématiques ont été intégrées naturellement pour que le discours soit fluide mais très scientifique.*

---

*(Transition avec la personne précédente)*
"Je vais maintenant vous présenter la première étape cruciale de notre architecture : **Le traitement des données et les entrées (Data Pipeline & Inputs).**

Comme vous pouvez le voir sur la gauche du schéma, tout commence par une vidéo brute en langue des signes. Cependant, injecter directement tous les pixels d'une vidéo dans un réseau de neurones serait beaucoup trop lourd en calcul et inefficace pour l'apprentissage. 

### [1. L'Extraction des Keypoints avec MediaPipe]
Pour résoudre ce problème, nous utilisons la technologie **MediaPipe**. Au lieu de traiter l'image, nous en extrayons un 'squelette 3D' composé de points clés. 

Mathématiquement, pour chaque vidéo, nous construisons une matrice d'entrée $X$. Si la vidéo dure $T$ séquences (ou frames), notre matrice appartient à l'espace $\mathbb{R}^{T \times 399}$. 
Le chiffre 399 n'est pas aléatoire : c'est la concaténation spatiale des vecteurs 3D $(x, y, z)$ pour nos 133 points clés. Nous avons séparé les éléments essentiels à la langue des signes :
*   **70 points pour le Visage**, pour capter les expressions faciales.
*   **42 points pour les Mains**, pour la forme des gestes.
*   **Et 21 points pour la posture du Corps**.
Nous obtenons donc un tenseur léger, ciblé sur le mouvement humain.

### [2. L'Augmentation de Données (Data Augmentation)]
Ensuite, pour éviter que notre modèle n'apprenne les données d'entraînement par cœur (le problème de *surapprentissage* ou *overfitting*), nous appliquons trois techniques d'**Augmentation de Données**, uniquement pendant la phase d'entraînement :

1.  **Le Temporal Resampling :** Nous modifions aléatoirement la vitesse de la vidéo pour simuler des personnes qui signent plus ou moins vite.
2.  **Le Bruit Gaussien (Gaussian Noise) :** À notre tenseur de coordonnées $X$, nous ajoutons une matrice d'erreur tirée d'une distribution normale, centrée sur zéro. La formule est $X_{augmenté} = X + \mathcal{N}(0, \sigma^2)$. Cela force le réseau à apprendre une fonction robuste aux imprécisions de la caméra.
3.  **La Rotation Aléatoire :** Nous appliquons des matrices de rotation 3D au squelette pour simuler différents angles de vue.

### [3. Normalisation et Padding]
Enfin, la dernière étape de mon pipeline est la **Normalisation et le Padding**. 

Les cartes graphiques (GPU) ont besoin de traiter des tenseurs strictement rectangulaires pour multiplier efficacement les matrices. Parce que nos vidéos n'ont pas toutes la même durée $T$, nous appliquons un masque de padding. Si une vidéo est trop courte, nous la remplissons avec des vecteurs de zéros jusqu'à atteindre notre constante de longueur maximale, soit 150 frames. 

Notre tenseur final, qui part dans l'encodeur, est donc strictement fixé à la dimension $\mathbb{R}^{150 \times 399}$, ce qui garantit une parallélisation parfaite des calculs matriciels.

*(Transition)*
Une fois que ces données pures, robustes et normalisées sont prêtes, elles sont envoyées dans notre Encodeur MSKA pour être analysées. Je laisse maintenant la parole à *(Nom de votre collègue)* qui va vous expliquer comment le modèle déchiffre ces mouvements."
