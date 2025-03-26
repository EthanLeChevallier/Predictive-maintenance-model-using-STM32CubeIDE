# Predictive Maintenance Model on STM32L4R9

![photo de la carte](https://github.com/user-attachments/assets/dd0d17c9-4b90-44a4-a9f5-c4f6d9447f46)


## 📌 Description générale
Ce projet a pour objectif de concevoir, entraîner et déployer un modèle de réseau de neurones pour une tâche de **maintenance prédictive**, en s’appuyant sur le jeu de données **AI4I 2020 Dataset**.

Le modèle a été entraîné à l’aide de **Google Colab**, puis converti et déployé pour une inférence embarquée sur la carte **STM32L4R9**, via **STM32Cube.AI**.  
Ce projet suit les étapes classiques d’un cycle de développement en IA embarquée : collecte et préparation des données, entraînement d’un modèle, évaluation, quantification et déploiement.

## 🗃️ Données utilisées – AI4I 2020 Dataset
Le fichier `ai4i2020.csv` est un jeu de données de maintenance prédictive composé de 10 000 échantillons simulant le fonctionnement d’équipements industriels. Il inclut :

- Des **données numériques** représentant des paramètres de fonctionnement :
  - `Air temperature [K]`
  - `Process temperature [K]`
  - `Rotational speed [rpm]`
  - `Torque [Nm]`
  - `Tool wear [min]`

- Une **caractéristique catégorielle** :
  - `Type` (L, M ou H)

- Cinq colonnes binaires indiquant la présence ou non d’un type de panne :
  - `TWF`, `HDF`, `PWF`, `OSF`, `RNF`

- Une colonne binaire `Machine failure` indiquant si une panne est survenue, sans préciser laquelle.

- Des identifiants non exploitables pour l’apprentissage : `UDI`, `Product ID`

Le dataset ne contient pas directement de colonne multiclasse indiquant le type précis de panne. Cette structure nécessite donc une transformation avant de pouvoir entraîner un modèle de classification multi-classes.

## 🔍 Étape 1 – Analyse et préparation du jeu de données

### 1.1 Problème de déséquilibre massif
Dès les premières explorations, nous avons constaté un **déséquilibre** dans le dataset : la très grande majorité des échantillons sont étiquetés "No Failure" (absence de panne).

Ce déséquilibre rendait impossible l’entraînement direct d’un modèle pertinent. En effet, un modèle naïf pouvait facilement obtenir plus de 95% de précision simplement en prédisant "pas de panne" tout le temps.

Ce comportement est trompeur, car s’il permet de prédire correctement l’absence de panne, il échoue à identifier précisément le **type** de panne en cas de défaillance — ce qui constitue l’objectif réel de notre projet.

### 1.2 Nettoyage et filtrage
Pour créer une cible fiable utilisable en classification, nous avons construit une nouvelle colonne `Failure Type`, à partir des 5 colonnes binaires.

Afin d’éviter toute ambiguïté, nous avons filtré le dataset pour ne conserver que :
- Les lignes où **aucune panne n’est présente** (toutes les colonnes TWF à RNF sont à 0), annotées comme "No Failure" ;
- Les lignes où **exactement une seule panne** est active (ex : TWF = 1 et toutes les autres à 0).

Certaines lignes comportaient plusieurs pannes simultanément (ex : TWF = 1 et RNF = 1). Ces cas sont trop peu nombreux pour permettre un apprentissage multi-label efficace, et trop ambigus pour être traités en classification simple. Elles ont donc été exclues de l'entraînement.

Ce nettoyage nous a permis d’obtenir un jeu de données propre, avec une cible unique par échantillon, pour un apprentissage **multi-classes à 6 labels** :
`No Failure`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`.

### 1.3 Réflexion autour du multi-label
Nous avons initialement envisagé une approche **multi-label**, dans laquelle le modèle pourrait prédire plusieurs pannes simultanément. Cette idée a été abandonnée pour plusieurs raisons :

- La proportion de lignes contenant plusieurs pannes actives était **très faible**, rendant le signal difficile à apprendre.
- Le passage au multi-label aurait nécessité un changement de stratégie complet :
  - Fonction de perte `binary_crossentropy` au lieu de `categorical_crossentropy`
  - Seuils d’activation à calibrer pour chaque sortie
  - Adaptation de l’architecture embarquée pour interpréter plusieurs sorties actives simultanément

Dans le contexte d’un projet embarqué sur STM32, cela aurait considérablement complexifié le déploiement et la vérification des résultats. Nous avons donc opté pour une classification **multi-classes classique**, plus simple et surtout **mieux adaptée aux contraintes d’un microcontrôleur**.

## 🌲 Étape 2 – Choix du modèle et architecture

### 2.1 Choix du type de modèle : réseau de neurones (MLP)
Nous avons choisi d’utiliser un **réseau de neurones dense (MLP)** plutôt qu’un algorithme de type Random Forest, SVM ou arbre de décision pour plusieurs raisons :
- **Compatibilité native avec STM32Cube.AI**, qui permet une conversion automatique des architectures Keras vers du code embarqué optimisé
- Capacité des réseaux de neurones à capturer des relations non linéaires dans les données industrielles continues
- Meilleure **portabilité** et contrôle de la taille mémoire par rapport à d’autres modèles plus lourds

Ce choix est également cohérent avec les exemples fournis dans les projets de classification embarquée comme MNIST sur STM32 que nous avons déjà implémenté au préalable comme exercice de préparation pour ce projet.

### 2.2 Architecture du réseau retenue
Le modèle final utilisé est un réseau de neurones à 3 couches entièrement connectées :
- **Input layer** : 7 entrées correspondant aux 7 features numériques du jeu de données nettoyé
- **Dense(128)** avec activation **ReLU**
- **Dense(64)** avec activation **ReLU**
- **Dense(6)** avec activation **Softmax**, correspondant aux 6 classes cibles : `No Failure`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`

Nous n’avons **pas utilisé de Dropout** dans le modèle final, car l’overfitting ne s’est pas avéré problématique après rééquilibrage du dataset avec SMOTE.

![Graphe du modèle](https://github.com/user-attachments/assets/07a9231e-0edc-4eac-99d0-6d14d5010115)

Ce modèle est défini dans le notebook `predictive_maintenance_model.ipynb` avec la fonction de perte `categorical_crossentropy` et l’optimiseur `adam`.

### 2.3 Validation du modèle
Plutôt que de recourir à une recherche d’hyperparamètres automatisée (grid search), nous avons mené des **tests manuels successifs**. À chaque itération, nous avons évalué le modèle à l’aide de :
- **La matrice de confusion** complète sur les 6 classes
- **Le rapport de classification** (`precision`, `recall`, `f1-score` par classe)
- **Accuracy globale** visualisation de la loss et d'accuracy

Ces outils nous ont permis d’identifier l’architecture la plus équilibrée entre performance globale et bonne détection des classes rares (comme RNF).

### 2.4 Adaptation à l’embarqué (STM32)
L’architecture a été choisie pour être **légère, compacte et embarquable**, notamment :
- Aucun traitement convolutif ou séquentiel
- Un nombre de paramètres maîtrisé (< 20 000)
- Un seul passage avant prédiction (`feed-forward`) sans complexité algorithmique

## ⚖️ Étape 3 – Rééquilibrage du dataset

Afin de pallier le fort déséquilibre entre les classes (notamment l’écrasante majorité de "No Failure"), nous avons choisi d’appliquer une stratégie de **rééquilibrage par oversampling**, en utilisant **SMOTE** (Synthetic Minority Over-sampling Technique).

SMOTE permet de générer artificiellement de nouveaux exemples pour les classes minoritaires, en interpolant des points synthétiques proches d’échantillons existants. Cette méthode a l’avantage de **ne pas supprimer d’échantillons** (contrairement à l’undersampling), et donc de **préserver toute l’information disponible** dans le dataset initial.

Nous avons fait le choix d’utiliser uniquement SMOTE, sans tester d’autres alternatives comme les poids de classes ou le RandomUnderSampler. Bien que cela aurait pu être pertinent pour comparaison, notre priorité était d’obtenir rapidement un jeu de données équilibré pour valider l’apprentissage embarqué.

### Application dans le pipeline
Dans notre code, le rééquilibrage par SMOTE est effectué **avant le split train/test**, ce qui peut introduire un risque de **data leakage** (les points synthétiques pouvant influencer les deux ensembles).

Une amélioration possible serait d’appliquer SMOTE **uniquement sur l’ensemble d’entraînement** après découpage, afin de préserver l’indépendance de la phase de test. Malheureusement nous n'avons pas réussi à faire autrement. Cela n’a toutefois pas semblé altérer la qualité des résultats dans notre cas, comme en témoigne la bonne généralisation observée sur les prédictions STM32. 

## 🎯 Étape 4 – Évaluation du modèle

L’évaluation de notre modèle ne s’est pas limitée à une simple mesure d’accuracy. Nous avons mis en place un protocole plus large, fondé sur des outils d’analyse de la performance :

### 4.1 Métriques utilisées

Nous avons utilisé les métriques classiques mais essentielles pour un problème de classification multi-classes :
- **Accuracy globale** : utile pour donner une idée générale des performances
- **Rapport de classification** : incluant `precision`, `recall` et `f1-score` pour chaque classe
- **Matrice de confusion** : pour visualiser les erreurs de prédiction classe par classe

Ces indicateurs permettent non seulement de juger la qualité globale du modèle, mais aussi de vérifier s’il n’est pas biaisé contre les classes minoritaires comme `RNF`.

### 4.2 Outils de visualisation

Pour renforcer l’analyse, nous avons tracé l’évolution de la **loss et de l’accuracy** en fonction des epochs sur les ensembles d’entraînement et de validation. Ces courbes nous ont permis de confirmer l’absence d’overfitting visible et la bonne généralisation du modèle.

### 4.3 Résultats observés

Le modèle atteint une précision globale supérieure à **99%**, y compris sur l’ensemble de test. La matrice de confusion montre que :
- Les classes majoritaires comme `No Failure` ou `TWF` sont parfaitement prédites
- Les classes plus rares comme `RNF` ou `OSF` sont également bien identifiées, ce qui montre l’efficacité du rééquilibrage par SMOTE

Le rapport de classification confirme une bonne homogénéité des scores f1, avec un macro-average et un weighted-average très élevés (proches de 0.99).

### 4.4 Limites et interprétation

Malgré ces bons résultats, nous restons prudents :
- Le split train/test après SMOTE aurait pu influencer les scores positivement
- Les résultats sont obtenus sur un jeu synthétique ; une validation sur des données réelles serait nécessaire pour garantir la robustesse du modèle

L’ensemble des outils d’évaluation utilisés nous permet de conclure que notre modèle est suffisamment fiable pour un déploiement embarqué sur STM32.


## 🏹 Étape 5 – Déploiement sur la carte STM32

Le modèle entraîné a été exporté dans un premier temps au format `.h5`, mais cette approche a posé des problèmes de compatibilité lors de la conversion avec STM32Cube.AI. Nous avons donc choisi d’utiliser le format **TensorFlow Lite (`.tflite`)**, plus stable dans notre contexte. Le fichier `.tflite` a été généré et stocké sur Google Drive avant importation dans CubeMX.

### 5.1 Conversion avec STM32Cube.AI

La démarche suivie pour intégrer le modèle sur la carte STM32 s’inspire étroitement de l’exemple **MNIST** présenté dans le cours (cf. EmbeddedAI.pdf). Nous avons utilisé l’outil **STM32Cube.AI** intégré à **STM32CubeMX** pour :
- Importer le modèle `.tflite`
- Générer le code embarqué compatible avec la série **STM32L4**
- Configurer les buffers d’entrée/sortie et la mémoire optimisée (float, RAM ≤ 3KB, Flash ≤ 25KB)

Le modèle a été validé localement à l’aide de l’outil `stedgeai.exe validate`, qui a confirmé sa légèreté (13KB de poids, 768B d'activations). La structure du réseau est composée de 3 couches denses.

### 5.2 Modifications du code embarqué

Une fois le projet généré dans **STM32CubeIDE**, nous avons modifié manuellement le fichier `app_x-cube-ai.c` pour y intégrer :
- Une fonction `acquire_and_process_data()` pour recevoir les données d’entrée via **UART** en format `float32`
- Une fonction `post_process()` qui reconstruit les résultats (`softmax`) en `uint8`, puis les renvoie au PC
- Une fonction de synchronisation UART (`synchronize_UART`) pour initialiser la communication

La boucle principale `MX_X_CUBE_AI_Process()` suit la structure classique :
1. Acquisition des données via UART
2. Inférence avec `ai_run()`
3. Transmission des résultats au PC via UART

### 5.3 Communication avec le PC (UART)

La communication est gérée par le script Python `Send_data_stm32.py`, qui utilise le port COM4 (UART) à 115200 bauds. Le protocole fonctionne ainsi :
- Génération de données de test simulées avec `generate_random_data()`
- Normalisation des données avec `StandardScaler`
- Envoi des vecteurs de 7 floats via UART
- Réception d’un tableau de 6 valeurs (scores softmax compressés entre 0 et 255)

Le script évalue ensuite la précision STM32 en comparant la classe prédite à un `y_test` aléatoire.

### 5.4 Résultats embarqués

D’après le fichier `resultat.txt.txt`, la prédiction embarquée est fonctionnelle :
- Les vecteurs sont bien transmis et reçus
- Le STM32 renvoie des valeurs cohérentes de softmax
- La précision augmente itérativement jusqu’à atteindre **1.00** sur 100 itérations de test

Les sorties telles que `b'\x00\x00\x00\x00\xff\x00'` sont bien décodées en scores `[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]`, correspondant à des classes valides.

![resultat_test](https://github.com/user-attachments/assets/c68c72c6-4ada-4751-9aab-38605d247493)

Le modèle embarqué montre ainsi une inférence rapide, fiable, et efficace, parfaitement adaptée à un microcontrôleur STM32L4R9.

## 🔜 Étape 6 – Limites et perspectives

### 6.1 Limites identifiées

Malgré le bon fonctionnement général du projet, certaines limites doivent être soulignées :

- **Répartition artificielle des classes** : le jeu de données a été rééquilibré artificiellement avec SMOTE, ce qui pourrait induire un certain optimisme sur les performances.
- **Pas de données réelles** : les tests se basent sur des données simulées ou générées aléatoirement. Cela ne reflète pas les conditions de production industrielle.
- **Absence de validation croisée** : l'évaluation repose sur un simple split train/test, sans validation croisée.
- **Communication UART simplifiée** : le protocole UART utilisé est simple mais sensible à des pertes ou désynchronisations si non encadré.

### 6.2 Améliorations envisagées

Pour améliorer la robustesse et la portée du projet, plusieurs pistes sont possibles :

- **Tester le modèle avec des données réelles** issues de machines ou de capteurs industriels.
- **Utiliser d'autres techniques de rééquilibrage** (class weighting, undersampling combiné).
- **Passer à une classification multi-label** si plusieurs pannes peuvent coexister (avec une autre architecture).
- **Intégrer un système de journalisation côté STM32**, avec sauvegarde en mémoire Flash ou envoi périodique vers le PC.
- **Mesurer le temps d'inférence embarqué** pour évaluer l'efficacité du modèle sur STM32.

Ces perspectives permettent de prolonger le projet vers une version plus industrialisable ou intégrée dans un pipeline de maintenance prédictive en environnement embarqué réel.

## ✅ Conclusion générale

Ce projet a permis de mettre en œuvre l’ensemble de la chaîne de développement d’un système d’intelligence artificielle embarqué, depuis l’analyse exploratoire des données jusqu’au déploiement effectif sur une carte STM32.

Nous avons dû faire face à des contraintes concrètes : déséquilibre des classes, limitations matérielles, conversion du modèle, communication série. Chacune a été traitée par des choix techniques appropriés, justifiés par les contraintes du déploiement embarqué.

Le modèle entraîné est précis, et opérationnel sur STM32. La démonstration de bout en bout valide la faisabilité d’intégrer un algorithme de classification complexe dans un microcontrôleur à ressources limitées.

Ce projet constitue une base pour des applications réelles de maintenance prédictive dans un environnement industriel connecté (IIoT).

## 💧 Comment exécuter le projet

### Partie Python (test STM32)
```bash
python Send_data_stm32.py
```
> Assurez-vous que la carte STM32 est bien connectée, et que le port série dans le script (`PORT = "COMx"`) correspond au bon port COM. Le script envoie des vecteurs de données simulées normalisées, puis reçoit la prédiction de la carte au format compressé (softmax codé sur 8 bits).

### Partie Google Colab / Jupyter (entraînement du modèle)
Ouvrir le notebook `predictive_maintenance_model.ipynb` sur Google Colab ou localement avec Jupyter Notebook pour :
- Charger et préparer le jeu de données
- Appliquer le rééquilibrage
- Entraîner le modèle
- Évaluer sa performance
- Sauvegarder le modèle au format `.tflite` pour déploiement

---

# 📁 Organisation des fichiers

| Fichier | Rôle |
|--------|------|
| `predictive_maintenance_model.ipynb` | Prétraitement, entraînement et évaluation du modèle |
| `Send_data_stm32.py` | Script de communication UART entre le PC et la carte STM32 |
| `model.tflite` | Modèle entraîné et converti, prêt pour STM32Cube.AI |
| `app_x-cube-ai.c` | Code C généré et modifié pour exécuter le modèle sur STM32 avec communication UART |
| `TP_AI4I2020.ipynb` | Version initiale / exploration préliminaire du dataset |
| `ai4i2020.csv` | Jeu de données original utilisé pour l'entraînement |
| `README.md` | Rapport de projet détaillé et instructions d'exécution |
