# Predictive Maintenance Model on STM32L4R9

![photo de la carte](https://github.com/user-attachments/assets/dd0d17c9-4b90-44a4-a9f5-c4f6d9447f46)


## 📌 Description générale
Ce projet a pour objectif de concevoir, entraîner et déployer un modèle de réseau de neurones pour une tâche de **maintenance prédictive**, en s’appuyant sur le jeu de données **AI4I 2020 Dataset**.

Le modèle a été entraîné à l’aide de **Google Colab**, puis converti et déployé pour une inférence embarquée sur la carte **STM32L4R9**, via **STM32Cube.AI**.  
Ce projet suit les étapes classiques d’un cycle de développement en IA embarquée : collecte et préparation des données, entraînement d’un modèle, évaluation, quantification et déploiement.

## 🗃️ Données utilisées – AI4I 2020 Dataset
Le fichier `ai4i2020.csv` est un jeu de données de maintenance prédictive composé de 10 000 échantillons simulant le fonctionnement d’équipements industriels. Il inclut :

- Des **données numériques continues** représentant des paramètres de fonctionnement :
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
Dès les premières explorations, nous avons constaté un **déséquilibre massif** dans le dataset : la très grande majorité des échantillons sont étiquetés "No Failure" (absence de panne).

Ce déséquilibre rendait impossible l’entraînement direct d’un modèle pertinent. En effet, un modèle naïf pouvait facilement obtenir plus de 95% de précision simplement en prédisant "pas de panne" tout le temps.

Ce comportement est trompeur, car s’il permet de prédire correctement l’absence de panne, il échoue à identifier précisément le **type** de panne en cas de défaillance — ce qui constitue l’objectif réel du projet.

### 1.2 Nettoyage et filtrage
Pour créer une cible fiable utilisable en classification, nous avons construit une nouvelle colonne `Failure Type`, à partir des 5 colonnes binaires.

Afin d’éviter toute ambiguïté, nous avons filtré le dataset pour ne conserver que :
- Les lignes où **aucune panne n’est présente** (toutes les colonnes TWF à RNF sont à 0), annotées comme "No Failure" ;
- Les lignes où **exactement une seule panne** est active (ex : TWF = 1 et toutes les autres à 0).

Certaines lignes comportaient plusieurs pannes simultanément (ex : TWF = 1 et RNF = 1). Ces cas sont trop peu nombreux pour permettre un apprentissage multi-label efficace, et trop ambigus pour être traités en classification simple. Elles ont donc été exclues.

Ce nettoyage nous a permis d’obtenir un jeu de données propre, avec une cible unique par échantillon, pour un apprentissage **multi-classes à 6 labels** :
`No Failure`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`.

### 1.3 Réflexion autour du multi-label
Nous avons initialement envisagé une approche **multi-label**, dans laquelle le modèle pourrait prédire plusieurs pannes simultanément. Cette idée a été abandonnée pour plusieurs raisons :

- La proportion de lignes contenant plusieurs pannes actives était **très faible**, rendant le signal difficile à apprendre.
- Le passage au multi-label aurait nécessité un changement de stratégie complet :
  - Fonction de perte `binary_crossentropy` au lieu de `categorical_crossentropy`
  - Seuils d’activation à calibrer pour chaque sortie
  - Adaptation de l’architecture embarquée pour interpréter plusieurs sorties actives simultanément

Dans le contexte d’un projet embarqué sur STM32, cela aurait considérablement complexifié le déploiement et la vérification des résultats. Nous avons donc opté pour une classification **multi-classes classique**, plus simple, plus robuste, et surtout **mieux adaptée aux contraintes d’un microcontrôleur**.

## 🔍 Étape 2 – Choix du modèle et architecture

### 2.1 Choix du type de modèle : réseau de neurones (MLP)
Nous avons choisi d’utiliser un **réseau de neurones dense (MLP)** plutôt qu’un algorithme de type Random Forest, SVM ou arbre de décision pour plusieurs raisons :
- **Compatibilité native avec STM32Cube.AI**, qui permet une conversion automatique des architectures Keras vers du code embarqué optimisé
- Capacité des réseaux de neurones à capturer des relations non linéaires dans les données industrielles continues
- Meilleure **portabilité** et contrôle de la taille mémoire par rapport à d’autres modèles plus lourds

Ce choix est également cohérent avec les exemples fournis dans les projets de classification embarquée comme MNIST sur STM32 que nous avons déjà implémenté au préalable.

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

Ces outils nous ont permis d’identifier l’architecture la plus équilibrée entre performance globale et bonne détection des classes rares (comme RNF).

### 2.4 Adaptation à l’embarqué (STM32)
L’architecture a été choisie pour être **légère, compacte et embarquable**, notamment :
- Aucun traitement convolutif ou séquentiel
- Un nombre de paramètres maîtrisé (< 20 000)
- Un seul passage avant prédiction (`feed-forward`) sans complexité algorithmique

## 📉 Étape 3 : Évaluation du modèle
- **Accuracy > 99%**, mais ajustée avec des métriques comme le **recall** par classe.
- Une **matrice de confusion complète 6×6** est générée dans le Colab.
- Chaque classe est correctement identifiée grâce à la correction du déséquilibre et du one-hot encoding.

## 🚀 Étape 4 : Déploiement sur STM32
Le modèle est exporté au format `tflite` (problème de compatibilité avec h5) et importé dans **STM32Cube.AI** via CubeMX :
- Le réseau est converti automatiquement en code C optimisé
- L'inférence est intégrée dans un projet STM32CubeIDE (carte STM32L4R9)
- La communication UART permet d’envoyer les features et de recevoir la prédiction

Cette étape suit une démarche identique à celle de l'exemple MNIST sur STM32 vu précédemment en cours.

## 📊 Étape 5 : Inférence embarquée & test
Le fichier Python `Send_data_stm32.py` pilote la communication :
- Envoie des vecteurs normalisés à la carte
- Lit les 6 scores softmax en retour
- Affiche la classe prédite

Une fonction d’évaluation compare (si disponible) la sortie STM32 à la vérité terrain. Si non, elle affiche simplement les prédictions STM32 pour une inspection manuelle.

## 💧 Comment exécuter le projet

### Partie Python (test STM32)
```bash
python Send_data_stm32.py
```
> Assurez-vous que la carte est branchée, le port correct dans `PORT = "COMx"`.

### Partie Colab (entraînement)
Lancez le notebook `predictive_maintenance_model.ipynb` sur Google Colab ou localement avec Jupyter.

## 📁 Organisation des fichiers

| Fichier | Rôle |
|--------|------|
| `predictive_maintenance_model.ipynb` | Prétraitement, entraînement, évaluation du modèle |
| `Send_data_stm32.py` | Communication UART avec la carte STM32 |
| `model.h5` | Modèle entraîné prêt à être importé dans STM32CubeAI |
| `README.md` | Rapport complet du projet |
| `TP_AI4I2020.ipynb` | Version initiale / alternative du traitement |
| `ai4i2020.csv` | Dataset de maintenance prédictive |

## 📊 Résultats obtenus
- Le modèle embarqué est capable de prédire en temps réel le type de panne
- La latence d’inférence est très faible (quelques ms)
- La communication UART est fiable, avec prédictions correctes
- Le modèle est suffisamment léger pour une exécution fluide sur STM32L4R9

## ✅ Conclusion
Ce projet couvre l’ensemble du cycle :
- De l’analyse de données jusqu’au déploiement embarqué
- Avec une architecture optimisée pour les contraintes d’un microcontrôleur
- Et une précision satisfaisante sur des données industrielles simulées

Il reflète une **intégration complète de l’IA embarquée sur STM32**.

