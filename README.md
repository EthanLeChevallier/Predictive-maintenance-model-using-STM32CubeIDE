# Predictive Maintenance Model on STM32L4R9

## 📌 Description générale
Ce projet a pour objectif de concevoir, entraîner et déployer un modèle de réseau de neurones pour une tâche de **maintenance prédictive**, en s’appuyant sur le jeu de données **AI4I 2020 Dataset**.

Le modèle a été entraîné à l’aide de **Google Colab**, puis converti et déployé pour une inférence embarquée sur la carte **STM32L4R9**, via **STM32Cube.AI**.  
Ce projet suit les étapes classiques d’un cycle de développement en IA embarquée : collecte et préparation des données, entraînement d’un modèle, évaluation, quantification et déploiement.

## 🗃️ Données utilisées – AI4I 2020 Dataset
Le jeu de données fournit des mesures de fonctionnement industriel :
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]
- Cinq types de pannes : TWF, HDF, PWF, OSF, RNF

L’objectif est de prédire s’il y aura une panne, et si oui, laquelle.

## 🔍 Étape 1 : Analyse du jeu de données
Une première analyse via Colab a permis de :
- Vérifier l’absence de valeurs manquantes
- Visualiser la distribution des types de pannes
- Constater un **fort déséquilibre** des classes (ex : très peu de RNF)

Un filtrage a été appliqué pour ne conserver que :
- Les lignes avec **une seule panne active**
- Les lignes sans panne ("No Failure")

## 🧐 Étape 2 : Entraînement du modèle
Le notebook `predictive_maintenance_model.ipynb` contient toutes les étapes :
- Préparation des données
- **Encodage** des étiquettes
- Rééquilibrage avec **SMOTE**
- Normalisation
- Conception d’un MLP avec `Keras` (3 couches, dropout)
- Entraînement avec `categorical_crossentropy`

Le modèle prédit **6 classes** :

1. No Failure  
2. TWF (Tool Wear Failure)  
3. HDF (Heat Dissipation Failure)  
4. PWF (Power Failure)  
5. OSF (Overstrain Failure)  
6. RNF (Random Failure)

Une attention particulière a été portée à l’équilibrage du jeu de données pour améliorer la détection des pannes rares.

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

