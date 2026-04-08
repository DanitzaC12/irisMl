# Documentation du projet irisMl


## Architecture
Ce projet utilise **Scikit-Learn** pour l'entraînement et **MLFlow** pour le suivi.

## Installation
1. Installer les dépendances : `pip install -r requirements.txt`
2. Lancer l'entraînement : `python train.py`

## Suivi des entraînements
Disponibles via l'UI MLflow.  Le modèle final est exporté en `.pkl` dans le dossier `/model`.