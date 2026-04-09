# 🌸 Projet IrisML

Bienvenue dans la documentation du projet **IrisML**. Cette application utilise le Machine Learning pour classifier les espèces d'Iris (Setosa, Versicolor, Virginica) et est déployée sur **Azure**.

---

## Architecture du Projet

Le projet est divisé en deux services principaux conteneurisés avec **Docker** :

* **Backend (FastAPI)** : Gère la logique de prédiction et expose l'API.
* **Frontend (Streamlit)** : Fournit une interface utilisateur pour saisir les données.
* **MLOps** : Suivi des expérimentations avec **MLflow** et automatisation via **GitHub Actions**.

---

## Installation et Développement

### 1. Gestion des Dépendances

* **Installation du Backend :**
    ```bash
    pip install -r backend/requirements.txt
    ```
* **Installation du Frontend :**
    ```bash
    pip install -r front/requirements.txt
    ```

### 2. Entraînement et Suivi
Le modèle est entraîné avec **Scikit-Learn**. 

1.  **Lancer l'entraînement** : `python train.py`
2.  **Suivi des métriques** : Les résultats (accuracy, logs) sont disponibles via l'interface **MLflow**.
3.  **Export** : Le modèle final est sauvegardé sous forme de fichier `.pkl` dans le dossier `/model`.

### 3. Tests Qualité
Nous utilisons **Pytest** pour valider les endpoints de l'API et la fiabilité du modèle :
```bash
pytest