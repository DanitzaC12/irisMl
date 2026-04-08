import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.data
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_model():
    mlflow.set_experiment("Iris_Classification")
    mlflow.autolog()

    with mlflow.start_run():
        data = load_iris()
        df_iris = pd.DataFrame(data.data, columns=data.feature_names)

        dataset_iris = mlflow.data.from_pandas(df_iris, name="iris_dataset", source="sklearn")
        mlflow.log_input(dataset_iris, context="training")
        mlflow.log_table(data=df_iris, artifact_file="view_data.json")
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

        n_estimators = 100

        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print(f"Modèle entrainé avec une accuracy de {acc}")

        model_path =os.path.join(MODEL_DIR, "model.pkl")
        joblib.dump(clf, model_path)
        print(f"Modèle sauvegardé localement dans : {model_path}")


if __name__ == "__main__":
    train_model()