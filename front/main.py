import streamlit as st
import requests
import os

BACKEND_URL = os.getenv("BACKEND_URL", "https://iris-backend-api-c6gudrgggff5dhhz.francecentral-01.azurewebsites.net")

st.set_page_config(page_title="Iris Prediction", page_icon="🌸", layout="centered")

st.markdown("""
    <style>
    div[data-testid="stForm"] {
        background-color: #e8f5e9; 
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #4CAF50; 
    }
    div[data-testid="stFormSubmitButton"] button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌸 Prédire la variété d'Iris")
st.write("Ajustez les paramètres ci-dessous pour identifier l'espèce.")

with st.form("iris_form"):
    st.subheader("Paramètres d'entrée")
    col1, col2 = st.columns(2)
    with col1:
        sl = st.number_input("Longueur du sépale (cm)", min_value=0.0, max_value=10.0, value=0.0, step=0.1, format="%.1f")
        sw = st.number_input("Largeur du sépale (cm)", min_value=0.0, max_value=10.0, value=0.0, step=0.1, format="%.1f")
    with col2:
        pl = st.number_input("Longueur du pétale (cm)", min_value=0.0, max_value=10.0, value=0.0, step=0.1, format="%.1f")
        pw = st.number_input("Largeur du pétale (cm)", min_value=0.0, max_value=10.0, value=0.0, step=0.1, format="%.1f")

    submit = st.form_submit_button("Lancer la prédiction")

if submit:
    payload = {
        "sepal_length": sl,
            "sepal_width": sw,
            "petal_length": pl,
            "petal_width": pw
    }
    try:
        with st.spinner("Analyse en cours..."):
            endpoint = f"{BACKEND_URL.rstrip('/')}/predict"
            response = requests.post(endpoint, json=payload, timeout=10)
            
            if response.status_code == 200:
                res_json = response.json()
                pred_id = res_json.get("prediction")

                iris_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
                nom_iris = iris_map.get(pred_id, f"Classe {pred_id}")

                st.success(f"### Résultat : {nom_iris}")
                if "probability" in res_json:
                    st.info(f"Confiance du modèle : {res_json['probability']}%")
            else:
                st.error(f"Erreur API ({response.status_code}) : {response.text}")

    except Exception as e:
        st.error(f"Connexion impossible à l'API: {e}")
