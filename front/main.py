import streamlit as st
import extra_streamlit_components as stx
import requests
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

st.set_page_config(page_title="Iris Prediction", page_icon="🌸", layout="centered")
chosen_id = stx.tab_bar(data=[
    stx.TabBarItemData(id="tab1", title="Prédiction", description="Saisir les données"),
], default="tab1")

if chosen_id == "tab1":
    st.title("Prédire la variété d'Iris")

with st.form("iris_form"):
    st.subheader("Paramètres d'entrée")
    col1, col2 = st.columns(2)
    with col1:
        sl = st.number_input("sepal_length", value=0.0, format="%.1f")
        sw = st.number_input("sepal_width", value=0.0, format="%.1f")
    with col2:
        pl = st.number_input("petal_length", value=0.0, format="%.1f")
        pw = st.number_input("petal_width", value=0.0, format="%.1f")

    submit = st.form_submit_button("Prédiction")

if submit:
    payload = {
        "sepal_length": sl,
            "sepal_width": sw,
            "petal_length": pl,
            "petal_width": pw
    }
    try:
        with st.spinner("Analyse en cours..."):
            response = requests.post(f"{BACKEND_URL}/predict", json=payload)
            
            if response.status_code == 200:
                res_json = response.json()
                pred_id = res_json.get("prediction")

                iris_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
                nom_iris = iris_map.get(pred_id, f"Classe {pred_id}")

                st.success(f"### Résultat : {nom_iris}")
            else:
                st.error(f"Erreur API ({response.status_code}) : {response.text}")

    except Exception as e:
        st.error(f"Connexion impossible à l'API: {e}")
