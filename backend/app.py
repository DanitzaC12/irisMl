import os
import joblib
import logging # 1.
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ])
logger = logging.getLogger(__name__)

app = FastAPI(title="Prédiction de variétés d'iris")

class InputData(BaseModel):
    sepal_length: float 
    sepal_width: float
    petal_length: float
    petal_width: float

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ml", "model", "model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Modèle chargé avec succès depuis : {MODEL_PATH}")
except Exception as e:
    logger.error(f"Erreur critique de chargement du modèle : {e}")
    model = None

@app.get("/")
def root():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
async def predict(data: InputData):
    if model is None:
        logger.warning("Tentative de prédiction mais modèle absent")
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    try:
        features = [[
            data.sepal_length, 
            data.sepal_width, 
            data.petal_length, 
            data.petal_width
        ]]
        prediction = model.predict(features)
        pred_value = int(prediction[0])
        logger.info(f"Prédiction réussie : Input={features} -> Output={pred_value}")
        return {"prediction": pred_value}
    except Exception as e:
        logger.exception("Erreur lors de la prédiction")
        raise HTTPException(status_code=500, detail=str(e))

