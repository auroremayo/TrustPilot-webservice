import os
from turtle import pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import joblib
import logging
import base64
import io
import pandas as pd
from typing import List

from streamlit import json
from data.processed_data.clean_text import clean_text
from data.processed_data.lemmatize_text import lemmatisation
from models.train_model import main as train_model_main
from models.predict_model import predict, calculate_metrics, save_metrics, save_predictions

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)
# Charge les variables d'environnement (pour le local et le cloud)
load_dotenv()

app = FastAPI(
    title="Trustpilot Sentiment Analysis API (LightGBM)",
    description="API sécurisée pour analyser le sentiment des avis clients en multiclasse",
    version="3.0"
)

# 🛡️ 1. Configuration de la sécurité
API_KEY = os.getenv("API_KEY") 
api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(status_code=403, detail="Accès refusé. Clé API invalide.")

# 2. Chargement du TF-IDF et du Modèle
try:
    tfidf = joblib.load("models/tfidf_vectorizer_sample.pkl")
    model = joblib.load("models/trustpilot_lgbm_model_sample.pkl")
    logger.info("Modèle et TF-IDF chargés avec succès !")
except Exception as e:
    model = None
    tfidf = None
    logger.error(f"Erreur lors du chargement : {e}")

class Review(BaseModel):
    text: str

class CSVBase64(BaseModel):
    csv_base64: str  # CSV encodé en base64

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API MLOps sécurisée. Allez sur /docs pour tester."}

# 🔒 3. Route de prédiction
@app.post("/predict")
def predict_sentiment(review: Review, api_key: str = Depends(get_api_key)):
    if model is None or tfidf is None:
        raise HTTPException(status_code=500, detail="Le modèle ou le TF-IDF n'est pas chargé.")
    
    texte_nettoye = clean_text(text=review.text)
    texte_lemmatise = lemmatisation(texte_nettoye)
    texte_vectorise = tfidf.transform([texte_lemmatise])
    
    # On fait la prédiction avec le Modèle
    prediction = model.predict(texte_vectorise)[0]
    
    # Calcul de la confiance (probabilité)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(texte_vectorise)[0]
        confidence = round(max(proba) * 100, 2)
    else:
        confidence = 100.0

    # ÉTAPE D : Mapping des 3 classes
    labels = {
        0: "Négatif",
        1: "Neutre",
        2: "Positif"
    }
    sentiment = labels.get(int(prediction), "Inconnu")
    
    return {
        "texte": review.text,
        "sentiment": sentiment,
        "prediction_score": f"{confidence}%",
        "class_id": int(prediction)
    }

@app.get("/health/live")
def health():
    return {"status": "live"}

@app.get("/health/ready")
def readiness():
    if model is not None and tfidf is not None:
        return {"status": "ready"}
    else:
        raise HTTPException(status_code=503, detail="Le modèle ou le TF-IDF n'est pas prêt.")
    
@app.get("/metrics")
def get_metrics(api_key: str = Depends(get_api_key)):
    try:
        with open("metrics/scores.json", "r") as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        logger.error(f"Erreur lors de la lecture des métriques : {e}")
        raise HTTPException(status_code=500, detail="Impossible de lire les métriques.")
    
@app.get("/feature-importance")
def feature_importance(api_key: str = Depends(get_api_key)):
    try:
        importances = pd.read_csv("metrics/feature_importance.csv")
        top_features = importances.head(20).to_dict(orient="records")
        return {"top_features": top_features}
    except Exception as e:
        logger.error(f"Erreur lors de la lecture de l'importance des features : {e}")
        raise HTTPException(status_code=500, detail="Impossible de lire l'importance des features.")

@app.post("/predict/batch")
def batch_predict(reviews: list[Review], api_key: str = Depends(get_api_key)):
    if model is None or tfidf is None:
        raise HTTPException(status_code=500, detail="Le modèle ou le TF-IDF n'est pas chargé.")
    
    results = []
    for review in reviews:
        texte_nettoye = clean_text(text=review.text)
        texte_lemmatise = lemmatisation(texte_nettoye)
        texte_vectorise = tfidf.transform([texte_lemmatise])
        
        prediction = model.predict(texte_vectorise)[0]
        
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(texte_vectorise)[0]
            confidence = round(max(proba) * 100, 2)
        else:
            confidence = 100.0

        labels = {
            0: "Négatif",
            1: "Neutre",
            2: "Positif"
        }
        sentiment = labels.get(int(prediction), "Inconnu")
        
        results.append({
            "texte": review.text,
            "sentiment": sentiment,
            "prediction_score": f"{confidence}%",
            "class_id": int(prediction)
        })
    
    return {"predictions": results}


@app.post("/extract_text_csv", response_model=List[str])
def extract_text_csv(data: CSVBase64, api_key: str = Depends(get_api_key)):
    """
    Décode un CSV en base64, convertit en DataFrame et retourne la colonne 'text' comme liste.
    """
    try:
        # 1️Décodage base64
        decoded = base64.b64decode(data.csv_base64)

        # Conversion en DataFrame
        csv_buffer = io.StringIO(decoded.decode("utf-8"))
        df = pd.read_csv(csv_buffer)

        # Vérification colonne 'text'
        if "text" not in df.columns:
            raise HTTPException(status_code=400, detail="La colonne 'text' est manquante dans le CSV.")

        # Récupération de la colonne 'text'
        text_list = df["text"].astype(str).tolist()

        return text_list

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement du CSV : {e}")
    
@app.post("/train-model")
def train_model():
    try:
        importances = train_model_main()
        logger.info(f"Entraînement terminé avec succès !")
        return {"message": "Entraînement terminé avec succès !", "top_features": importances.head(20).to_dict(orient="records")}
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement : {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de l'entraînement du modèle.")

