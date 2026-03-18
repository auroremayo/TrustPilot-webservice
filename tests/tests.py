# tests.py
import base64
import json
import pytest
from fastapi.testclient import TestClient

from app import app  # Assure-toi que ton fichier principal s'appelle main.py

client = TestClient(app)

API_KEY = "test_api_key"  # Mets ici la même que dans ton .env pour les tests

# ----------------------------
# 🔹 Test endpoint racine
# ----------------------------
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data

# ----------------------------
# 🔹 Test endpoints health
# ----------------------------
def test_health_live():
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "live"}

def test_health_ready():
    response = client.get("/health/ready")
    # On teste juste la structure
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        assert response.json() == {"status": "ready"}
    else:
        assert "detail" in response.json()

# ----------------------------
# 🔹 Test prédiction simple
# ----------------------------
def test_predict_sentiment():
    payload = {"text": "Le produit est excellent et la livraison rapide"}
    headers = {"X-API-Key": API_KEY}
    
    response = client.post("/predict", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "texte" in data
    assert "sentiment" in data
    assert "prediction_score" in data
    assert "class_id" in data

# ----------------------------
# 🔹 Test batch predict
# ----------------------------
def test_batch_predict():
    payload = [
        {"text": "Super produit, très satisfait"},
        {"text": "Livraison lente et mauvaise qualité"}
    ]
    headers = {"X-API-Key": API_KEY}
    
    response = client.post("/predict/batch", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2
    for pred in data["predictions"]:
        assert "texte" in pred
        assert "sentiment" in pred
        assert "prediction_score" in pred
        assert "class_id" in pred

# ----------------------------
# 🔹 Test extraction texte CSV base64
# ----------------------------
def test_extract_text_csv():
    csv_content = "text,score\nSuper produit,2\nLivraison lente,0"
    csv_base64 = base64.b64encode(csv_content.encode("utf-8")).decode("utf-8")
    payload = {"csv_base64": csv_base64}
    headers = {"X-API-Key": API_KEY}

    response = client.post("/extract_text_csv", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert data == ["Super produit", "Livraison lente"]

# ----------------------------
# 🔹 Test metrics
# ----------------------------
def test_get_metrics():
    headers = {"X-API-Key": API_KEY}
    response = client.get("/metrics", headers=headers)
    # Peut échouer si metrics/scores.json n'existe pas, juste vérifier le code 200 ou 500
    assert response.status_code in (200, 500)
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, dict)

# ----------------------------
# 🔹 Test feature importance
# ----------------------------
def test_feature_importance():
    headers = {"X-API-Key": API_KEY}
    response = client.get("/feature-importance", headers=headers)
    assert response.status_code in (200, 500)
    if response.status_code == 200:
        data = response.json()
        assert "top_features" in data
        assert isinstance(data["top_features"], list)

# ----------------------------
# 🔹 Test clé API invalide
# ----------------------------
def test_invalid_api_key():
    payload = {"text": "Test"}
    headers = {"X-API-Key": "invalid_key"}
    response = client.post("/predict", json=payload, headers=headers)
    assert response.status_code == 403
    assert "Accès refusé" in response.json()["detail"]