
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import json


def predict(model, X_test_tfidf):
    y_pred = model.predict(X_test_tfidf)
    return y_pred


def calculate_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, f1, confusion_matrix(y_test, y_pred)


def save_metrics(accuracy, f1, conf_matrix):
    metrics = {
        "accuracy": accuracy,
        "f1_score": f1,
        "confusion_matrix": conf_matrix.tolist()  # Convertir en liste pour JSON
    }
    with open("metrics/scores.json", "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info("Métriques sauvegardées dans metrics/scores.json")


def save_predictions(X_test, y_test, y_pred):
    predictions_df = X_test
    predictions_df["y_true"] = y_test
    predictions_df["y_pred"] = y_pred

    predictions_df.to_csv("data/predictions/predictions.csv", index=False)
    logger.info("Prédictions sauvegardées dans data/predictions.csv")





if __name__ == "__main__":
    try:
        log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_fmt)
        logger = logging.getLogger(__name__)
        
        X_train_tfidf = np.load("data/processed_data/X_train_tfidf.npy")
        X_test_tfidf = np.load("data/processed_data/X_test_tfidf.npy")
        y_train = pd.read_csv("data/processed_data/y_train.csv", sep=',')
        y_test = pd.read_csv("data/processed_data/y_test.csv", sep=',')
        X_test = pd.read_pickle("data/processed_data/X_test.pickle")
        tfidf = joblib.load("models/tfidf_vectorizer_sample.pickle")

        model = joblib.load("models/trustpilot_lgbm_model_sample.pkl")
        

        logger.info("Predicting on test set")
        y_pred = predict(model, np.array(X_test_tfidf))

        logger.info("Calculating metrics")  
        accuracy, f1, conf_matrix = calculate_metrics(y_test, y_pred)
        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"F1-score: {f1}")
        logger.info(f"Confusion Matrix: {conf_matrix}")
        
        save_metrics(accuracy, f1, conf_matrix)
        logger.info("Metrics saved")
        
        save_predictions(X_test, y_test, y_pred)
        logger.info("Predictions saved")


    except Exception as e:
        logger.error(e)