from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import mlflow
from model_pipeline import load_model, prepare_data, train_model, save_model
from elasticsearch import Elasticsearch
from datetime import datetime

es = Elasticsearch("http://localhost:9200")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = "mon_secret_mlops_2026"
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Cle API invalide !")
    return api_key

model, scaler = load_model()

class ClientData(BaseModel):
    CreditScore: float
    Geography: int
    Gender: int
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

class RetrainData(BaseModel):
    n_estimators: int = 100
    random_state: int = 42

@app.get("/")
def home():
    return {"message": "API Churn Modelling - fonctionnelle !"}

@app.post("/predict")
def predict(client: ClientData, api_key: str = Security(verify_api_key)):
    try:
        data = np.array([[
            client.CreditScore,
            client.Geography,
            client.Gender,
            client.Age,
            client.Tenure,
            client.Balance,
            client.NumOfProducts,
            client.HasCrCard,
            client.IsActiveMember,
            client.EstimatedSalary
        ]])
        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)[0]
        probability = model.predict_proba(data_scaled)[0][1]
        result = "Quitte" if prediction == 1 else "Reste"

        # Sauvegarder dans Elasticsearch
        try:
            es.index(index="predictions-history", document={
                "timestamp": datetime.now().isoformat(),
                "credit_score": client.CreditScore,
                "geography": ["France", "Germany", "Spain"][client.Geography],
                "gender": "Homme" if client.Gender == 1 else "Femme",
                "age": client.Age,
                "tenure": client.Tenure,
                "balance": client.Balance,
                "num_products": client.NumOfProducts,
                "has_cr_card": client.HasCrCard,
                "is_active": client.IsActiveMember,
                "salary": client.EstimatedSalary,
                "prediction": int(prediction),
                "result": result,
                "probability": round(float(probability) * 100, 2)
            })
            print("Prediction sauvegardee dans Elasticsearch !")
        except Exception as e:
            print(f"Erreur ES : {e}")

        return {
            "prediction": int(prediction),
            "resultat": "Ce client va QUITTER la banque" if prediction == 1 else "Ce client va RESTER",
            "probabilite_de_depart": round(float(probability) * 100, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
def retrain(params: RetrainData, api_key: str = Security(verify_api_key)):
    try:
        global model, scaler
        X_train, X_test, y_train, y_test, scaler = prepare_data("Churn_Modelling.csv")
        # Nouveau run MLflow à chaque entraînement
        with mlflow.start_run():
            model = train_model(X_train, y_train, n_estimators=params.n_estimators, random_state=params.random_state)
            save_model(model, scaler)
        return {"message": "Modele reentraine avec succes !", "n_estimators": params.n_estimators}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))