from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from model_pipeline import load_model, prepare_data, train_model, save_model

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
        raise HTTPException(status_code=401, detail="Clé API invalide !")
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
        result = "Ce client va QUITTER la banque" if prediction == 1 else "Ce client va RESTER"
        return {
            "prediction": int(prediction),
            "resultat": result,
            "probabilite_de_depart": round(float(probability) * 100, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
def retrain(params: RetrainData, api_key: str = Security(verify_api_key)):
    try:
        global model, scaler
        X_train, X_test, y_train, y_test, scaler = prepare_data("Churn_Modelling.csv")
        model = train_model(X_train, y_train, n_estimators=params.n_estimators)
        save_model(model, scaler)
        return {"message": "Modele reentraine avec succes !", "n_estimators": params.n_estimators}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))