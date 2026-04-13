from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from model_pipeline import load_model

app = FastAPI()

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

@app.get("/")
def home():
    return {"message": "API Churn Modelling - fonctionnelle !"}

@app.post("/predict")
def predict(client: ClientData):
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
