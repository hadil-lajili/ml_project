import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

def send_to_elasticsearch(data):
    try:
        es.index(index="mlflow-metrics", document=data)
        print("Logs envoyés à Elasticsearch !")
    except Exception as e:
        print(f"Erreur Elasticsearch : {e}")

def prepare_data(filepath):
    df = pd.read_csv(filepath)
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
    le = LabelEncoder()
    df["Geography"] = le.fit_transform(df["Geography"])
    df["Gender"] = le.fit_transform(df["Gender"])
    X = df.drop(columns=["Exited"])
    y = df["Exited"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("Données prêtes !")
    return X_train, X_test, y_train, y_test, scaler

def train_model(X_train, y_train, n_estimators=200, random_state=42):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        max_depth=10,
        min_samples_split=5
    )
    model.fit(X_train, y_train)
    # Log seulement si un run MLflow est actif
    try:
        if mlflow.active_run():
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("random_state", random_state)
    except Exception as e:
        print(f"MLflow log ignoré : {e}")
    print("Modèle entraîné !")
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("Accuracy  :", round(accuracy, 4))
    print("Precision :", round(precision, 4))
    print("Recall    :", round(recall, 4))
    print("F1-Score  :", round(f1, 4))
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }
    send_to_elasticsearch(log_data)
    return accuracy, precision, recall, f1

def save_model(model, scaler):
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("Modèle sauvegardé !")

def load_model():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("Modèle chargé !")
    return model, scaler
