import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import mlflow
import mlflow.sklearn


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


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)
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
