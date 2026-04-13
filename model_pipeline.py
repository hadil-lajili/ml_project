import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


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
    print("Modèle entraîné !")
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def save_model(model, scaler):
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("Modèle sauvegardé !")


def load_model():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("Modèle chargé !")
    return model, scaler
