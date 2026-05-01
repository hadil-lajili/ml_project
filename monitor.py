import psutil
import time
import joblib
import pandas as pd
from datetime import datetime
from elasticsearch import Elasticsearch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

es = Elasticsearch("http://localhost:9200")

SEUIL_CPU      = 80.0
SEUIL_RAM      = 90.0
SEUIL_ACCURACY = 0.80
LOG_FILE       = "alerts.log"
INTERVALLE     = 30

def write_alert(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alert = f"[ALERTE] {timestamp} -> {message}"
    print(alert)
    with open(LOG_FILE, "a") as f:
        f.write(alert + "\n")

def get_system_metrics():
    return {
        "timestamp"   : datetime.now().isoformat(),
        "cpu_percent" : psutil.cpu_percent(interval=1),
        "ram_percent" : psutil.virtual_memory().percent,
        "ram_used_gb" : round(psutil.virtual_memory().used / (1024**3), 2),
        "disk_percent": psutil.disk_usage("C:/").percent,
    }

def check_model_accuracy():
    try:
        model  = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        df = pd.read_csv("Churn_Modelling.csv")
        df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
        le = LabelEncoder()
        df["Geography"] = le.fit_transform(df["Geography"])
        df["Gender"]    = le.fit_transform(df["Gender"])
        X = df.drop(columns=["Exited"])
        y = df["Exited"]
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_test_scaled = scaler.transform(X_test)
        y_pred   = model.predict(X_test_scaled)
        accuracy = round(accuracy_score(y_test, y_pred), 4)
        return accuracy
    except Exception as e:
        print("Erreur accuracy :", e)
        return None

def check_alerts(metrics, accuracy):
    alerts = []
    if metrics["cpu_percent"] > SEUIL_CPU:
        msg = f"CPU critique : {metrics['cpu_percent']}% > {SEUIL_CPU}%"
        write_alert(msg)
        alerts.append(msg)
    if metrics["ram_percent"] > SEUIL_RAM:
        msg = f"RAM critique : {metrics['ram_percent']}% > {SEUIL_RAM}%"
        write_alert(msg)
        alerts.append(msg)
    if metrics["disk_percent"] > 95:
        msg = f"Disque critique : {metrics['disk_percent']}% utilise !"
        write_alert(msg)
        alerts.append(msg)
    if accuracy is not None and accuracy < SEUIL_ACCURACY:
        msg = f"Accuracy trop basse : {accuracy} < {SEUIL_ACCURACY}"
        write_alert(msg)
        alerts.append(msg)
    if not alerts:
        print(f"[OK] {datetime.now().strftime('%H:%M:%S')} -> Tout va bien !")
    return alerts

def send_to_elasticsearch(metrics, accuracy, alerts):
    try:
        doc = {
            "timestamp"   : metrics["timestamp"],
            "cpu_percent" : metrics["cpu_percent"],
            "ram_percent" : metrics["ram_percent"],
            "disk_percent": metrics["disk_percent"],
            "accuracy"    : accuracy,
            "nb_alertes"  : len(alerts),
            "alertes"     : alerts,
        }
        es.index(index="monitoring-alertes", document=doc)
        print("Donnees envoyees a Elasticsearch !")
    except Exception as e:
        print("Erreur Elasticsearch :", e)

def run_monitoring():
    compteur = 1
    print("=" * 50)
    print("Monitoring continu demarre !")
    print(f"Verification toutes les {INTERVALLE} secondes")
    print("Appuyez sur Ctrl+C pour arreter")
    print("=" * 50)

    while True:
        print(f"\n--- Verification {compteur} ---")
        metrics  = get_system_metrics()
        accuracy = check_model_accuracy()

        print(f"CPU      : {metrics['cpu_percent']}%")
        print(f"RAM      : {metrics['ram_percent']}%")
        print(f"Accuracy : {accuracy}")

        alerts = check_alerts(metrics, accuracy)
        send_to_elasticsearch(metrics, accuracy, alerts)

        compteur += 1
        print(f"Prochaine verification dans {INTERVALLE} secondes...")
        time.sleep(INTERVALLE)

if __name__ == "__main__":
    try:
        run_monitoring()
    except KeyboardInterrupt:
        print("\nMonitoring arrete proprement !")