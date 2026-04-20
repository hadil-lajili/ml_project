import psutil
import time
from datetime import datetime
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

def get_system_metrics():
    return {
        "timestamp": datetime.now().isoformat(),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "ram_percent": psutil.virtual_memory().percent,
        "ram_used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "disk_percent": psutil.disk_usage("C:/").percent,
        "disk_used_gb": round(psutil.disk_usage("C:/").used / (1024**3), 2),
    }

def send_metrics(metrics):
    try:
        es.index(index="system-metrics", document=metrics)
        print("CPU:", metrics["cpu_percent"], "% | RAM:", metrics["ram_percent"], "%")
        print("Metriques systeme envoyees a Elasticsearch !")
    except Exception as e:
        print("Erreur :", e)

if __name__ == "__main__":
    print("Demarrage du monitoring systeme...")
    for i in range(5):
        metrics = get_system_metrics()
        send_metrics(metrics)
        time.sleep(2)
    print("Monitoring termine !")