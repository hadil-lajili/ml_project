from prefect import flow, task
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model
import subprocess

# ============================================
# PIPELINE 1 : Data
# ============================================

@task
def task_prepare():
    X_train, X_test, y_train, y_test, scaler = prepare_data("Churn_Modelling.csv")
    print("Données prêtes !")
    return X_train, X_test, y_train, y_test, scaler

@flow(name="Pipeline Data")
def pipeline_data():
    task_prepare()
    print("Pipeline Data terminée !")

# ============================================
# PIPELINE 2 : Model
# ============================================

@task
def task_train(X_train, y_train):
    model = train_model(X_train, y_train)
    return model

@task
def task_evaluate(model, X_test, y_test):
    evaluate_model(model, X_test, y_test)

@task
def task_save(model, scaler):
    save_model(model, scaler)

@flow(name="Pipeline Model")
def pipeline_model():
    X_train, X_test, y_train, y_test, scaler = task_prepare()
    model = task_train(X_train, y_train)
    task_evaluate(model, X_test, y_test)
    task_save(model, scaler)
    print("Pipeline Model terminée !")

# ============================================
# PIPELINE 3 : Code
# ============================================

@task
def task_format():
    subprocess.run(["black", "model_pipeline.py", "main.py", "app.py"])
    print("Formatage terminé !")

@task
def task_quality():
    subprocess.run(["flake8", "model_pipeline.py", "main.py", "app.py"])
    print("Qualité vérifiée !")

@task
def task_security():
    subprocess.run(["bandit", "-r", "model_pipeline.py", "main.py", "app.py"])
    print("Sécurité vérifiée !")

@flow(name="Pipeline Code")
def pipeline_code():
    task_format()
    task_quality()
    task_security()
    print("Pipeline Code terminée !")

# ============================================
# PIPELINE PRINCIPALE : tout ensemble
# ============================================

@flow(name="Pipeline Principale")
def pipeline_principale():
    pipeline_data()
    pipeline_model()
    pipeline_code()
    print("Toutes les pipelines terminées !")

if __name__ == "__main__":
    pipeline_principale()
