import sys
import mlflow
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

DATA_PATH = "Churn_Modelling.csv"
mlflow.set_experiment("Churn_Modelling")

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "--all"

    if arg == "--prepare":
        prepare_data(DATA_PATH)

    elif arg == "--train":
        X_train, X_test, y_train, y_test, scaler = prepare_data(DATA_PATH)
        with mlflow.start_run():
            model = train_model(X_train, y_train)
            accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.sklearn.log_model(model, "model")
            save_model(model, scaler)

    elif arg == "--evaluate":
        X_train, X_test, y_train, y_test, scaler = prepare_data(DATA_PATH)
        model, scaler = load_model()
        evaluate_model(model, X_test, y_test)

    elif arg == "--all":
        X_train, X_test, y_train, y_test, scaler = prepare_data(DATA_PATH)
        with mlflow.start_run():
            model = train_model(X_train, y_train)
            accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.sklearn.log_model(model, "model")
            save_model(model, scaler)
            print("Pipeline terminé avec succès !")
