import sys
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

DATA_PATH = "Churn_Modelling.csv"

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "--all"

    if arg == "--prepare":
        X_train, X_test, y_train, y_test, scaler = prepare_data(DATA_PATH)

    elif arg == "--train":
        X_train, X_test, y_train, y_test, scaler = prepare_data(DATA_PATH)
        model = train_model(X_train, y_train)
        save_model(model, scaler)

    elif arg == "--evaluate":
        X_train, X_test, y_train, y_test, scaler = prepare_data(DATA_PATH)
        model, scaler = load_model()
        evaluate_model(model, X_test, y_test)

    elif arg == "--all":
        X_train, X_test, y_train, y_test, scaler = prepare_data(DATA_PATH)
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
        save_model(model, scaler)
        print("Pipeline terminé avec succès !")
