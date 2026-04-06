from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model

# Étape 1 - Préparer les données
X_train, X_test, y_train, y_test, scaler = prepare_data('Churn_Modelling.csv')

# Étape 2 - Entraîner le modèle
model = train_model(X_train, y_train)

# Étape 3 - Évaluer le modèle
evaluate_model(model, X_test, y_test)

# Étape 4 - Sauvegarder le modèle
save_model(model, scaler)

# Étape 5 - Charger le modèle (pour vérifier)
model, scaler = load_model()
print("Pipeline terminé avec succès !")