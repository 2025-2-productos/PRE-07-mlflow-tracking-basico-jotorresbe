"""
Prediccion script for the MLFlow model

This script loads a trained model from MLFlow and uses it to make predictions

$python make_predictions.py

"""

import mlflow
import pandas as pd

FILE_PATH = "data/winequality-red.csv"
MODEL_PATH = "models/best_model.pkl"
OUTPUT_PATH = "data/predictions.csv"

df = pd.read_csv(FILE_PATH)
y = df["quality"]
X = df.drop(columns=["quality"])

# Debe verificarse el run_id del modelo que se quiere cargar
# Se puede obtener el run_id desde la UI de MLFlow

logged_model = 'runs:/0f884e07dc6047429bfe2f6ac8382bb2/model'
loaded_model = mlflow.pyfunc.load_model(logged_model)
y = loaded_model.predict(X)

print(y)