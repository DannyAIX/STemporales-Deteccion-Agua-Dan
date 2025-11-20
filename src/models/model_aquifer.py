# MODELO PARA ACUÍFEROS
# src/models/model_aquifer.py

from .base_model import train_sarima

def run_aquifers(datasets):
    results = {}
    for name, df in datasets.items():
        preds, mae, rmse = train_sarima(df, name, target=df.columns[-1])
        results[name] = {"mae": mae, "rmse": rmse}
    return results