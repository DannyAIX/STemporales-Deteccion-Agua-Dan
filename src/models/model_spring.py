# MODELO PARA MANANTIALES
# src/models/model_spring.py

from .base_model import train_sarima

def run_springs(datasets):
    results = {}
    for name, df in datasets.items():
        preds, mae, rmse = train_sarima(df, name, target=df.columns[-1])
        results[name] = {"mae": mae, "rmse": rmse}
    return results