# --- Entrenar y predecir con SARIMA automáticamente --------------------------
import math
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_sarima(df, name, target):
    print(f"\n=== Entrenando SARIMA para {name} ===")

    # split 80/20
    n = len(df)
    split = int(n * 0.8)
    train = df[target].iloc[:split]
    test = df[target].iloc[split:]

    # Modelo básico (después se puede optimizar)
    model = SARIMAX(train, order=(3,1,1), seasonal_order=(1,1,1,12))
    model_fit = model.fit(disp=False)

    # Predicción
    preds = model_fit.predict(start=test.index[0], end=test.index[-1])

    # Métricas
    mae = mean_absolute_error(test, preds)
    rmse = math.sqrt(mean_squared_error(test, preds))

    print(f"MAE  → {mae:.4f}")
    print(f"RMSE → {rmse:.4f}")

    # Guardar predicción
    preds.to_csv(f"pred_{name}.csv")

    print(f"Archivo generado: pred_{name}.csv")

    return preds, mae, rmse