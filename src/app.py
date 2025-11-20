# Series de tiempo - Detección de Agua

#Este proyecto se va a realizar en Kaggle.
#El propósito es predecir el volumen de agua
#Acea Group es la mayor empresa de agua en Italia y busca predecir los niveles diarios de agua en fuentes, 
# ríos y acuíferos para gestionar mejor el consumo y proteger estos recursos, especialmente porque se llenan en invierno
#  y se vacían en verano.
#Los datos representan distintos tipos de cuerpos de agua, cada uno con variables propias porque su comportamiento es único. 
# Acea trabaja con manantiales, lagos, ríos y acuíferos, y necesita entender su disponibilidad para proteger el recurso.
# El reto es predecir cuánta agua habrá en cada tipo, 
# creando cuatro modelos —uno por categoría— que permitan anticipar niveles diarios o mensuales y asegurar una buena
#  planificación del uso del agua.

#La evaluación se basa en tres áreas: metodología, presentación y aplicación, cada una con un máximo de 5 puntos. 
# Se revisa si los modelos son apropiados, si se mide su desempeño (MAE y RMSE), 
# si el notebook explica bien la historia con visualizaciones y análisis claros, 
# y si el modelo puede predecir niveles o flujos de agua y aplicarse a nuevos conjuntos de datos.
# Las entregas deben enviarse al organizador y serán evaluadas por Acea. 
# La competencia inició el 10 de diciembre de 2020, cerró el 17 de febrero de 2021 
# y los ganadores se anunciaron el 10 de marzo de 2021.

#El concurso utiliza nueve datasets totalmente independientes, cada uno representando un tipo distinto de cuerpo de agua. 
# Acea trabaja con cuatro categorías: acuíferos (4 datasets), manantiales (3), un río y un lago. 
# Cada uno tiene variables propias según su origen, alimentación y comportamiento, 
# por lo que las características cambian entre un manantial, un lago, un acuífero o un río.
# Los acuíferos incluyen Auser, Petrignano, Doganella y Luco, 
# cada uno influido por factores como lluvia, temperaturas, profundidad del agua o volúmenes drenados.
#  Los manantiales (Amiata, Madonna di Canneto y Lupa) se alimentan por infiltración o cuencas específicas.
#  El río Arno se evalúa por su nivel hidrométrico, y el lago Bilancino sirve como reserva para alimentar al Arno en verano. 
# Cada dataset exige predecir una variable clave distinta según el comportamiento del cuerpo de agua.

# luego del EDA ejecutar train_all.py para generar los modelos y los archivos

# Paso 1: Obtener Datos de CSV

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math


#9 Dasets acuíferos (4 datasets), manantiales (3), un río y un lago. 
Aquifer_Auser = pd.read_csv("/workspaces/STemporales-Deteccion-Agua-Dan/data/raw/Aquifer_Auser.csv")  # pasar la data a un data fram
Aquifer_Doganella = pd.read_csv("/workspaces/STemporales-Deteccion-Agua-Dan/data/raw/Aquifer_Doganella.csv")  # pasar la data a un data fram

Aquifer_Luco = pd.read_csv("/workspaces/STemporales-Deteccion-Agua-Dan/data/raw/Aquifer_Luco.csv")  # pasar la data a un data fram
Aquifer_Petrignano = pd.read_csv("/workspaces/STemporales-Deteccion-Agua-Dan/data/raw/Aquifer_Petrignano.csv")  # pasar la data a un data fram
Lake_Bilancino = pd.read_csv("/workspaces/STemporales-Deteccion-Agua-Dan/data/raw/Lake_Bilancino.csv")  # pasar la data a un data fram
River_Arno = pd.read_csv("/workspaces/STemporales-Deteccion-Agua-Dan/data/raw/River_Arno.csv")  # pasar la data a un data fram
Water_Spring_Amiata = pd.read_csv("/workspaces/STemporales-Deteccion-Agua-Dan/data/raw/Water_Spring_Amiata.csv")  # pasar la data a un data fram
Water_Spring_Lupa = pd.read_csv("/workspaces/STemporales-Deteccion-Agua-Dan/data/raw/Water_Spring_Lupa.csv")  # pasar la data a un data fram
Water_Spring_Madonna_di_Canneto = pd.read_csv("/workspaces/STemporales-Deteccion-Agua-Dan/data/raw/Water_Spring_Madonna_di_Canneto.csv")  # pasar la data a un data fram

#Paso 2 Entender o explorar la data

print("Aquifer_Auser")
print(Aquifer_Auser.head()) #ver rapidamente si cargo la info
print(Aquifer_Auser.columns) # ver las columnas, date object y sales float
print(Aquifer_Auser.info()) # ves tipos de datos, valores nulos y memoria usada, todo de un vistazo. float64(1), object(1)
print(Aquifer_Auser.describe()) # ESTADISTICAS de cada columna,

print("Aquifer_Doganella")
print(Aquifer_Doganella.head()) #ver rapidamente si cargo la info
print(Aquifer_Doganella.columns) # ver las columnas, date object y sales float
print(Aquifer_Doganella.info()) # ves tipos de datos, valores nulos y memoria usada, todo de un vistazo. float64(1), object(1)
print(Aquifer_Doganella.describe()) # ESTADISTICAS de cada columna,

print("Aquifer_Luco")
print(Aquifer_Luco.head()) #ver rapidamente si cargo la info
print(Aquifer_Luco.columns) # ver las columnas, date object y sales float
print(Aquifer_Luco.info()) # ves tipos de datos, valores nulos y memoria usada, todo de un vistazo. float64(1), object(1)
print(Aquifer_Luco.describe()) # ESTADISTICAS de cada columna,

print("Aquifer_Petrignano")
print(Aquifer_Petrignano.head()) #ver rapidamente si cargo la info
print(Aquifer_Petrignano.columns) # ver las columnas, date object y sales float
print(Aquifer_Petrignano.info()) # ves tipos de datos, valores nulos y memoria usada, todo de un vistazo. float64(1), object(1)
print(Aquifer_Petrignano.describe()) # ESTADISTICAS de cada columna,

print("Lake_Bilancino")
print(Lake_Bilancino.head()) #ver rapidamente si cargo la info
print(Lake_Bilancino.columns) # ver las columnas, date object y sales float
print(Lake_Bilancino.info()) # ves tipos de datos, valores nulos y memoria usada, todo de un vistazo. float64(1), object(1)
print(Lake_Bilancino.describe()) # ESTADISTICAS de cada columna,

print("River_Arno")
print(River_Arno.head()) #ver rapidamente si cargo la info
print(River_Arno.columns) # ver las columnas, date object y sales float
print(River_Arno.info()) # ves tipos de datos, valores nulos y memoria usada, todo de un vistazo. float64(1), object(1)
print(River_Arno.describe()) # ESTADISTICAS de cada columna,

print("Water_Spring_Amiata")
print(Water_Spring_Amiata.head()) #ver rapidamente si cargo la info
print(Water_Spring_Amiata.columns) # ver las columnas, date object y sales float
print(Water_Spring_Amiata.info()) # ves tipos de datos, valores nulos y memoria usada, todo de un vistazo. float64(1), object(1)
print(Water_Spring_Amiata.describe()) # ESTADISTICAS de cada columna,

print("Water_Spring_Lupa")
print(Water_Spring_Lupa.head()) #ver rapidamente si cargo la info
print(Water_Spring_Lupa.columns) # ver las columnas, date object y sales float
print(Water_Spring_Lupa.info()) # ves tipos de datos, valores nulos y memoria usada, todo de un vistazo. float64(1), object(1)
print(Water_Spring_Lupa.describe()) # ESTADISTICAS de cada columna,

print("Water_Spring_Madonna_di_Canneto")
print(Water_Spring_Madonna_di_Canneto.head()) #ver rapidamente si cargo la info
print(Water_Spring_Madonna_di_Canneto.columns) # ver las columnas, date object y sales float
print(Water_Spring_Madonna_di_Canneto.info()) # ves tipos de datos, valores nulos y memoria usada, todo de un vistazo. float64(1), object(1)
print(Water_Spring_Madonna_di_Canneto.describe()) # ESTADISTICAS de cada columna,

"""
Paso 3  Limpieza y Preprocesamiento 

Todos los datasets tienen inconsistencias similares:
* Date está como object/string → convertir a datetime.
* Muchísimas columnas con NaN, muchas veces sistemáticas (no aleatorias).
* Algunos datasets tienen columnas que son prácticamente constantes → eliminarlas.
* Hay valores negativos en variables que no deberían tenerlos (ej. volúmenes).
* Fechas no alineadas entre datasets (no se juntan entre sí, pero deben ser series continuas internamente).

Por lo que buscaremos crear DF por dataset limpio, indexado por fecha, sin NaN (al menos imputados), 
y con features útiles para modelos de series de tiempo.
"""

# PASO 3A — Crear una función general de limpieza

def clean_dataset(df):
    import pandas as pd

    # 1. Detectar posibles columnas de fecha
    possible_date_cols = ["cDate", "Date", "date", "Day", "day", "DATE"]

    date_col = None
    for col in df.columns:
        if col in possible_date_cols:
            date_col = col
            break

    # Último recurso: detectar columna con formato de fecha
    if date_col is None:
        for col in df.columns:
            try:
                pd.to_datetime(df[col], dayfirst=True)
                date_col = col
                break
            except:
                pass

    if date_col is None:
        raise ValueError("No se encontró ninguna columna de fecha en este dataset.")

    # 2. Convertir fechas correctamente
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)

    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col).set_index(date_col)

    # 3. Convertir objetos a numéricos
    df = df.infer_objects(copy=False)
    df = df.apply(pd.to_numeric, errors="coerce")

    # 4. Eliminar columnas completamente vacías
    df = df.dropna(axis=1, how="all")

    # 5. Interpolar usando tiempo
    df = df.interpolate(method="time", limit_direction="both")

    # 6. Rellenar restante sin usar método deprecated
    df = df.ffill().bfill()

    return df

# PASO 3B — Limpiar los 9 datasets

Auser = clean_dataset(Aquifer_Auser)
Doganella = clean_dataset(Aquifer_Doganella)
Luco = clean_dataset(Aquifer_Luco)
Petrignano = clean_dataset(Aquifer_Petrignano)

Bilancino = clean_dataset(Lake_Bilancino)
Arno = clean_dataset(River_Arno)

Amiata = clean_dataset(Water_Spring_Amiata)
Lupa = clean_dataset(Water_Spring_Lupa)
Canneto = clean_dataset(Water_Spring_Madonna_di_Canneto)

# PASO 4 — Análisis Exploratorio (EDA) para elegir modelos

# 4A — Visualización base

def plot_main_variable(df, title):
    plt.figure(figsize=(14,4))
    df.iloc[:,0].plot()
    plt.title(title)
    plt.show()

plot_main_variable(Auser, "Aquifer Auser")
plot_main_variable(Doganella, "Aquifer Doganella")
plot_main_variable(Bilancino, "Lake Bilancino")
plot_main_variable(Arno, "River Arno")


# ---------------------------------------------------
# 1. Detectar columna objetivo automáticamente
# ---------------------------------------------------
def detect_target_column(df):
    num_cols = df.select_dtypes(include=[np.number]).columns

    if len(num_cols) == 1:
        return num_cols[0]

    # Elegir la columna con:
    # - menos NaN
    # - mayor varianza
    # - mayor continuidad temporal
    scores = {}
    for col in num_cols:
        na_score = df[col].isna().mean()
        var_score = df[col].var()
        cont_score = df[col].notna().astype(int).rolling(10).sum().mean()

        scores[col] = (1 - na_score) * 0.4 + (var_score) * 0.4 + (cont_score) * 0.2

    best_col = max(scores, key=scores.get)
    return best_col


# ---------------------------------------------------
# 2. EDA Automático Completo
# ---------------------------------------------------
def full_eda(df, name="Dataset"):
    print(f"\n======================")
    print(f" EDA: {name}")
    print(f"======================\n")

    # --- Detectar target ---
    target = detect_target_column(df)
    print(f"➡ Variable objetivo detectada: **{target}**")

    # --- Resumen estadístico ---
    print("\n📌 Resumen estadístico:\n")
    print(df[target].describe())

    # --- Plot 1: Serie completa ---
    plt.figure(figsize=(12,4))
    plt.plot(df[target], linewidth=1)
    plt.title(f"{name} - Serie Temporal ({target})")
    plt.grid(True)
    plt.show()

    # --- Plot 2: Descomposición estacional ---
    try:
        decomposition = seasonal_decompose(df[target], model="additive", period=365)
        fig = decomposition.plot()
        fig.set_size_inches(12, 8)
        plt.suptitle(f"Descomposición Estacional - {name}", y=0.93)
        plt.show()
    except:
        print("\n⚠ No se pudo descomponer (serie muy corta o frecuencia irregular)")

    # --- Plot 3: ACF ---
    plt.figure(figsize=(10,3))
    plot_acf(df[target].dropna(), lags=50)
    plt.title(f"ACF - {name}")
    plt.show()

    # --- Plot 4: PACF ---
    plt.figure(figsize=(10,3))
    plot_pacf(df[target].dropna(), lags=50, method='ywm')
    plt.title(f"PACF - {name}")
    plt.show()

    # --- Detección de outliers ---
    q1 = df[target].quantile(0.25)
    q3 = df[target].quantile(0.75)
    iqr = q3 - q1
    outliers = df[(df[target] < q1 - 1.5*iqr) | (df[target] > q3 + 1.5*iqr)]

    print(f"\n🔍 Outliers detectados: {len(outliers)}")
    if len(outliers) > 0:
        print(outliers.head())

    # --- Sugerencia de modelo ---
    print("\n🤖 SUGERENCIA DE MODELO:")

    # Reglas heurísticas basadas en expertos en forecasting
    if df[target].autocorr() > 0.7:
        print("• Fuerte autocorrelación → **ARIMA / SARIMA** recomendado")

    if df[target].rolling(365).mean().std() > df[target].std()*0.3:
        print("• Estacionalidad fuerte → **SARIMA o Prophet**")

    if df[target].diff().abs().mean() > df[target].std()*1.2:
        print("• Mucho ruido → **Prophet o RandomForest**")

    if len(df) > 2000:
        print("• Serie larga → **LSTM / Deep Learning** puede ser útil")

    print("\n--- EDA COMPLETADO ---\n")




"""

El script train_all.py entrenó SARIMA para 9 series y generó 9 archivos de predicción.
Los warnings son normales y no afectan el resultado.

* Modelos que salieron bien (buen desempeño)
Estos tienen errores bajos → modelos estables:

Acuíferos

* Auser → MAE 0.37, RMSE 0.55
* Petrignano → MAE 0.40, RMSE 0.49
* Río Arno → MAE 0.49, RMSE 0.77

*  Aceptable / Medio

No perfectos pero razonables:

* Lago Bilancino → MAE 1.72, RMSE 3.98
* Manantial Amiata → MAE 2.67, RMSE 2.97

* Mal desempeño (SARIMA no funciona bien aquí)

SARIMA no está capturando la dinámica, los errores son muy altos:

* Acuíferos Doganella → MAE 5.19, RMSE 6.74
* Luco → MAE 71.27, RMSE 81.44 ❗
* Manantiales Lupa → MAE 3.65, RMSE 9.46
* Madonna di Canneto → MAE 52.58, RMSE 60.91 ❗

→ Estas series necesitan otro tipo de modelo (XGBoost, LightGBM, LSTM) porque son más complejas y ruidosas.
    """









