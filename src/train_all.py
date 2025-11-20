import pandas as pd
import os
from models.model_aquifer import run_aquifers
from models.model_lake import run_lakes
from models.model_river import run_rivers
from models.model_spring import run_springs
from app import clean_dataset, detect_target_column

DATA_PATH = "/workspaces/STemporales-Deteccion-Agua-Dan/data/raw/"

aquifers = ["Aquifer_Auser", "Aquifer_Doganella", "Aquifer_Luco", "Aquifer_Petrignano"]
lakes = ["Lake_Bilancino"]
rivers = ["River_Arno"]
springs = ["Water_Spring_Amiata", "Water_Spring_Lupa", "Water_Spring_Madonna_di_Canneto"]

def load_and_clean(names):
    data = {}
    for name in names:
        file = os.path.join(DATA_PATH, f"{name}.csv")
        df = pd.read_csv(file)
        clean_df = clean_dataset(df)
        data[name] = clean_df
    return data

if __name__ == "__main__":
    print("\n=== Cargando y limpiando datasets ===")

    aquifer_data = load_and_clean(aquifers)
    lake_data = load_and_clean(lakes)
    river_data = load_and_clean(rivers)
    spring_data = load_and_clean(springs)

    print("\n=== Entrenando modelos ===")

    res_aq = run_aquifers(aquifer_data)
    res_lk = run_lakes(lake_data)
    res_rv = run_rivers(river_data)
    res_sp = run_springs(spring_data)

    print("\n=== Métricas finales ===")
    print("Acuíferos:", res_aq)
    print("Lagos:", res_lk)
    print("Ríos:", res_rv)
    print("Manantiales:", res_sp)

    print("\nListo. 9 predicciones generadas.")