import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_agrupar_y_normalizar():

    n = random.randint(10, 30)

    df = pd.DataFrame({
        "group": np.random.randint(0, 5, n),
        "value": np.random.uniform(0, 100, n)
    })

    input_data = {"df": df.copy()}

    grouped = df.groupby("group")["value"].mean().reset_index()

    min_val = grouped["value"].min()
    max_val = grouped["value"].max()

    grouped["normalized_value"] = (grouped["value"] - min_val) / (max_val - min_val)

    output_data = grouped[["group", "normalized_value"]]

    return input_data, output_data
