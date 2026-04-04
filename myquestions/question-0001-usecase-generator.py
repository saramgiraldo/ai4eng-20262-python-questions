import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_filtrar_y_ordenar():

    n = random.randint(5, 15)

    df = pd.DataFrame({
        "price": np.random.uniform(50, 200, n),
        "rating": np.random.uniform(1, 5, n)
    })

    min_price = random.uniform(60, 150)
    min_rating = random.uniform(1, 5)

    input_data = {
        "df": df.copy(),
        "min_price": min_price,
        "min_rating": min_rating
    }

    output_data = df[(df["price"] > min_price) & (df["rating"] >= min_rating)] \
                    .sort_values(by="price", ascending=False)

    return input_data, output_data
