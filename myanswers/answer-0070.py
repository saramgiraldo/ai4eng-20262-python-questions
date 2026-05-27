import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold

def limpiar_y_codificar(df):
    
    # identificar columnas categóricas y numéricas
    cat_cols = df.select_dtypes(include=['object']).columns
    num_cols = df.select_dtypes(exclude=['object']).columns
    
    # aplicar OneHotEncoder
    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown='ignore'
    )
    
    if len(cat_cols) > 0:
        X_cat = encoder.fit_transform(df[cat_cols])
    else:
        X_cat = np.empty((len(df), 0))
    
    # obtener variables numéricas
    X_num = df[num_cols].to_numpy()
    
    # combinar matrices
    X = np.hstack([X_cat, X_num])
    
    # eliminar columnas con varianza 0
    selector = VarianceThreshold(threshold=0.0)
    X_filtrado = selector.fit_transform(X)
    
    return X_filtrado
