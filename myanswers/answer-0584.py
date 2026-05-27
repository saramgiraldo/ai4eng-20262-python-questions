import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def varianza_acumulada_hasta_umbral(X, umbral=0.85):
    
    # estandarizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ajustar PCA completo
    pca = PCA()
    pca.fit(X_scaled)
    
    # calcular varianza acumulada
    varianza_acumulada = np.cumsum(pca.explained_variance_ratio_)
    
    # encontrar número mínimo de componentes
    n_componentes = int(np.argmax(varianza_acumulada >= umbral) + 1)
    
    return n_componentes
