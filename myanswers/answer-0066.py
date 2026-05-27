def detectar_secuencia_alternante(df):
    # no modificar el DataFrame original
    df_copy = df.copy()
    
    sensores_validos = []
    
    # recorrer cada sensor
    for sensor_id, grupo in df_copy.groupby("sensor_id"):
        
        # ordenar por "orden"
        grupo = grupo.sort_values("orden")
        
        estados = grupo["estado"].tolist()
        
        # verificar alternancia perfecta
        alterna = all(estados[i] != estados[i-1] for i in range(1, len(estados)))
        
        if alterna:
            sensores_validos.append(sensor_id)
    
    return sorted(sensores_validos)
