import numpy as np
from sklearn.metrics import precision_recall_curve, auc

def curva_precision_recall(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    
    # calcular curva precision-recall
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # calcular F1 para cada umbral
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = np.where(
            (precision[:-1] + recall[:-1]) == 0,
            0,
            2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1])
        )
    
    # índice del mejor F1
    best_idx = np.argmax(f1)
    
    # construir diccionario de salida
    resultado = {
        'mejor_umbral': float(thresholds[best_idx]),
        'mejor_f1': float(f1[best_idx]),
        'precision_en_umbral': float(precision[best_idx]),
        'recall_en_umbral': float(recall[best_idx]),
        'auc_pr': float(auc(recall, precision))
    }
    
    return resultado
