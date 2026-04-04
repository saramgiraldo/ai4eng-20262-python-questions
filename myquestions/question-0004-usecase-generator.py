import numpy as np
from sklearn.metrics import confusion_matrix
import random

def generar_caso_de_uso_evaluar_modelo_avanzado():

    n = random.randint(10, 50)

    y_true = np.random.randint(0,2,n)
    y_pred = np.random.randint(0,2,n)

    input_data = {"y_true": y_true, "y_pred": y_pred}

    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn=fp=fn=tp=0

    total = tn+fp+fn+tp

    acc = (tn+tp)/total if total>0 else 0
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    rec = tp/(tp+fn) if (tp+fn)>0 else 0

    output_data = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec
    }

    return input_data, output_data
