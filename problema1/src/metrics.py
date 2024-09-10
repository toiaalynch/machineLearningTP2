import numpy as np

# FALTA LO DE LOS GRAFICOS!
def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))  # Verdaderos Positivos
    TN = np.sum((y_true == 0) & (y_pred == 0))  # Verdaderos Negativos
    FP = np.sum((y_true == 0) & (y_pred == 1))  # Falsos Positivos
    FN = np.sum((y_true == 1) & (y_pred == 0))  # Falsos Negativos
    return np.array([[TP, FP], [FN, TN]])

def accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return (cm[0, 0] + cm[1, 1]) / np.sum(cm)

def precision(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TP, FP = cm[0, 0], cm[0, 1]
    return TP / (TP + FP) if (TP + FP) != 0 else 0

def recall(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TP, FN = cm[0, 0], cm[1, 0]
    return TP / (TP + FN) if (TP + FN) != 0 else 0

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)  # Cambia el nombre de la variable para evitar conflicto
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) != 0 else 0

# Cálculo del AUC-ROC
def roc_auc(y_true, y_prob):
    sorted_indices = np.argsort(-y_prob)
    y_true_sorted = y_true[sorted_indices]
    
    pos_count = np.sum(y_true == 1)
    neg_count = np.sum(y_true == 0)
    
    if pos_count == 0 or neg_count == 0:
        return 0  # Si no hay positivos o negativos, AUC-ROC no tiene sentido.
    
    tpr = 0  # True Positive Rate
    fpr = 0  # False Positive Rate
    auc = 0
    
    for i in range(len(y_true)):
        if y_true_sorted[i] == 1:
            tpr += 1 / pos_count
        else:
            auc += tpr / neg_count
            fpr += 1 / neg_count
    
    return auc

# Cálculo del AUC-PR
def auc_pr(y_true, y_prob):
    sorted_indices = np.argsort(-y_prob)
    y_true_sorted = y_true[sorted_indices]
    
    tp = 0
    fp = 0
    fn = np.sum(y_true == 1)
    
    if fn == 0:
        return 0  # Si no hay positivos, AUC-PR no tiene sentido.
    
    precision_points = []
    recall_points = []
    
    for i in range(len(y_true)):
        if y_true_sorted[i] == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1
        
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        
        precision_points.append(precision)
        recall_points.append(recall)
    
    auc_pr = 0
    for i in range(1, len(precision_points)):
        auc_pr += (recall_points[i] - recall_points[i - 1]) * precision_points[i]
    
    return auc_pr

# Función para guardar las métricas en un diccionario
def guardar_metricas(y_true, y_pred, y_prob):
    metrics = {}
    metrics["Accuracy"] = accuracy(y_true, y_pred)
    metrics["Precision"] = precision(y_true, y_pred)
    metrics["Recall"] = recall(y_true, y_pred)
    metrics["F1-Score"] = f1_score(y_true, y_pred)
    metrics["AUC-ROC"] = roc_auc(y_true, y_prob)
    metrics["AUC-PR"] = auc_pr(y_true, y_prob)
    return metrics

# Función para imprimir la tabla de resultados
def imprimir_tabla_resultados(resultados):
    print(f"{'Modelo':<20}{'Accuracy':<10}{'Precision':<10}{'Recall':<10}{'F1-Score':<10}{'AUC-ROC':<10}{'AUC-PR':<10}")
    for modelo, metricas in resultados.items():
        print(f"{modelo:<20}{metricas['Accuracy']:<10.4f}{metricas['Precision']:<10.4f}{metricas['Recall']:<10.4f}{metricas['F1-Score']:<10.4f}{metricas['AUC-ROC']:<10.4f}{metricas['AUC-PR']:<10.4f}")
