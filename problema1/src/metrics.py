import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve


def confusion_matrix(y_true, y_pred):
    tp = sum(1 for vt, vp in zip(y_true, y_pred) if vt == 1 and vp == 1)
    tn = sum(1 for vt, vp in zip(y_true, y_pred) if vt == 0 and vp == 0)
    fp = sum(1 for vt, vp in zip(y_true, y_pred) if vt == 0 and vp == 1)
    fn = sum(1 for vt, vp in zip(y_true, y_pred) if vt == 1 and vp == 0)
    return [tn, fp], [fn, tp]  # np.array para asegurar que sea un numpy array

def precision(y_true, y_pred):
    tp = sum(1 for vt, vp in zip(y_true, y_pred) if vt == 1 and vp == 1)
    fp = sum(1 for vt, vp in zip(y_true, y_pred) if vt == 0 and vp == 1)
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall(y_true, y_pred):
    tp = sum(1 for vt, vp in zip(y_true, y_pred) if vt == 1 and vp == 1)
    fn = sum(1 for vt, vp in zip(y_true, y_pred) if vt == 1 and vp == 0)
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

def accuracy(y_true, y_pred):
    correctas = sum(1 for vt, vp in zip(y_true, y_pred) if vt == vp)
    return correctas / len(y_true)

import numpy as np


def auc_roc(y_true, y_prob):
    try:
        auc = roc_auc_score(y_true, y_prob)
        return auc
    except ValueError as e:
        print(f"Error en AUC-ROC: {e}")
        return 0

def auc_pr(y_true, y_prob):
    try:
        auc = average_precision_score(y_true, y_prob)
        return auc
    except ValueError as e:
        return 0


def plot_combined_roc_curve(model_results):
    plt.figure(figsize=(8, 6))
    for nombre_modelo, (y_true, y_prob) in model_results.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_value = auc_roc(y_true, y_prob)
        plt.plot(fpr, tpr, label=f'{nombre_modelo} (AUC = {auc_value:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal de referencia
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC combinada')
    plt.legend(loc="lower right")
    plt.show()

def plot_combined_pr_curve(model_results):
    plt.figure(figsize=(8, 6))
    for nombre_modelo, (y_true, y_prob) in model_results.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auc_value = auc_pr(y_true, y_prob)
        plt.plot(recall, precision, label=f'{nombre_modelo} (AUC PR = {auc_value:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall combinada')
    plt.legend(loc="lower left")
    plt.show()


def save_metrics(y_true, y_pred, y_prob):
    metrics = {}
    metrics["Accuracy"] = accuracy(y_true, y_pred)
    metrics["Precision"] = precision(y_true, y_pred)
    metrics["Recall"] = recall(y_true, y_pred)
    metrics["F1-Score"] = f1_score(y_true, y_pred)
    metrics["AUC-ROC"] = auc_roc(y_true, y_prob)
    metrics["AUC-PR"] = auc_pr(y_true, y_prob)
    return metrics


def print_table(resultados):
    print(f"{'Modelo':<20}{'Accuracy':<10}{'Precision':<10}{'Recall':<10}{'F1-Score':<10}{'AUC-ROC':<10}{'AUC-PR':<10}")
    for modelo, metricas in resultados.items():
        print(f"{modelo:<20}{metricas['Accuracy']:<10.4f}{metricas['Precision']:<10.4f}{metricas['Recall']:<10.4f}{metricas['F1-Score']:<10.4f}{metricas['AUC-ROC']:<10.4f}{metricas['AUC-PR']:<10.4f}")

def print_matrix(matriz, nombre_modelo):
    plt.figure(figsize=(6, 4))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', xticklabels=["Pred. 0", "Pred. 1"], yticklabels=["True 0", "True 1"])
    plt.title(f'Matriz de Confusión para {nombre_modelo}')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Verdadero')
    plt.show()
