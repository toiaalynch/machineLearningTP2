import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import f1_score as sklearn_f1_score

def matriz_de_confusion(y_true, y_pred):
    tp = sum(1 for vt, vp in zip(y_true, y_pred) if vt == 1 and vp == 1)
    tn = sum(1 for vt, vp in zip(y_true, y_pred) if vt == 0 and vp == 0)
    fp = sum(1 for vt, vp in zip(y_true, y_pred) if vt == 0 and vp == 1)
    fn = sum(1 for vt, vp in zip(y_true, y_pred) if vt == 1 and vp == 0)
    return [[tn, fp], [fn, tp]]

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

def roc_auc(y_true, y_prob):
    try:
        auc = roc_auc_score(y_true, y_prob)
        return auc
    except ValueError as e:
        return 0


def auc_pr(y_true, y_prob):
    try:
        auc = average_precision_score(y_true, y_prob)
        return auc
    except ValueError as e:
        return 0





def plot_roc_curves(models, y_true, y_probs):
    plt.figure(figsize=(6, 6))
    
    for model_name, y_prob in zip(models, y_probs):
        sorted_indices = np.argsort(-y_prob)
        
        # Asegurarse de que y_true y y_prob tengan el mismo tamaño
        y_true_sorted = y_true[sorted_indices]
        if len(y_true_sorted) != len(y_prob):
            continue  # Saltar este modelo si las longitudes no coinciden

        pos_count = np.sum(y_true == 1)
        neg_count = np.sum(y_true == 0)

        tpr = []
        fpr = []
        cumulative_tpr = 0
        cumulative_fpr = 0

        for i in range(len(y_true_sorted)):
            if y_true_sorted[i] == 1:
                cumulative_tpr += 1 / pos_count
            else:
                cumulative_fpr += 1 / neg_count

            tpr.append(cumulative_tpr)
            fpr.append(cumulative_fpr)

        plt.plot(fpr, tpr, label=f"{model_name} ROC Curve")

    plt.plot([0, 1], [0, 1], 'r--', label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curves Comparison")
    plt.legend(loc="lower right")
    plt.show()



def plot_pr_curves(models, y_true, y_probs):
    plt.figure(figsize=(6, 6))

    for model_name, y_prob in zip(models, y_probs):
        sorted_indices = np.argsort(-y_prob)
        
        # Asegurarse de que y_true y y_prob tengan el mismo tamaño
        y_true_sorted = y_true[sorted_indices]
        if len(y_true_sorted) != len(y_prob):
            continue  # Saltar este modelo si las longitudes no coinciden

        tp = 0
        fp = 0
        fn = np.sum(y_true == 1)

        precision_points = []
        recall_points = []

        for i in range(len(y_true_sorted)):
            if y_true_sorted[i] == 1:
                tp += 1
                fn -= 1
            else:
                fp += 1

            precision = tp / (tp + fp) if (tp + fp) != 0 else 0
            recall = tp / (tp + fn) if (tp + fn) != 0 else 0

            precision_points.append(precision)
            recall_points.append(recall)

        plt.plot(recall_points, precision_points, label=f"{model_name} PR Curve")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves Comparison")
    plt.legend(loc="lower left")
    plt.show()

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
