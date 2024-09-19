import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve


def confusion_matrix(y_true, y_pred):
    classes = np.unique(y_true)
    n_classes = len(classes)
    matriz = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        matriz[int(true_label), int(pred_label)] += 1
    return matriz

def precision(y_true, y_pred):
    classes = np.unique(y_true)
    precisions = []
    for clase in classes:
        tp = sum(1 for vt, vp in zip(y_true, y_pred) if vt == clase and vp == clase)
        fp = sum(1 for vt, vp in zip(y_true, y_pred) if vt != clase and vp == clase)
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
    return np.mean(precisions)  # Promedio macro

def recall(y_true, y_pred):
    classes = np.unique(y_true)
    recalls = []
    for clase in classes:
        tp = sum(1 for vt, vp in zip(y_true, y_pred) if vt == clase and vp == clase)
        fn = sum(1 for vt, vp in zip(y_true, y_pred) if vt == clase and vp != clase)
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
    return np.mean(recalls)  # Promedio macro

# def f1_score(y_true, y_pred):
#     precisions = precision(y_true, y_pred)
#     recalls = recall(y_true, y_pred)
#     return 2 * (precisions * recalls) / (precisions + recalls) if (precisions + recalls) > 0 else 0

def f1_score(y_true, y_pred):
    classes = np.unique(y_true)
    f1_scores = []
    for clase in classes:
        prec = precision_per_class(y_true, y_pred, clase)
        rec = recall_per_class(y_true, y_pred, clase)
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        f1_scores.append(f1)
    return np.mean(f1_scores)  # Promedio macro

def precision_per_class(y_true, y_pred, clase):
    tp = sum(1 for vt, vp in zip(y_true, y_pred) if vt == clase and vp == clase)
    fp = sum(1 for vt, vp in zip(y_true, y_pred) if vt != clase and vp == clase)
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall_per_class(y_true, y_pred, clase):
    tp = sum(1 for vt, vp in zip(y_true, y_pred) if vt == clase and vp == clase)
    fn = sum(1 for vt, vp in zip(y_true, y_pred) if vt == clase and vp != clase)
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def accuracy(y_true, y_pred):
    correctas = sum(1 for vt, vp in zip(y_true, y_pred) if vt == vp)
    return correctas / len(y_true)


def auc_roc(y_true, y_prob):
    """
    Calcula el AUC-ROC para un problema multiclase usando One-vs-Rest.
    
    Argumentos:
    - y_true: Vector con las clases verdaderas.
    - y_prob: Matriz con las probabilidades predichas (una columna por clase).
    
    Retorna:
    - AUC-ROC promedio.
    """
    classes = np.unique(y_true)  # Las clases presentes en y_true
    auc_roc_values = []

    for clase in classes:
        # Convertir el problema en One-vs-Rest para cada clase
        y_true_binary = (y_true == clase).astype(int)
        y_prob_class = y_prob[:, int(clase)]
        
        # Calcular AUC-ROC para esta clase
        auc_roc_class = roc_auc_score(y_true_binary, y_prob_class)
        auc_roc_values.append(auc_roc_class)
    
    # Retornar el promedio de AUC-ROC para todas las clases
    return np.mean(auc_roc_values)


def auc_pr(y_true, y_prob):
    """
    Calcula el AUC-PR para un problema multiclase usando One-vs-Rest.
    
    Argumentos:
    - y_true: Vector con las clases verdaderas.
    - y_prob: Matriz con las probabilidades predichas (una columna por clase).
    
    Retorna:
    - AUC-PR promedio.
    """
    classes = np.unique(y_true)  # Las clases presentes en y_true
    auc_pr_values = []

    for clase in classes:
        # Convertir el problema en One-vs-Rest para cada clase
        y_true_binary = (y_true == clase).astype(int)
        y_prob_class = y_prob[:, int(clase)]
        
        # Calcular AUC-PR para esta clase
        auc_pr_class = average_precision_score(y_true_binary, y_prob_class)
        auc_pr_values.append(auc_pr_class)
    
    # Retornar el promedio de AUC-PR para todas las clases
    return np.mean(auc_pr_values)


def plot_combined_roc_curve(y_true, y_prob_list, nombre_modelo):
    plt.figure(figsize=(8, 6))
    
    model_names = ["Regresión Logística", "LDA", "Random Forest"]  # Puedes modificar los nombres si es necesario
    
    for idx, y_prob in enumerate(y_prob_list):
        for clase in np.unique(y_true):
            y_true_binary = (y_true == clase).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, int(clase)])
            auc_value = roc_auc_score(y_true_binary, y_prob[:, int(clase)])
            plt.plot(fpr, tpr, label=f'{model_names[idx]} - Clase {clase} (AUC = {auc_value:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal de referencia
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Curva ROC para {nombre_modelo}')
    plt.legend(loc="lower right")
    plt.show()

def plot_combined_pr_curve(y_true, y_prob_list, nombre_modelo):
    plt.figure(figsize=(8, 6))
    
    model_names = ["Logistic Regression", "LDA", "Random Forest"]  # Puedes modificar los nombres si es necesario
    
    for idx, y_prob in enumerate(y_prob_list):
        for clase in np.unique(y_true):
            y_true_binary = (y_true == clase).astype(int)
            # Aquí accedemos directamente a las probabilidades por clase
            precision_vals, recall_vals, _ = precision_recall_curve(y_true_binary, y_prob[:, int(clase)])
            auc_value = average_precision_score(y_true_binary, y_prob[:, int(clase)])
            plt.plot(recall_vals, precision_vals, label=f'{model_names[idx]} - Clase {clase} (AUC PR = {auc_value:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Curva Precision-Recall para {nombre_modelo}')
    plt.legend(loc="lower left")
    plt.show()

def save_metrics(y_true, y_pred, y_prob):
    metrics = {}
    metrics["Accuracy"] = accuracy(y_true, y_pred)
    metrics["Precision"] = precision(y_true, y_pred)  # Ahora es un promedio macro
    metrics["Recall"] = recall(y_true, y_pred)
    metrics["F1-Score"] = f1_score(y_true, y_pred)
    metrics["AUC-ROC"] = auc_roc(y_true, y_prob)  # Promedio macro
    metrics["AUC-PR"] = auc_pr(y_true, y_prob)
    return metrics




def print_table(resultados):
    print(f"{'Modelo':<20}{'Accuracy':<10}{'Precision':<10}{'Recall':<10}{'F1-Score':<10}{'AUC-ROC':<10}{'AUC-PR':<10}")
    for modelo, metricas in resultados.items():
        print(f"{modelo:<20}{metricas['Accuracy']:<10.4f}{metricas['Precision']:<10.4f}{metricas['Recall']:<10.4f}{metricas['F1-Score']:<10.4f}{metricas['AUC-ROC']:<10.4f}{metricas['AUC-PR']:<10.4f}")

def print_matrix(matriz, nombre_modelo):
    plt.figure(figsize=(6, 4))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Pred. 0", "Pred. 1", "Pred. 2"], 
                yticklabels=["True 0", "True 1", "True 2"])
    plt.title(f'Matriz de Confusión para {nombre_modelo}')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Verdadero')
    plt.show()
