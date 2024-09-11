import numpy as np
from data_splitting import split_train_validation
from modelo import LogisticRegressionL2
from metrics import guardar_metricas, imprimir_tabla_resultados, plot_roc_curves, plot_pr_curves


def cargar_datos(file_path):
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)  
    X = data[:, :-1]  
    y = data[:, -1]   
    return X, y

def calcular_pesos_clase(y):
    class_0_weight = np.sum(y == 1) / len(y)
    class_1_weight = np.sum(y == 0) / len(y)
    
    class_0_weight *= 1.8 
    class_1_weight *= 0.4
    
    return {0: class_0_weight, 1: class_1_weight}


def evaluar_modelo(file_path, modelo):    
    # Cargar datos
    X, y = cargar_datos(file_path)
    
    # Dividir datos en entrenamiento y validación
    X_train, X_val, y_train, y_val = split_train_validation(X, y, 0.2, 42)
    
    # Entrenar el modelo
    modelo.fit(X_train, y_train)
    
    # Predecir en datos de validación
    y_pred = modelo.predict(X_val)
    y_prob = modelo.predict_proba(X_val)
    # Guardar métricas
    metricas = guardar_metricas(y_val, y_pred, y_prob)
    
    return metricas, y_val, y_prob

def main():    
    # Definir datasets
    datasets = {
        "Raw Data": "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/raw/breast_cancer_dev.csv",
        "Undersampling": "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/preprocessed/data_undersampling.csv",
        "Oversampling Duplication": "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/preprocessed/data_oversampling_duplication.csv",
        "Oversampling SMOTE": "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/preprocessed/data_oversampling_smote.csv",
        "Cost Re-weighting": "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/raw/breast_cancer_dev.csv"
    }

    resultados = {}
    y_trues = {}
    y_probs = {}

    # Evaluar modelo en cada dataset
    for nombre, file_path in datasets.items():
        print(f"\n=== Evaluando {nombre} ===")
        
        # Cargar los datos primero
        X, y = cargar_datos(file_path)
        
        # Si estamos evaluando Cost Re-weighting, calcular los pesos de las clases
        if nombre == "Cost Re-weighting":
            class_weights = calcular_pesos_clase(y)
            modelo = LogisticRegressionL2(learning_rate=0.01, lambda_=1, num_iters=10000, class_weights=class_weights)
        else:
            modelo = LogisticRegressionL2(learning_rate=0.01, lambda_=1, num_iters=10000, threshold=0.5)

        # Evaluar el modelo
        metricas, y_val, y_prob = evaluar_modelo(file_path, modelo)
        
        resultados[nombre] = metricas
        y_trues[nombre] = y_val
        y_probs[nombre] = y_prob
    
    # Imprimir resultados comparativos
    print("\n=== Resultados Comparativos ===")
    imprimir_tabla_resultados(resultados)

    # Descomentar para graficar curvas ROC y PR
    # plot_roc_curves(list(datasets.keys()), y_trues["Raw Data"], list(y_probs.values()))
    # plot_pr_curves(list(datasets.keys()), y_trues["Raw Data"], list(y_probs.values()))

# Ejecutar el main si se llama desde la línea de comandos
if __name__ == "__main__":
    main()



