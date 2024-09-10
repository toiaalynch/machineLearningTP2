# import numpy as np
# from data_splitting import split_train_validation
# from modelo import LogisticRegressionL2
# from metrics import guardar_metricas, imprimir_tabla_resultados, plot_roc_curves, plot_pr_curves


# def cargar_datos(file_path):
#     data = np.loadtxt(file_path, delimiter=',', skiprows=1)  
#     X = data[:, :-1]  
#     y = data[:, -1]   
#     return X, y


# def evaluar_modelo(file_path, modelo):
#     X, y = cargar_datos(file_path)
    
#     X_train, X_val, y_train, y_val = split_train_validation(X, y, 0.2, 42)
    
#     modelo.fit(X_train, y_train)
#     y_pred = modelo.predict(X_val)
#     y_prob = modelo.predict_proba(X_val)
#     print(f"Predicciones para {file_path} (primeras 10): {y_pred[:10].flatten()}")
#     print(f"Probabilidades para {file_path} (primeras 10): {y_prob[:10].flatten()}")

#     metricas = guardar_metricas(y_val, y_pred, y_prob)
#     return metricas, y_val, y_prob

# def main():
#     modelo = LogisticRegressionL2(learning_rate=0.01, lambda_=0.01, num_iters=10000)

#     datasets = {
#         "Raw Data": "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/raw/breast_cancer_dev.csv",
#         "Undersampling": "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/preprocessed/data_undersampling.csv",
#         "Oversampling Duplication": "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/preprocessed/data_oversampling_duplication.csv",
#         "Oversampling SMOTE": "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/preprocessed/data_oversampling_smote.csv",
#         "Cost Re-weighting": "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/raw/breast_cancer_dev.csv"
#     }

#     resultados = {}
#     y_trues = {}
#     y_probs = {}

#     for nombre, file_path in datasets.items():
#         metricas, y_val, y_prob = evaluar_modelo(file_path, modelo)
        
#         resultados[nombre] = metricas
        
#         y_trues[nombre] = y_val
#         y_probs[nombre] = y_prob
    
#     imprimir_tabla_resultados(resultados)

#     # plot_roc_curves(list(datasets.keys()), y_trues["Raw Data"], list(y_probs.values()))
#     # plot_pr_curves(list(datasets.keys()), y_trues["Raw Data"], list(y_probs.values()))

# # Ejecutar el main si se llama desde la línea de comandos
# if __name__ == "__main__":
#     main()









































import numpy as np
from data_splitting import split_train_validation
from modelo import LogisticRegressionL2
from metrics import guardar_metricas, imprimir_tabla_resultados, plot_roc_curves, plot_pr_curves


def cargar_datos(file_path):
    print(f"Cargando datos desde: {file_path}")
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)  
    print(f"Datos cargados: {data.shape[0]} filas, {data.shape[1]} columnas")
    X = data[:, :-1]  
    y = data[:, -1]   
    print(f"Características (X): {X.shape}, Etiquetas (y): {y.shape}")
    return X, y


def evaluar_modelo(file_path, modelo):
    print(f"\nEvaluando modelo en dataset: {file_path}")
    
    # Cargar datos
    X, y = cargar_datos(file_path)
    
    # Dividir datos en entrenamiento y validación
    print("Dividiendo datos en entrenamiento y validación (80-20)...")
    X_train, X_val, y_train, y_val = split_train_validation(X, y, 0.2, 42)
    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}")
    print(f"y_train: {y_train.shape}, y_val: {y_val.shape}")
    
    # Entrenar el modelo
    print("Entrenando el modelo...")
    modelo.fit(X_train, y_train)
    
    # Predecir en datos de validación
    y_pred = modelo.predict(X_val)
    y_prob = modelo.predict_proba(X_val)
    print(f"Predicciones (primeras 10): {y_pred[:10].flatten()}")
    print(f"Probabilidades (primeras 10): {y_prob[:10].flatten()}")

    # Guardar métricas
    metricas = guardar_metricas(y_val, y_pred, y_prob)
    print(f"Métricas calculadas para {file_path}: {metricas}")
    
    return metricas, y_val, y_prob

def main():
    print("Inicializando el modelo con learning_rate=0.01, lambda_=0.01, num_iters=10000")
    modelo = LogisticRegressionL2(learning_rate=0.01, lambda_=0.01, num_iters=10000)

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


