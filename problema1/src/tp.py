import numpy as np
from data_splitting import split_train_validation
from modelo import LogisticRegressionL2
from metrics import save_metrics, imprimir_tabla_resultados, matriz_de_confusion, graficar_curvas_manual, mostrar_graficos, print_matrix


def cargar_datos(file_path):
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)  
    X = data[:, :-1]  
    y = data[:, -1]   
    return X, y

def calcular_pesos_clase(y):
    class_0_weight = np.sum(y == 1) / len(y)
    class_1_weight = np.sum(y == 0) / len(y)
    
    class_0_weight *= 1.8
    class_1_weight *= 0.5
    
    return {0: class_0_weight, 1: class_1_weight}
    

def evaluar_modelo(file_path, modelo):    
    X, y = cargar_datos(file_path)
    X_train, X_val, y_train, y_val = split_train_validation(X, y, 0.2, 42)
    modelo.fit(X_train, y_train)
    
    y_pred = modelo.predict(X_val)
    y_prob = modelo.predict_proba(X_val)
    metricas = save_metrics(y_val, y_pred, y_prob)
    
        # Calcular la matriz de confusión
    matriz_conf = matriz_de_confusion(y_val, y_pred)

    return metricas, y_val, y_prob, matriz_conf

def evaluar_modelo_test(file_path_test, modelo):    
    X_test, y_test = cargar_datos(file_path_test)
    
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)
    metricas = save_metrics(y_test, y_pred, y_prob)
    matriz_conf = matriz_de_confusion(y_test, y_pred)
    
    return metricas, y_test, y_prob, matriz_conf

def main():    
    datasets = {
        "Raw Data": "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/raw/breast_cancer_dev.csv",
        "Undersampling": "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/preprocessed/data_undersampling.csv",
        "Oversampling Duplication": "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/preprocessed/data_oversampling_duplication.csv",
        "Oversampling SMOTE": "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/preprocessed/data_oversampling_smote.csv",
        "Cost Re-weighting": "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/raw/breast_cancer_dev.csv"
    }

    # Evaluación con dataset de validación (entrenamiento)
    resultados = {}
    y_trues = {}
    y_probs = {}

    modelos_entrenados = {}

    for nombre, file_path in datasets.items():
        
        X, y = cargar_datos(file_path)
        
        if nombre == "Cost Re-weighting":
            class_weights = calcular_pesos_clase(y)
            modelo = LogisticRegressionL2(learning_rate=0.1, lambda_=1, num_iters=20000, class_weights=class_weights, threshold=0.40)
        if nombre == "Raw Data":
            modelo = LogisticRegressionL2(learning_rate=0.01, lambda_=1, num_iters=20000, threshold=0.4)
        if nombre == "Undersampling":
            modelo = LogisticRegressionL2(learning_rate=0.00001, lambda_=1, num_iters=10000, threshold=0.4)
        if nombre == "Oversampling Duplication":
            modelo = LogisticRegressionL2(learning_rate=0.00001, lambda_=1, num_iters=10000, threshold=0.4)
        if nombre == "Oversampling SMOTE":
            modelo = LogisticRegressionL2(learning_rate=0.00001, lambda_=1, num_iters=10000, threshold=0.4)
    
        metricas, y_val, y_prob, matriz_conf = evaluar_modelo(file_path, modelo)
        
        resultados[nombre] = metricas
        y_trues[nombre] = y_val
        y_probs[nombre] = y_prob

        # Guardamos los modelos entrenados para evaluación con test
        modelos_entrenados[nombre] = modelo

        # Imprimir matriz de confusión
        print_matrix(matriz_conf, nombre)

    # Imprimir resultados comparativos del dataset de validación
    print("\n=== Resultados Comparativos Dataset Validation ===")
    imprimir_tabla_resultados(resultados)

    # Evaluación con dataset de prueba (test)
    test_file = "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/raw/breast_cancer_test.csv"
    
    resultados_test = {}
    y_trues_test = {}
    y_probs_test = {}
    

    for nombre, modelo in modelos_entrenados.items():
        # Evaluar con el dataset de test usando los modelos ya entrenados
        metricas_test, y_test, y_prob_test, matriz_conf_test = evaluar_modelo_test(test_file, modelo)
        
        resultados_test[nombre] = metricas_test
        y_trues_test[nombre] = y_test
        y_probs_test[nombre] = y_prob_test
    
            # Imprimir matriz de confusión del test
        print_matrix(matriz_conf_test, f"{nombre} Test")

    # Imprimir resultados comparativos del dataset de test
    print("\n=== Resultados Comparativos Dataset Test ===")
    imprimir_tabla_resultados(resultados_test)

if __name__ == "__main__":
    main()

