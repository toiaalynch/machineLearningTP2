import numpy as np
from data_splitting import split_train_validation
from modelo import LogisticRegressionL2
from metrics import guardar_metricas, imprimir_tabla_resultados


# Cargar los datos desde un archivo CSV
def cargar_datos(file_path):
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)  # Ajusta la ruta y el delimitador según tu archivo
    X = data[:, :-1]  # Todas las columnas menos la última son las características
    y = data[:, -1]   # La última columna es la etiqueta
    return X, y

def evaluar_modelo(file_path, modelo):
    # Cargar los datos
    X, y = cargar_datos(file_path)
    
    # Hacer el split en entrenamiento y prueba (80-20)
    X_train, X_val, y_train, y_val = split_train_validation(X, y, 0.2, 42)
    
    # Entrenar el modelo de regresión logística
    modelo.fit(X_train, y_train)
    
    # Hacer predicciones en los datos de prueba
    y_pred = modelo.predict(X_val)
    y_prob = modelo.predict_proba(X_val)
    
    # Guardar las métricas
    return guardar_metricas(y_val, y_pred, y_prob)

def main():
    # Definir el modelo
    modelo = LogisticRegressionL2(learning_rate=0.01, lambda_=0.1, num_iters=1000)

    # Archivos preprocesados
    datasets = {
        "Raw Data": "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/raw/breast_cancer_dev.csv",
        "Undersampling": "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/preprocessed/data_undersampling.csv",
        "Oversampling Duplication": "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/preprocessed/data_oversampling_duplication.csv",
        "Oversampling SMOTE": "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/preprocessed/data_oversampling_smote.csv",
        "Cost Re-weighting": "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/raw/breast_cancer_dev.csv"
    }

    # Evaluar el modelo en cada dataset y almacenar resultados
    resultados = {}
    for nombre, file_path in datasets.items():
        resultados[nombre] = evaluar_modelo(file_path, modelo)
    
    # Imprimir los resultados comparativos
    imprimir_tabla_resultados(resultados)

# Ejecutar el main si se llama desde la línea de comandos
if __name__ == "__main__":
    main()
