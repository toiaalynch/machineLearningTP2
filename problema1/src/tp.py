import numpy as np
from data_splitting import split_train_validation
from modelo import LogisticRegressionL2
from metrics import guardar_metricas, imprimir_tabla_resultados


# Cargar los datos (aquí asumo que tienes los datos en un archivo CSV)
def cargar_datos():
    # Ejemplo para cargar datos de un CSV. Puedes adaptarlo a tu formato de datos.
    # Asumo que tienes las características (X) y etiquetas (y) separadas en el CSV.
    data = np.loadtxt('/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/raw/breast_cancer_dev.csv', delimiter=',', skiprows=1)  # Ajusta la ruta y el delimitador según tu archivo
    X = data[:, :-1]  # Todas las columnas menos la última son las características
    y = data[:, -1]   # La última columna es la etiqueta
    return X, y

def main():
    # Cargar los datos
    X, y = cargar_datos()
    
    # Hacer el split en entrenamiento y prueba (80-20)
    X_train, X_val, y_train, y_val = split_train_validation(X, y, 0.2, 42)
    
    # Entrenar el modelo de regresión logística
    modelo = LogisticRegressionL2(learning_rate=0.01, lambda_=0.1, num_iters=1000)
    modelo.fit(X_train, y_train)
    
    # Hacer predicciones en los datos de prueba
    y_pred = modelo.predict(X_val)
    y_prob = modelo.predict_proba(X_val)

        # Imprimir las probabilidades predichas y las etiquetas verdaderas
    print("Probabilidades predichas (y_prob):", y_prob)
    print("Etiquetas verdaderas (y_test):", y_val)

    
    # Guardar las métricas
    resultados = {}
    resultados['Logistic Regression L2'] = guardar_metricas(y_val, y_pred, y_prob)
    
    # Imprimir los resultados
    imprimir_tabla_resultados(resultados)

# Ejecutar el main si se llama desde la línea de comandos
if __name__ == "__main__":
    main()
