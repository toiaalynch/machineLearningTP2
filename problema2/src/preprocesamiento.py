import numpy as np
import os

def load_data(file_path):
    """
    Cargar el dataset desde un archivo CSV usando solo numpy.
    El índice 0 contiene las clases (etiquetas), mientras que las demás columnas son características.
    """
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)  # Cargar el CSV
    y = data[:, 0]  # La etiqueta está en la primera columna
    X = data[:, 1:]  # Las características están en las columnas restantes
    return X, y

def shuffle_data(X, y):
    """
    Mezclar aleatoriamente los datos (X, y).
    """
    indices = np.random.permutation(len(y))  
    return X[indices], y[indices]

def split_train_validation(X, y, validation_size=0.2, seed=42):
    """
    Dividir los datos en conjuntos de entrenamiento y validación.
    """
    np.random.seed(seed)
    shuffle_X, shuffle_y = shuffle_data(X, y)
    split_idx = int((1 - validation_size) * len(shuffle_y))
    X_train, X_val = shuffle_X[:split_idx], shuffle_X[split_idx:]
    y_train, y_val = shuffle_y[:split_idx], shuffle_y[split_idx:]
    return X_train, X_val, y_train, y_val

# def standardize(X):
#     """
#     Estandariza las características (X) para que tengan media 0 y desviación estándar 1.
#     """
#     mean = np.mean(X, axis=0)
#     std = np.std(X, axis=0)
#     std[std == 0] = 1
#     return (X - mean) / std

def save_data(X, y, file_path):
    """
    Guardar el dataset procesado en un archivo CSV.
    Las etiquetas (y) se colocan en la primera columna.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    data = np.hstack((y.reshape(-1, 1), X))  # Colocar 'y' en la primera columna
    np.savetxt(file_path, data, delimiter=',', fmt='%f')
    print(f"Datos guardados en: {file_path}")

def oversampling_duplication(X, y):
    """
    Realizar oversampling mediante duplicación de las clases minoritarias en comparación con la clase mayoritaria.
    """
    classes, class_counts = np.unique(y, return_counts=True)
    max_class_count = np.max(class_counts)  # Número de muestras en la clase mayoritaria

    # Inicializamos las listas para las nuevas muestras
    X_resampled = []
    y_resampled = []

    for clase in classes:
        X_class = X[y == clase]
        class_count = len(X_class)
        
        if class_count < max_class_count:
            # Realizar oversampling para la clase minoritaria
            oversample_idx = np.random.choice(class_count, size=max_class_count, replace=True)
            X_oversampled = X_class[oversample_idx]
            y_oversampled = np.full(max_class_count, clase)
        else:
            # No hacer oversampling si la clase ya tiene el número máximo de muestras
            X_oversampled = X_class
            y_oversampled = np.full(class_count, clase)

        # Agregar las muestras balanceadas a las listas
        X_resampled.append(X_oversampled)
        y_resampled.append(y_oversampled)

    # Convertimos las listas en arrays de NumPy y mezclamos los datos
    X_resampled = np.vstack(X_resampled)
    y_resampled = np.hstack(y_resampled)

    return shuffle_data(X_resampled, y_resampled)

def preprocess_data_and_save(file_path):
    """
    Cargar los datos, dividirlos en train y validation, aplicar oversampling al train y guardar los resultados.
    """
    # Cargar datos
    X, y = load_data(file_path)

    # Estandarizar los datos
    # X = standardize(X)

    # Dividir los datos en train y validation
    X_train, X_val, y_train, y_val = split_train_validation(X, y, validation_size=0.2, seed=42)

    # Guardar conjuntos de train y validation
    save_data(X_train, y_train, "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema2/data/processed/diabetes_train.csv")
    save_data(X_val, y_val, "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema2/data/processed/diabetes_validation.csv")

    # Aplicar oversampling por duplicación al conjunto de train
    X_train_oversampled, y_train_oversampled = oversampling_duplication(X_train, y_train)

    # Guardar el conjunto de train con oversampling
    save_data(X_train_oversampled, y_train_oversampled, "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema2/data/processed/diabetes_train_oversampled.csv")


if __name__ == "__main__":
    dataset_file = "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema2/data/diabetes_dev.csv"
    preprocess_data_and_save(dataset_file)

# # Clase 0: 170962
# # Clase 1: 3705
# # Clase 2: 28277
# # 0 = no diabetes, 1 = prediabetes, 2 = diabetes.


def cargar_y_contar_clases(file_path):
    """
    Cargar el dataset desde un archivo CSV y contar la cantidad de muestras de cada clase.
    """
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    y = data[:, 0]  # La etiqueta está en la primera columna
    
    unique, counts = np.unique(y, return_counts=True)
    clase_count = dict(zip(unique, counts))
    
    print(f"Distribución de clases para {file_path}:")
    for clase, count in clase_count.items():
        print(f"Clase {int(clase)}: {count} muestras")
    print("\n")


# Lista de archivos a verificar
files = {
    "Train Original": "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema2/data/processed/diabetes_train.csv",
    "Validation": "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema2/data/processed/diabetes_validation.csv",
    "Train SMOTE": "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema2/data/processed/diabetes_train_oversampled.csv"
}

# Contar clases en cada archivo
for nombre, file_path in files.items():
    print(f"Revisando el archivo: {nombre}")
    cargar_y_contar_clases(file_path)

# Revisando el archivo: Train Original
# Distribución de clases para /Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema2/data/processed/diabetes_train.csv:
# Clase 0: 136698 muestras
# Clase 1: 2949 muestras
# Clase 2: 22707 muestras


# Revisando el archivo: Validation
# Distribución de clases para /Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema2/data/processed/diabetes_validation.csv:
# Clase 0: 34262 muestras
# Clase 1: 756 muestras
# Clase 2: 5570 muestras


# Revisando el archivo: Train SMOTE
# Distribución de clases para /Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema2/data/processed/diabetes_train_oversampled.csv:
# Clase 0: 136699 muestras
# Clase 1: 136699 muestras
# Clase 2: 136698 muestras