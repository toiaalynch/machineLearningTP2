import os
import numpy as np

def load_data(file_path):
    """
    Cargar el dataset desde un archivo CSV usando solo numpy.
    """
    return np.genfromtxt(file_path, delimiter=',', skip_header=1)

def save_data(data, file_path):
    """
    Guardar el dataset procesado en la carpeta de data/processed.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.savetxt(file_path, data, delimiter=',', fmt='%f')
    print(f"Datos guardados en: {file_path}")

def undersampling(X, y):
    """
    Realizar undersampling de la clase mayoritaria.
    """
    # Separar clases
    majority_class = X[y == 0]
    minority_class = X[y == 1]

    # Seleccionar aleatoriamente tantas muestras de la clase mayoritaria como hay en la clase minoritaria
    majority_downsampled_idx = np.random.choice(len(majority_class), size=len(minority_class), replace=False)
    majority_downsampled = majority_class[majority_downsampled_idx]

    # Combinar y devolver
    X_resampled = np.vstack((majority_downsampled, minority_class))
    y_resampled = np.hstack((np.zeros(len(majority_downsampled)), np.ones(len(minority_class))))

    return X_resampled, y_resampled

def oversampling_duplication(X, y):
    """
    Realizar oversampling mediante duplicación de la clase minoritaria.
    """
    # Separar clases
    majority_class = X[y == 0]
    minority_class = X[y == 1]

    # Duplicar aleatoriamente muestras de la clase minoritaria hasta que ambas clases tengan igual proporción
    minority_upsampled_idx = np.random.choice(len(minority_class), size=len(majority_class), replace=True)
    minority_upsampled = minority_class[minority_upsampled_idx]

    # Combinar y devolver
    X_resampled = np.vstack((majority_class, minority_upsampled))
    y_resampled = np.hstack((np.zeros(len(majority_class)), np.ones(len(minority_upsampled))))

    return X_resampled, y_resampled

def oversampling_smote(X, y, k=5):
    """
    Realizar oversampling utilizando SMOTE (versión simplificada).
    Generar ejemplos sintéticos a partir de k vecinos más cercanos.
    """
    minority_class = X[y == 1]

    # Calcular cuántos ejemplos nuevos necesitamos
    majority_class_size = np.sum(y == 0)
    minority_class_size = len(minority_class)
    num_new_samples = majority_class_size - minority_class_size

    # Crear nuevos ejemplos sintéticos
    new_samples = []
    for _ in range(num_new_samples):
        # Elegir un punto de la clase minoritaria aleatoriamente
        idx = np.random.randint(0, len(minority_class))
        point = minority_class[idx]

        # Encontrar los k vecinos más cercanos (versión simplificada: elegimos aleatoriamente en lugar de k vecinos reales)
        neighbor_idx = np.random.randint(0, len(minority_class))
        neighbor = minority_class[neighbor_idx]

        # Generar un nuevo punto sintético interpolando entre el punto y su vecino
        diff = neighbor - point
        new_sample = point + np.random.rand() * diff
        new_samples.append(new_sample)

    new_samples = np.array(new_samples)

    # Combinar los nuevos ejemplos con el dataset original
    X_resampled = np.vstack((X, new_samples))
    y_resampled = np.hstack((y, np.ones(len(new_samples))))

    return X_resampled, y_resampled


def preprocess_data(file_path, target_col_idx, method="undersampling"):
    """
    Preprocesar el dataset según el método especificado y guardarlo en 'data/processed/'.
    method puede ser:
    - "undersampling"
    - "oversampling_duplication"
    - "oversampling_smote"
    """
    data = load_data(file_path)
    X = data[:, :-1]
    y = data[:, target_col_idx]

    if method == "undersampling":
        X_resampled, y_resampled = undersampling(X, y)
        output_filename = "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/preprocessed/data_undersampling.csv"
    elif method == "oversampling_duplication":
        X_resampled, y_resampled = oversampling_duplication(X, y)
        output_filename = "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/preprocessed/data_oversampling_duplication.csv"
    elif method == "oversampling_smote":
        X_resampled, y_resampled = oversampling_smote(X, y)
        output_filename = "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/preprocessed/data_oversampling_smote.csv"
    else:
        raise ValueError("Método no válido. Usa: 'undersampling', 'oversampling_duplication', 'oversampling_smote' o 'cost_reweighting'.")

    processed_data = np.hstack((X_resampled, y_resampled.reshape(-1, 1)))
    save_data(processed_data, output_filename)

# Ejemplo de uso:
if __name__ == "__main__":
    dataset_file = "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/raw/breast_cancer_dev.csv"
    target_column_idx = -1  # Índice de la columna objetivo (la última columna)

    preprocess_data(dataset_file, target_column_idx, method="undersampling")
    preprocess_data(dataset_file, target_column_idx, method="oversampling_duplication")
    preprocess_data(dataset_file, target_column_idx, method="oversampling_smote")

# SMOTE: La implementación aquí es una versión simplificada. En lugar de calcular los
#  k vecinos más cercanos, selecciona puntos de la clase minoritaria aleatoriamente 
# para la interpolación. Esto puede no ser tan preciso como la versión estándar de SMOTE, 
# pero es una solución viable utilizando solo numpy.

def cargar_y_contar_clases(file_path):
    """
    Cargar el dataset desde un archivo CSV y contar la cantidad de 0s y 1s en la columna objetivo.
    """
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    y = data[:, -1]  # La última columna es la columna de las etiquetas

    # Contar la cantidad de 0s y 1s
    unique, counts = np.unique(y, return_counts=True)
    clase_count = dict(zip(unique, counts))
    
    # Mostrar los resultados
    print(f"Distribución de clases para {file_path}:")
    print(f"Clase 0: {clase_count.get(0.0, 0)} muestras")
    print(f"Clase 1: {clase_count.get(1.0, 0)} muestras")
    print("\n")

# Archivos preprocesados
datasets = {
    "Raw Data": "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/raw/breast_cancer_dev.csv",
    "Undersampling": "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/preprocessed/data_undersampling.csv",
    "Oversampling Duplication": "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/preprocessed/data_oversampling_duplication.csv",
    "Oversampling SMOTE": "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/preprocessed/data_oversampling_smote.csv",
}

# Verificar la cantidad de 0s y 1s en cada dataset
for nombre, file_path in datasets.items():
    print(f"=== Verificando el dataset: {nombre} ===")
    cargar_y_contar_clases(file_path)


# === Verificando el dataset: Undersampling ===
# Distribución de clases para /Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/preprocessed/data_undersampling.csv:
# Clase 0: 85 muestras
# Clase 1: 86 muestras


# === Verificando el dataset: Oversampling Duplication ===
# Distribución de clases para /Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/preprocessed/data_oversampling_duplication.csv:
# Clase 0: 301 muestras
# Clase 1: 302 muestras


# === Verificando el dataset: Oversampling SMOTE ===
# Distribución de clases para /Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/preprocessed/data_oversampling_smote.csv:
# Clase 0: 302 muestras
# Clase 1: 301 muestras