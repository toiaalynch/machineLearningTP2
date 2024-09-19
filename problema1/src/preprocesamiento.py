import os
import numpy as np

def load_data(file_path):
    """
    Cargar el dataset desde un archivo CSV usando solo numpy.
    Mezclar aleatoriamente los datos después de cargarlos.
    """
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)

    X = data[:, :-1]  
    y = data[:, -1]  

    return X, y

def shuffle_data(X, y):
    """
    Mezclar aleatoriamente los datos (X, y).
    """
    indices = np.random.permutation(len(y))  
    return X[indices], y[indices]

def standardize(X):
    """
    Estandariza las características (X) para que tengan media 0 y desviación estándar 1.
    Excluye la última columna que es la etiqueta de predicción binaria.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1
    return (X - mean) / std

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
    majority_class = X[y == 0]
    minority_class = X[y == 1]

    majority_downsampled_idx = np.random.choice(len(majority_class), size=len(minority_class), replace=False)
    majority_downsampled = majority_class[majority_downsampled_idx]

    X_resampled = np.vstack((majority_downsampled, minority_class))
    y_resampled = np.hstack((np.zeros(len(majority_downsampled)), np.ones(len(minority_class))))

    return shuffle_data(X_resampled, y_resampled)

def oversampling_duplication(X, y):
    """
    Realizar oversampling mediante duplicación de la clase minoritaria.
    """
    majority_class = X[y == 0]
    minority_class = X[y == 1]

    minority_upsampled_idx = np.random.choice(len(minority_class), size=len(majority_class), replace=True)
    minority_upsampled = minority_class[minority_upsampled_idx]

    X_resampled = np.vstack((majority_class, minority_upsampled))
    y_resampled = np.hstack((np.zeros(len(majority_class)), np.ones(len(minority_upsampled))))

    return shuffle_data(X_resampled, y_resampled)


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def find_k_nearest_neighbors(X, point, k=8):
    distances = []
    
    for i in range(len(X)):
        distance = euclidean_distance(point, X[i])
        distances.append((i, distance))
    
    distances.sort(key=lambda x: x[1])
    
    neighbors = [idx for idx, _ in distances[1:k+1]]  
    
    return neighbors


def oversampling_smote(X, y, k=8):
    minority_class = X[y == 1]

    majority_class_size = np.sum(y == 0)
    minority_class_size = len(minority_class)
    num_new_samples = majority_class_size - minority_class_size

    new_samples = []
    for _ in range(num_new_samples):
        idx = np.random.randint(0, len(minority_class))
        point = minority_class[idx]

        neighbors = find_k_nearest_neighbors(minority_class, point, k)
        
        neighbor_idx = np.random.choice(neighbors)
        neighbor = minority_class[neighbor_idx]

        diff = neighbor - point
        new_sample = point + np.random.rand() * diff
        new_samples.append(new_sample)

    new_samples = np.array(new_samples)

    X_resampled = np.vstack((X, new_samples))
    y_resampled = np.hstack((y, np.ones(len(new_samples))))

    return shuffle_data(X_resampled, y_resampled)



def preprocess_data(file_path, target_col_idx, method="undersampling"):
    """
    Preprocesar el dataset según el método especificado y guardarlo en 'data/processed/'.
    method puede ser:
    - "undersampling"
    - "oversampling_duplication"
    - "oversampling_smote"
    """
    X, y = load_data(file_path)

    X = standardize(X)

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
        raise ValueError("Método no válido. Usa: 'undersampling', 'oversampling_duplication', 'oversampling_smote'.")

    processed_data = np.hstack((X_resampled, y_resampled.reshape(-1, 1)))
    save_data(processed_data, output_filename)

if __name__ == "__main__":
    dataset_file = "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/preprocessed/train.csv"
    target_column_idx = -1  # Índice de la columna objetivo (la última columna)

    preprocess_data(dataset_file, target_column_idx, method="undersampling")
    preprocess_data(dataset_file, target_column_idx, method="oversampling_duplication")
    preprocess_data(dataset_file, target_column_idx, method="oversampling_smote")


def cargar_y_contar_clases(file_path):
    """
    Cargar el dataset desde un archivo CSV y contar la cantidad de 0s y 1s en la columna objetivo.
    """
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    y = data[:, -1]  

    unique, counts = np.unique(y, return_counts=True)
    clase_count = dict(zip(unique, counts))
    
    print(f"Distribución de clases para {file_path}:")
    print(f"Clase 0: {clase_count.get(0.0, 0)} muestras")
    print(f"Clase 1: {clase_count.get(1.0, 0)} muestras")
    print("\n")

datasets = {
    "Raw Data": "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/raw/breast_cancer_dev.csv",
    "Undersampling": "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/preprocessed/data_undersampling.csv",
    "Oversampling Duplication": "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/preprocessed/data_oversampling_duplication.csv",
    "Oversampling SMOTE": "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/preprocessed/data_oversampling_smote.csv",
}

for nombre, file_path in datasets.items():
    cargar_y_contar_clases(file_path)



# # === Verificando el dataset: Raw Data ===
# # Distribución de clases para /Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/raw/breast_cancer_dev.csv:
# # Clase 0: 302 muestras
# # Clase 1: 86 muestras

# # === Verificando el dataset: Undersampling ===
# # Distribución de clases para /Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/preprocessed/data_undersampling.csv:
# # Clase 0: 85 muestras
# # Clase 1: 86 muestras


# # === Verificando el dataset: Oversampling Duplication ===
# # Distribución de clases para /Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/preprocessed/data_oversampling_duplication.csv:
# # Clase 0: 301 muestras
# # Clase 1: 302 muestras


# # === Verificando el dataset: Oversampling SMOTE ===
# # Distribución de clases para /Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/preprocessed/data_oversampling_smote.csv:
# # Clase 0: 302 muestras
# # Clase 1: 301 muestras




