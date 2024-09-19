import os
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Cargar el dataset desde un archivo CSV usando solo numpy.
    """
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    X = data[:, :-1]  
    y = data[:, -1]  
    return X, y

def save_data(X, y, file_path):
    """
    Guardar el dataset en un archivo CSV.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    data = np.hstack((X, y.reshape(-1, 1)))
    np.savetxt(file_path, data, delimiter=',', fmt='%f')
    print(f"Datos guardados en: {file_path}")

def split_train_validation(X, y, validation_size=0.2, random_state=42):
    """
    Divide los datos en conjuntos de entrenamiento y validación.
    """
    return train_test_split(X, y, test_size=validation_size, random_state=random_state)

if __name__ == "__main__":
    dataset_file = "/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/raw/breast_cancer_dev.csv"
    
    # Cargar los datos
    X, y = load_data(dataset_file)
    
    # Dividir en conjuntos de train y validation (80% train, 20% validation)
    X_train, X_validation, y_train, y_validation = split_train_validation(X, y, validation_size=0.2)

    # Guardar los conjuntos de train y validation
    save_data(X_train, y_train, "problema1/data/preprocessed/train.csv")
    save_data(X_validation, y_validation, "problema1/data/preprocessed/validation.csv")
