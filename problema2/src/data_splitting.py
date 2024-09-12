import numpy as np

def split_train_validation(X, y, validation_size=0.2, seed=None):
    """
    Divide el conjunto de datos en entrenamiento (80%) y validación (20%).
    
    Parámetros:
    - X: numpy array, matriz de características.
    - y: numpy array, vector de etiquetas.
    - validation_size: tamaño del conjunto de validación (default 0.2 para 80-20).
    - seed: semilla opcional para asegurar reproducibilidad.
    
    Retorna:
    - X_train: conjunto de características para entrenamiento.
    - X_validation: conjunto de características para validación.
    - y_train: etiquetas para entrenamiento.
    - y_validation: etiquetas para validación.
    """
    if seed:
        np.random.seed(seed)  # Para hacer el proceso reproducible si se proporciona una semilla

    # Obtener el número total de muestras
    num_samples = X.shape[0]
    
    # Crear un arreglo con los índices y mezclarlos aleatoriamente
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    # Determinar el tamaño del conjunto de validación
    validation_size = int(num_samples * validation_size)
    
    # Separar los índices para entrenamiento y validación
    validation_indices = indices[:validation_size]
    train_indices = indices[validation_size:]
    
    # Crear los conjuntos de entrenamiento y validación
    X_train = X[train_indices]
    X_validation = X[validation_indices]
    y_train = y[train_indices]
    y_validation = y[validation_indices]
    if X.shape[0] != y.shape[0]:
        raise ValueError("El número de muestras en X y y debe ser igual.")

    
    return X_train, X_validation, y_train, y_validation
