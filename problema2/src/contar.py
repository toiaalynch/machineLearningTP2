import numpy as np

# Función para cargar los datos y contar las instancias de cada clase
def cargar_datos_y_contar_clases(file_path):
    print(f"Cargando datos desde: {file_path}")
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)  # Cargar datos desde archivo CSV
    y = data[:, 0]  # Las etiquetas están en la primera columna
    
    # Contar el número de instancias de cada clase
    clases, conteo = np.unique(y, return_counts=True)
    
    # Mostrar el resultado
    print("Distribución de clases:")
    for clase, cantidad in zip(clases, conteo):
        print(f"Clase {int(clase)}: {cantidad} instancias")
    
    return clases, conteo

# Ruta del archivo
file_train = '/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema2/data/processed/diabetes_train_oversampled.csv'

# Ejecutar la función
cargar_datos_y_contar_clases(file_train)
