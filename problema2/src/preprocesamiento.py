import numpy as np

def contar_clases(y):
    """
    Cuenta cuántas observaciones hay de cada clase en la columna 'Diabetes'.
    
    Argumentos:
    y -- array o lista de etiquetas de clase (0, 1, 2)
    
    Retorna:
    Un diccionario con la cuenta de cada clase.
    """
    clases, conteos = np.unique(y, return_counts=True)
    resultado = dict(zip(clases, conteos))
    
    return resultado

# Ejemplo de uso:
# Cargar los datos (etiquetas están en la columna 'Diabetes', asumiendo que ya has cargado los datos con np.loadtxt)
file_path = '/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema2/data/diabetes_dev.csv'
data = np.loadtxt(file_path, delimiter=',', skiprows=1)
y = data[:, 0]  # Asumiendo que la primera columna es 'Diabetes'

# Contar cuántos hay de cada clase
resultado = contar_clases(y)
print(resultado)
