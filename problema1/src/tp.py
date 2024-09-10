import numpy as np

class LogisticRegressionL2:
    def __init__(self, learning_rate=0.01, lambda_=0.1, num_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.num_iters = num_iters
        self.theta = None
        self.cost_history = []

    # Función sigmoide
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Función de costo con regularización L2
    def compute_cost(self, X, y):
        m = len(y)
        h = self.sigmoid(X @ self.theta)
        # Costo sin regularización
        cost = (-1/m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))
        # Regularización (sin incluir theta_0)
        reg_cost = (self.lambda_ / (2 * m)) * np.sum(np.square(self.theta[1:]))
        return cost + reg_cost

    # Función para calcular el gradiente con regularización L2
    def compute_gradient(self, X, y):
        m = len(y)
        h = self.sigmoid(X @ self.theta)
        # Gradiente sin regularización
        gradient = (1/m) * (X.T @ (h - y))
        # Agregar regularización (sin incluir theta_0)
        gradient[1:] += (self.lambda_ / m) * self.theta[1:]
        return gradient

    # Función para entrenar el modelo
    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros((n, 1))  # Inicializar pesos
        y = y.reshape(-1, 1)  # Asegurarse de que y tenga la forma correcta
        
        for i in range(self.num_iters):
            # Calcular costo y gradiente
            cost = self.compute_cost(X, y)
            gradient = self.compute_gradient(X, y)
            
            # Actualizar pesos
            self.theta -= self.learning_rate * gradient
            
            # Guardar historial del costo
            self.cost_history.append(cost)
            
            if i % 100 == 0:
                print(f"Iteración {i}: Costo = {cost:.4f}")
    
    # Función para predecir las etiquetas
    def predict(self, X):
        prob = self.sigmoid(X @ self.theta)
        return (prob >= 0.5).astype(int)

    # Función para predecir probabilidades
    def predict_proba(self, X):
        return self.sigmoid(X @ self.theta)

# Ejemplo de uso
# Cargar tus datos aquí (X_train, y_train)
# X_train debe incluir la columna de sesgo (bias) como primera columna
# y_train debe ser una columna con valores 0 o 1

# Inicializar el modelo
model = LogisticRegressionL2(learning_rate=0.01, lambda_=0.1, num_iters=10000)

