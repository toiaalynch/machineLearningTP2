import numpy as np

class LogisticRegressionL2:
    def __init__(self, learning_rate, lambda_, num_iters, class_weights=None):
        """
        Agregar class_weights como argumento opcional.
        Si se proporciona, se usa para el Cost Re-weighting.
        """
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.num_iters = num_iters
        self.theta = None
        self.cost_history = []
        self.class_weights = class_weights  # Argumento opcional para los pesos de las clases

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, y):
        m = len(y)
        h = self.sigmoid(X @ self.theta)

        if self.class_weights:
            weights = np.where(y == 1, self.class_weights[1], self.class_weights[0])
            cost = (-1/m) * np.sum(weights * (y * np.log(h) + (1 - y) * np.log(1 - h)))
        else:
            cost = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        
        reg_cost = (self.lambda_ / (2 * m)) * np.sum(np.square(self.theta[1:]))
        return cost + reg_cost


    def compute_gradient(self, X, y):
        m = len(y)
        h = self.sigmoid(X @ self.theta)

        if self.class_weights:
            weights = np.where(y == 1, self.class_weights[1], self.class_weights[0])
            gradient = (1/m) * (X.T @ ((h - y) * weights))
        else:
            gradient = (1/m) * (X.T @ (h - y))

        gradient[1:] += (self.lambda_ / m) * self.theta[1:]
        return gradient

    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros((n, 1))  # Inicialización de parámetros
        y = y.reshape(-1, 1)  # Asegurar que y tenga la forma correcta

        for i in range(self.num_iters):
            cost = self.compute_cost(X, y)
            gradient = self.compute_gradient(X, y)

            # Actualizar parámetros
            self.theta -= self.learning_rate * gradient
            
            # Almacenar el costo en cada iteración
            self.cost_history.append(cost)

    def predict(self, X):
        prob = self.sigmoid(X @ self.theta)
        return (prob >= 0.5).astype(int)

    def predict_proba(self, X):
        return self.sigmoid(X @ self.theta)
