import numpy as np

class LogisticRegressionL2:
    def __init__(self, learning_rate=0.01, lambda_=0.1, num_iters=1000):
        # TENGO QUE RE VER EL LAMDA
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.num_iters = num_iters
        self.theta = None
        self.cost_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, y):
        m = len(y)
        h = self.sigmoid(X @ self.theta)
        
        # Costo sin regularización
        cost = (-1/m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))
        
        # Regularización (sin incluir theta_0)
        reg_cost = (self.lambda_ / (2 * m)) * np.sum(np.square(self.theta[1:]))
        
        # Asegurarse de que cost sea un escalar
        return np.sum(cost) + reg_cost
    
    def compute_gradient(self, X, y):
        m = len(y)
        h = self.sigmoid(X @ self.theta)
        gradient = (1/m) * (X.T @ (h - y))
        gradient[1:] += (self.lambda_ / m) * self.theta[1:]
        return gradient

    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros((n, 1))  
        y = y.reshape(-1, 1)  
        
        for i in range(self.num_iters):
            cost = self.compute_cost(X, y)
            gradient = self.compute_gradient(X, y)
            
            self.theta -= self.learning_rate * gradient
            self.cost_history.append(cost)
            
            if i % 100 == 0:
                print(f"Iteración {i}: Costo = {cost:.4f}")
    
    def predict(self, X):
        prob = self.sigmoid(X @ self.theta)
        return (prob >= 0.5).astype(int)

    def predict_proba(self, X):
        return self.sigmoid(X @ self.theta)


