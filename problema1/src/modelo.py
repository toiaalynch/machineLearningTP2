import numpy as np

class LogisticRegressionL2:
    def __init__(self, learning_rate, lambda_, num_iters):
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

            # Actualización de los parámetros
            self.theta -= self.learning_rate * gradient
            
            # Almacenar el costo en la lista
            self.cost_history.append(cost)

            # Mostrar el costo cada 100 iteraciones (opcional)
            # if i % 100 == 0:
            #     print(f"Iteración {i}: Costo = {cost:.4f}")


    
    def predict(self, X):
        prob = self.sigmoid(X @ self.theta)
        return (prob >= 0.5).astype(int)

    def predict_proba(self, X):
        return self.sigmoid(X @ self.theta)


















































# from sklearn.base import BaseEstimator
# import numpy as np

# class LogisticRegressionL2(BaseEstimator):
#     def __init__(self, learning_rate=0.01, lambda_=0.1, num_iters=1000):
#         self.learning_rate = learning_rate
#         self.lambda_ = lambda_
#         self.num_iters = num_iters
#         self.theta = None
#         self.cost_history = []

#     def sigmoid(self, z):
#         return 1 / (1 + np.exp(-z))

#     def compute_cost(self, X, y):
#         m = len(y)
#         h = self.sigmoid(X @ self.theta)
#         cost = (-1/m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))
#         reg_cost = (self.lambda_ / (2 * m)) * np.sum(np.square(self.theta[1:]))
#         return np.sum(cost) + reg_cost

#     def compute_gradient(self, X, y):
#         m = len(y)
#         h = self.sigmoid(X @ self.theta)
#         gradient = (1/m) * (X.T @ (h - y))
#         gradient[1:] += (self.lambda_ / m) * self.theta[1:]
#         return gradient

#     def fit(self, X, y):
#         m, n = X.shape
#         self.theta = np.zeros((n, 1))  
#         y = y.reshape(-1, 1)  

#         for i in range(self.num_iters):
#             gradient = self.compute_gradient(X, y)
#             self.theta -= self.learning_rate * gradient

#     def predict_proba(self, X):
#         return self.sigmoid(X @ self.theta)

#     def predict(self, X):
#         return (self.predict_proba(X) >= 0.5).astype(int)

# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import make_scorer, accuracy_score

# # Definir el modelo
# model = LogisticRegressionL2()

# # Definir los hiperparámetros a buscar
# param_grid = {
#     'learning_rate': [0.001, 0.01, 0.1],
#     'lambda_': [0.01, 0.1, 1.0],
#     'num_iters': [1000, 5000, 10000]
# }

# # Definir el 'scorer' (métrica de evaluación)
# scorer = make_scorer(accuracy_score)

# # Definir el GridSearch
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scorer, cv=5)
# def cargar_datos(file_path):
#     data = np.loadtxt(file_path, delimiter=',', skiprows=1)  
#     X = data[:, :-1]  
#     y = data[:, -1]   
#     return X, y

# # Cargar los datos
# X, y = cargar_datos('/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema1/data/raw/breast_cancer_dev.csv')

# # Ajustar el GridSearch
# grid_search.fit(X, y)

# # Imprimir los mejores hiperparámetros encontrados
# print("Mejores hiperparámetros encontrados:", grid_search.best_params_)

# # Imprimir la mejor puntuación
# print("Mejor precisión obtenida:", grid_search.best_score_)
