import numpy as np
from collections import Counter


class LogisticRegressionMulticlass:
    def __init__(self, alpha, lambda_, num_iters, num_labels):
        """
        Inicializa el modelo de regresión logística multinomial.
        
        Argumentos:
        alpha: Tasa de aprendizaje.
        lambda_: Parámetro de regularización L2.
        num_iters: Número de iteraciones para el descenso por gradiente.
        num_labels: Número de clases (para multinomial).
        """
        self.alpha = alpha
        self.lambda_ = lambda_
        self.num_iters = num_iters
        self.num_labels = num_labels
        self.all_theta = None
    
    # Función sigmoide
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    # Función de costo con regularización L2
    def compute_cost(self, X, y, theta):
        m = X.shape[0]
        h = self.sigmoid(np.dot(X, theta))
        h = np.clip(h, 1e-10, 1 - 1e-10)  # Limitar los valores de h para evitar log(0)
        cost = -(1/m) * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))
        reg_cost = (self.lambda_ / (2 * m)) * np.sum(np.square(theta[1:]))
        return cost + reg_cost
    
    # Gradiente de la función de costo con regularización L2
    def compute_gradient(self, X, y, theta):
        m = X.shape[0]
        h = self.sigmoid(np.dot(X, theta))
        # Gradiente sin regularización
        gradient = (1/m) * np.dot(X.T, (h - y))
        # Regularización L2 (excepto para theta[0])
        gradient[1:] += (self.lambda_ / m) * theta[1:]
        return gradient
    
    # Descenso por gradiente
    def gradient_descent(self, X, y, theta):
        for i in range(self.num_iters):
            gradient = self.compute_gradient(X, y, theta)
            theta = theta - self.alpha * gradient
            # Imprimir el costo cada 100 iteraciones
            if i % 100 == 0:
                cost = self.compute_cost(X, y, theta)
                print(f"Iteración {i}: Costo = {cost}")
        return theta
    
    # Preprocesar datos (Agregar columna de 1s para el bias)
    def preprocess_data(self, X):
        m = X.shape[0]
        X = np.concatenate([np.ones((m, 1)), X], axis=1)
        return X
    
    # Entrenamiento de la regresión logística multinomial
    def fit(self, X, y):
        """
        Entrena el modelo de regresión logística multinomial usando la técnica uno vs. resto.
        
        Argumentos:
        X: Matriz de características.
        y: Vector de etiquetas (clases).
        """
        m, n = X.shape
        self.all_theta = np.zeros((self.num_labels, n + 1))  # Una theta por cada clase
        
        # Preprocesamos los datos agregando la columna de bias
        X = self.preprocess_data(X)
        
        # Entrenamos una regresión logística binaria por cada clase
        for k in range(self.num_labels):
            y_k = np.where(y == k, 1, 0)  # Etiquetas binarias para la clase k
            theta_k = np.zeros(n + 1)  # Inicialización de theta para la clase k
            self.all_theta[k, :] = self.gradient_descent(X, y_k, theta_k)
    
    # Predicción
    def predict(self, X):
        """
        Realiza predicciones utilizando el modelo entrenado.
        
        Argumentos:
        X: Matriz de características de los datos de prueba.
        
        Retorna:
        Vector de predicciones (clase para cada ejemplo).
        """
        X = self.preprocess_data(X)
        h = self.sigmoid(np.dot(X, self.all_theta.T))
        return np.argmax(h, axis=1)



class LinearDiscriminantAnalysis:
    def __init__(self, num_components=None):
        """
        Inicializa el modelo de LDA.
        
        Argumentos:
        num_components: Número de componentes a utilizar (reducción de dimensionalidad). Si es None, usa el máximo posible.
        """
        self.num_components = num_components
        self.means_ = None  # Para almacenar las medias de cada clase
        self.W = None  # Para almacenar los vectores discriminantes

    def fit(self, X, y):
        """
        Ajusta el modelo LDA a los datos.
        
        Argumentos:
        X: Matriz de características de tamaño (m, n), donde m es el número de muestras y n es el número de características.
        y: Vector de etiquetas de tamaño (m,), donde cada valor corresponde a una clase.
        """
        n_features = X.shape[1]
        class_labels = np.unique(y)
        
        # Calcular la media global
        mean_overall = np.mean(X, axis=0)

        # Inicializar matrices de dispersión
        S_W = np.zeros((n_features, n_features))  # Matriz de dispersión intra-clase
        S_B = np.zeros((n_features, n_features))  # Matriz de dispersión entre-clases

        self.means_ = {}  # Diccionario para guardar las medias por clase
        
        for c in class_labels:
            X_c = X[y == c]  # Filtrar las muestras de la clase c
            mean_c = np.mean(X_c, axis=0)  # Media de la clase c
            self.means_[c] = mean_c  # Almacenar la media de la clase
            
            # Dispersión intra-clase para cada clase
            S_W += np.dot((X_c - mean_c).T, (X_c - mean_c))
            
            # Dispersión entre-clases
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            S_B += n_c * np.dot(mean_diff, mean_diff.T)

        # Resolver el problema de autovalores para (S_W^-1 S_B)
        A = np.linalg.inv(S_W).dot(S_B)
        eigenvalues, eigenvectors = np.linalg.eig(A)
        
        # Ordenar los autovalores y autovectores
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        
        # Seleccionar los primeros num_components autovectores
        if self.num_components:
            eigenvectors = eigenvectors[:self.num_components]

        # Guardar los vectores discriminantes (W)
        self.W = eigenvectors.T

    def transform(self, X):
        """
        Proyecta los datos en las nuevas direcciones discriminantes y retorna la parte real de los valores.
        
        Argumentos:
        X: Matriz de características a proyectar (muestras, características).
        
        Retorna:
        Matriz transformada con la parte real.
        """
        return np.real(np.dot(X, self.W))

    def predict(self, X):
        """
        Clasifica las muestras después de la proyección.
        
        Argumentos:
        X: Matriz de características de tamaño (muestras, características).
        
        Retorna:
        Vector de predicciones.
        """
        X_projected = self.transform(X)
        
        # Calcula la distancia euclidiana de los puntos proyectados a las medias de cada clase
        preds = [self._predict_sample(x) for x in X_projected]
        return np.array(preds)

    def _predict_sample(self, x):
        """
        Clasifica una muestra proyectada según la distancia a las medias de las clases.
        
        Argumentos:
        x: Muestra proyectada.
        
        Retorna:
        La clase predicha.
        """
        class_labels = list(self.means_.keys())
        distances = [np.linalg.norm(x - np.dot(self.means_[label], self.W)) for label in class_labels]
        return class_labels[np.argmin(distances)]



class DecisionTree:
    def __init__(self, max_depth=None, n_features=None):
        self.max_depth = max_depth
        self.n_features = n_features
        self.tree = None

    
    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        num_labels = len(np.unique(y))

        if depth >= self.max_depth or num_labels == 1:
            leaf_value = self._most_common_label(y)
            return leaf_value
        
        feat_idx, threshold = self._best_split(X, y)
        
        left_idx, right_idx = self._split(X[:, feat_idx], threshold)
        left = self._grow_tree(X[left_idx, :], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx, :], y[right_idx], depth + 1)
        return (feat_idx, threshold, left, right)

    def _best_split(self, X, y):
        best_gain = -1
        split_idx, split_threshold = None, None
        n_samples, n_features = X.shape

        feature_idxs = np.random.choice(n_features, self.n_features, replace=False)

        for feat_idx in feature_idxs:  
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(X_column, y, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    def _information_gain(self, X_column, y, threshold):
        left_idx, right_idx = self._split(X_column, threshold)
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        parent_entropy = self._entropy(y)
        n = len(y)
        n_left, n_right = len(left_idx), len(right_idx)
        entropy_left, entropy_right = self._entropy(y[left_idx]), self._entropy(y[right_idx])
        child_entropy = (n_left / n) * entropy_left + (n_right / n) * entropy_right

        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, threshold):
        left_idx = np.argwhere(X_column <= threshold).flatten()
        right_idx = np.argwhere(X_column > threshold).flatten()
        return left_idx, right_idx

    def _entropy(self, y):
        y = y.astype(int)
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _predict(self, inputs):
        node = self.tree
        while isinstance(node, tuple):
            feat_idx, threshold, left, right = node
            if inputs[feat_idx] <= threshold:
                node = left
            else:
                node = right
        return node
    
    def score(self, X, y):
        """
        Calcula la precisión del modelo en los datos dados.
        
        Argumentos:
        X: Matriz de características (muestras, características).
        y: Vector de etiquetas.
        
        Retorna:
        Precisión del modelo.
        """
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy


class RandomForestClassifier:
    def __init__(self, n_trees=10, max_depth=None, n_features=None):
        """
        Inicializa el Bosque Aleatorio.
        
        Argumentos:
        n_trees: Número de árboles en el bosque.
        max_depth: Profundidad máxima de cada árbol.
        n_features: Número de características a utilizar en cada división (si es None, se usa todas).
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        """
        Entrena el bosque aleatorio.
        
        Argumentos:
        X: Matriz de características (muestras, características).
        y: Vector de etiquetas.
        """
        self.trees = []
        n_samples, n_features = X.shape
        if self.n_features is None:
            self.n_features = int(np.sqrt(n_features))  

        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, n_features=self.n_features)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        """
        Realiza predicciones combinando las predicciones de todos los árboles.
        
        Argumentos:
        X: Matriz de características (muestras, características).
        
        Retorna:
        Vector de predicciones.
        """
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)  
        return np.array([self._most_common_label(tree_pred) for tree_pred in tree_preds])

    def predict_proba(self, X):
        """
        Devuelve las probabilidades de clase para cada muestra.
        
        Argumentos:
        X: Matriz de características (muestras, características).
        
        Retorna:
        Matriz de probabilidades de clase (muestras, clases).
        """
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)  

        print(f"Clases predichas por los árboles: {np.unique(tree_preds)}")  

        n_samples = X.shape[0]
        n_classes = len(np.unique(tree_preds))  
        proba = np.zeros((n_samples, n_classes))

        for i, preds in enumerate(tree_preds):
            counter = Counter(preds)
            for label in counter:
                if label < n_classes:  
                    proba[i, int(label)] = counter[label] / len(self.trees)
                else:
                    print(f"Advertencia: Clase {label} fuera de rango")  

        return proba


    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
