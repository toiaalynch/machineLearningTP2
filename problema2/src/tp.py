import numpy as np
from metricas import guardar_metricas, imprimir_tabla_resultados, plot_roc_curves, plot_pr_curves
from data_splitting import split_train_validation
from modelos import LogisticRegressionMulticlass
from modelos import LinearDiscriminantAnalysis
from modelos import RandomForestClassifier
import csv



def cargar_datos(file_path):
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)  
    X = data[:, :-1]  
    y = data[:, -1]   
    return X, y



def main():
    # Cargar datos
    file_path = '/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema2/data/diabetes_dev.csv'
    X, y = cargar_datos(file_path)


    # Dividir datos en entrenamiento y validación
    X_train, X_val, y_train, y_val = split_train_validation(X, y, validation_size=0.2, seed=42)

    # 1. Entrenar el modelo de Regresión Logística Multinomial
    print("Entrenando modelo de Regresión Logística Multinomial...")
    lr_model = LogisticRegressionMulticlass(alpha=0.1, lambda_=1, num_iters=1000, num_labels=3)
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_val)
    y_prob_lr = lr_model.sigmoid(np.dot(lr_model.preprocess_data(X_val), lr_model.all_theta.T))

    # 2. Entrenar el modelo de Análisis Discriminante Lineal (LDA)
    print("Entrenando modelo de LDA...")
    lda_model = LinearDiscriminantAnalysis(num_components=2)
    lda_model.fit(X_train, y_train)
    y_pred_lda = lda_model.predict(X_val)
    y_prob_lda = lda_model.transform(X_val)  

    # 3. Entrenar el modelo de Bosque Aleatorio (Random Forest)
    print("Entrenando modelo de Bosque Aleatorio...")
    rf_model = RandomForestClassifier(n_trees=5, max_depth=5)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_val)
    y_prob_rf = y_pred_rf  # No hay probabilidades en esta implementación manual, así que usamos las predicciones

    # Calcular métricas
    resultados = {}
    resultados["Logistic Regression"] = guardar_metricas(y_val, y_pred_lr, y_prob_lr)
    resultados["LDA"] = guardar_metricas(y_val, y_pred_lda, y_prob_lda)
    resultados["Random Forest"] = guardar_metricas(y_val, y_pred_rf, y_prob_rf)

    # Imprimir tabla de resultados
    imprimir_tabla_resultados(resultados)

    # Graficar las curvas ROC
    print("Generando curvas ROC...")
    plot_roc_curves(["Logistic Regression", "LDA", "Random Forest"], y_val, [y_prob_lr[:, 1], y_prob_lda[:, 1], y_prob_rf])

    # Graficar las curvas Precision-Recall
    print("Generando curvas Precision-Recall...")
    plot_pr_curves(["Logistic Regression", "LDA", "Random Forest"], y_val, [y_prob_lr[:, 1], y_prob_lda[:, 1], y_prob_rf])

if __name__ == "__main__":
    main()
