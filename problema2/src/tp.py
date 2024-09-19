import numpy as np
from metrics import save_metrics, print_table, plot_combined_roc_curve, plot_combined_pr_curve
from models import LogisticRegressionMulticlass
from models import LinearDiscriminantAnalysis
from models import RandomForestClassifier

def cargar_datos(file_path):
    print(f"Cargando datos desde: {file_path}")
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)  # Cargar datos desde archivo CSV
    y = data[:, 0] 
    X = data[:, 1:]  # Características
    print(f"Datos cargados. Dimensiones de X: {X.shape}, Dimensiones de y: {y.shape}")
    return X, y

def main():
    # Rutas de archivos preprocesados
    file_train = '/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema2/data/processed/diabetes_train_oversampled.csv'
    file_val = '/Users/victoria/Desktop/5tocuatrimestre/ml/tps/tp3ml/machineLearningTP2/problema2/data/processed/diabetes_validation.csv'
    
    # Cargar datos preprocesados
    print("Cargando datos de entrenamiento y validación preprocesados...")
    X_train, y_train = cargar_datos(file_train)
    X_val, y_val = cargar_datos(file_val)
    
    print(f"Clases en y_train: {np.unique(y_train)}, Clases en y_val: {np.unique(y_val)}")

    # Modelo de Regresión Logística Multinomial
    # print("Entrenando modelo de Regresión Logística Multinomial...")
    # lr_model = LogisticRegressionMulticlass(alpha=0.001, lambda_=0.5, num_iters=1000, num_labels=3)
    # lr_model.fit(X_train, y_train)
    # y_pred_lr = lr_model.predict(X_val)
    # y_prob_lr = lr_model.sigmoid(np.dot(lr_model.preprocess_data(X_val), lr_model.all_theta.T))  

    # # Modelo LDA
    # print("Entrenando modelo de LDA...")
    # lda_model = LinearDiscriminantAnalysis(num_components=None)
    # lda_model.fit(X_train, y_train)
    # print("Prediciendo con LDA...")
    # y_pred_lda = lda_model.predict(X_val)
    # print(f"Predicciones (LDA): {y_pred_lda}")
    # y_prob_lda = lda_model.transform(X_val)
    # print(f"Probabilidades (LDA): {y_prob_lda}")

    # # Modelo Bosque Aleatorio
    print("Entrenando modelo de Bosque Aleatorio...")
    rf_model = RandomForestClassifier(n_trees=5, max_depth=5)
    rf_model.fit(X_train, y_train)
    print("Prediciendo con Random Forest...")
    y_pred_rf = rf_model.predict(X_val)
    print(f"Predicciones (RF): {y_pred_rf}")
    y_prob_rf = rf_model.predict_proba(X_val)
    print(f"Probabilidades (RF): {y_prob_rf}")

    # Verificar si hay valores complejos en las probabilidades
    # print("Verificando probabilidades de Regresión Logística:", np.any(np.iscomplex(y_prob_lr)))
    # print("Verificando probabilidades de LDA:", np.any(np.iscomplex(y_prob_lda)))
    print("Verificando probabilidades de Random Forest:", np.any(np.iscomplex(y_prob_rf)))

    # Guardar métricas
    print("Guardando métricas...")
    resultados = {}
    # resultados["Logistic Regression"] = save_metrics(y_val, y_pred_lr, y_prob_lr)
    # resultados["LDA"] = save_metrics(y_val, y_pred_lda, y_prob_lda)
    resultados["Random Forest"] = save_metrics(y_val, y_pred_rf, y_prob_rf)

    # Mostrar tabla de resultados
    print_table(resultados)

    # Graficar las curvas ROC y PR
    # print("Generando curvas ROC y Precision-Recall...")
    # plot_combined_roc_curve(y_val, [y_prob_lr, y_prob_lda, y_prob_rf], "Modelos")
    # plot_combined_pr_curve(y_val, [y_prob_lr, y_prob_lda, y_prob_rf], "Modelos")

if __name__ == "__main__":
    main()
