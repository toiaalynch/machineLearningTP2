Proyecto de Machine Learning - TP3
Este proyecto incluye la implementación de varios modelos de aprendizaje automático aplicados a dos problemas distintos: detección de cáncer de mama (problema 1) y predicción del riesgo de diabetes (problema 2).

Problemas
Problema 1: Detección de Cáncer de Mama
Este problema utiliza datos sobre características computadas a partir de imágenes de biopsias mamarias para detectar si un tumor es benigno o maligno. Los modelos implementados incluyen:

Regresión Logística con diferentes técnicas de re-balanceo de clases (submuestreo, sobremuestreo, SMOTE, etc.).
Evaluación de los modelos usando métricas como precisión, recall, AUC-ROC, y AUC-PR.

Problema 2: Predicción del Riesgo de Diabetes
Este problema predice el riesgo de desarrollar diabetes utilizando datos del CDC. Los modelos implementados incluyen:

Análisis Discriminante Lineal (LDA)
Regresión Logística Multiclase
Bosques Aleatorios

Estructura del Proyecto
├── problema1
│   ├── data/               
│   ├── src/                
│   
├── problema2
│   ├── data/               
│   ├── src/                
│ 
└── requirements.txt        

Para ambos problemas la carpeta se estructura de la siguiente manera:
src/
├── data_splitting.py   
├── metrics.py          
├── modelo.py           
├── preprocesamiento.py  
├── Entrega_TP3.ipynb              
