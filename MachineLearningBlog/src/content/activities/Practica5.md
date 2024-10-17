---
fileName: p06
title: Práctica 6 SVM
student: Carlos Humberto Avila Sanchez
school: Centro Universitario de Ciencias Exactas e Ingenierías (CUCEI)
subject: Aprendizaje Máquina
teacher: Avila Cardenas Karla
classSection: Sección D01
---

## Introducción
Las Máquinas de Soporte Vectorial (SVM) son una poderosa técnica de clasificación supervisada que busca encontrar un hiperplano óptimo que separe las clases de un conjunto de datos de manera que se maximice el margen entre los puntos más cercanos de cada clase. Este enfoque es particularmente efectivo en problemas de clasificación binaria. En este proyecto, se implementa un modelo SVM utilizando el conjunto de datos de iris, enfocado en la clasificación de dos especies de flores: *Setosa* y *Versicolor*.

El objetivo principal es entrenar un modelo SVM y optimizar el valor del parámetro `C`, que controla la regularización del modelo, mediante una búsqueda en rejilla (*GridSearchCV*). A continuación, se evalúa el rendimiento del modelo utilizando validación cruzada y una matriz de confusión para medir métricas clave como la precisión, exactitud y recall. Además, se visualizan las fronteras de decisión del modelo para observar cómo clasifica las muestras y dónde comete errores. Este proyecto destaca el uso de técnicas avanzadas para optimizar y evaluar modelos de clasificación basados en SVM.

## Contenido
Este programa implementa una **Máquina de Soporte Vectorial (SVM)** utilizando el conjunto de datos de iris para clasificar entre dos especies de flores: *Setosa* y *Versicolor*. El objetivo es encontrar el mejor valor del parámetro `C` utilizando una búsqueda en rejilla (**GridSearchCV**), evaluar el modelo mediante validación cruzada, y analizar su rendimiento utilizando una matriz de confusión y métricas como la precisión, exactitud y recall.

### Librerías Utilizadas
```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
```
Estas librerías permiten:

1. Cargar y manipular los datos de iris.
2. Implementar el modelo de Máquina de Soporte Vectorial (SVM).
3. Evaluar el rendimiento del modelo a través de una matriz de confusión y validación cruzada.
4. Visualizar los resultados gráficamente.

### Funciones
*plotConfusionMatrix(cm)*
```python
def plotConfusionMatrix(cm):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Matriz de confusión")
    plt.colorbar()
    plt.xticks([0, 1], ["Clase 0", "Clase 1"])
    plt.yticks([0, 1], ["Clase 0", "Clase 1"])
    plt.xlabel("Etiqueta Predicha")
    plt.ylabel("Etiqueta Verdadera")
    plt.show()
```
Esta función genera y visualiza una matriz de confusión que muestra cómo se han clasificado las instancias reales versus las predicciones del modelo. La matriz es una herramienta útil para analizar el rendimiento del modelo en términos de verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos.

*getStatistics(TN, FP, FN, TP)*
```python
def getStatistics(TN, FP, FN, TP):
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return accuracy, precision, recall
```
Esta función calcula las principales métricas de evaluación: exactitud (accuracy), precisión (precision) y recall. Estas métricas permiten evaluar qué tan bien el modelo clasifica correctamente las muestras.

*cm_statistics(cm)*
```python
def cm_statistics(cm):
    TN, FP, FN, TP = cm.ravel()
    accuracy, precision, recall = getStatistics(TN, FP, FN, TP)
    print(f"Accuracy: {accuracy}, 
            Precision: {precision}, 
            Recall: {recall}"
    )
```
Esta función extrae los valores de la matriz de confusión y calcula las métricas de rendimiento del modelo, imprimiendo la exactitud, precisión y recall.

*cross_validation(X, Y, model, training_size=0.7)*
```python
def cross_validation(X, Y, model, training_size=0.7):
    n = len(X)
    training_size = int(n * training_size)
    scores = []

    for _ in range(10):
        idx = np.random.permutation(n)
        X_train, X_test = X[idx][:training_size], 
                          X[idx][training_size:]
        Y_train, Y_test = Y[idx][:training_size], 
                          Y[idx][training_size:]

        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        cm = confusion_matrix(Y_test, predictions)
        cm_statistics(cm)
        score = np.mean(predictions == Y_test)
        scores.append(score)

    return np.mean(scores)
```
Esta función implementa una validación cruzada repitiendo el entrenamiento y la evaluación del modelo en 10 iteraciones. En cada iteración, divide aleatoriamente los datos en conjuntos de entrenamiento y prueba, calcula la matriz de confusión, y devuelve la media de las exactitudes obtenidas.

*best_C()*
```python
def best_C():
    param_grid = {"C": np.logspace(-5, 7, 20)}
    grid = GridSearchCV(
        estimator=SVC(kernel="rbf", gamma="scale"),
        param_grid=param_grid,
        scoring="accuracy",
        n_jobs=-1,
        cv=3,
        verbose=0,
        return_train_score=True,
    )
    grid.fit(X=X_train, y=y_train)
    return grid.best_params_["C"], grid.best_estimator_
```
La función **best_C()** utiliza GridSearchCV para encontrar el mejor valor del parámetro **C**, que controla la regularización del modelo SVM. Se evalúan diferentes valores de **C** en un rango logarítmico, y el valor que maximiza la exactitud es seleccionado.

*plotSVM(X, y, X_test, y_test, y_pred, model, features_names)*
```python
def plotSVM(X, y, X_test, y_test, y_pred, model, features_names):
    plt.figure(figsize=(10, 6))
    plt.scatter(
        X[y == 0][:, 0], X[y == 0][:, 1], 
        color="r", label="Setosa"
    )
    plt.scatter(
        X[y == 1][:, 0], X[y == 1][:, 1], 
        color="g", label="Versicolor"
    )
    plt.scatter(
        X_test[y_test != y_pred][:, 0],
        X_test[y_test != y_pred][:, 1],
        color="b",
        label="Misclassified",
    )

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)
    ax.contour(
        XX,
        YY,
        Z,
        colors="k",
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=["--", "-", "--"],
    )

    plt.xlabel(features_names[0])
    plt.ylabel(features_names[1])
    plt.legend()
    plt.show()
```
Esta función visualiza las fronteras de decisión del modelo SVM, mostrando cómo separa las dos clases (Setosa y Versicolor). También resalta las muestras mal clasificadas para ayudar a identificar los errores del modelo.

### Bloque Principal
```python
if __name__ == "__main__":

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X = X[:, :2]

    X = X[y != 2]
    y = y[y != 2]

    features_names = iris.feature_names
    target_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    best_c, model = best_C()
    print(f"El mejor valor de C es: {best_c}")
    cross_validation(X, y, model)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plotConfusionMatrix(cm)
    cm_statistics(cm)

    plotSVM(X, y, X_test, y_test, y_pred, model, features_names)
```
En este bloque principal:
1. Se cargan los datos de iris y se filtran para trabajar solo con dos clases (Setosa y Versicolor).
2. Se buscan los mejores hiperparámetros para el modelo SVM mediante GridSearchCV.
3. Se entrena y evalúa el modelo mediante validación cruzada y una matriz de confusión.
4. Se visualizan las fronteras de decisión del modelo y las muestras mal clasificadas.

### Resultados
<div class="charts">
    <div>
        <img src="../images/evidence/p05_img_01.jpg" alt="Matriz de confusión">
        <span>Matriz de confusión</span>
    </div>
    <div>
        <img src="../images/evidence/p05_img_02.jpg" alt="Plot de SVM">
        <span>Plot de SVM</span>
    </div>
</div>

<div class="screenshot">
    <div>
        <img src="../images/evidence/p05_img_03.jpg" alt="Captura de pantalla de los resultados">
        <span>Captura de pantalla de los resultados</span>
    </div>
</div>

## Conclusión 

En este proyecto se ha implementado y evaluado una Máquina de Soporte Vectorial (SVM) para la clasificación de dos especies de flores del conjunto de datos de iris: *Setosa* y *Versicolor*. A través de la optimización del parámetro `C` mediante una búsqueda en rejilla (*GridSearchCV*), se logró seleccionar el mejor valor para mejorar el rendimiento del modelo, lo que resalta la importancia de ajustar adecuadamente los hiperparámetros en algoritmos de clasificación.

El uso de la validación cruzada permitió obtener una evaluación precisa del modelo, mostrando una exactitud consistente y destacando la capacidad del SVM para generalizar correctamente en datos no vistos. La matriz de confusión proporcionó información detallada sobre los errores cometidos, y la visualización de las fronteras de decisión ayudó a comprender cómo el modelo separa las dos clases de flores y qué muestras fueron mal clasificadas.

En resumen, el modelo SVM demostró ser una técnica eficaz para la clasificación binaria, ofreciendo resultados sólidos cuando se optimizan sus parámetros. La combinación de evaluación numérica y visualización gráfica permitió un análisis completo del rendimiento del modelo, destacando su capacidad para manejar problemas de clasificación complejos.