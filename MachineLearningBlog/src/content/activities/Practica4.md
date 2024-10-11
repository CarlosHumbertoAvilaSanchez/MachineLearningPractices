---
fileName: p05
title: Práctica 5 kNN
student: Carlos Humberto Avila Sanchez
school: Centro Universitario de Ciencias Exactas e Ingenierías (CUCEI)
subject: Aprendizaje Máquina
teacher: Avila Cardenas Karla
classSection: Sección D01
---

## Introducción
El algoritmo K-Nearest Neighbors (KNN) es una técnica de clasificación supervisada que clasifica una muestra basándose en las clases de sus vecinos más cercanos. Es uno de los algoritmos más sencillos en el ámbito del aprendizaje automático y se utiliza comúnmente para resolver problemas de clasificación y regresión. En este proyecto, se implementa desde cero el algoritmo KNN para clasificar especies de pingüinos utilizando el conjunto de datos de `seaborn`, que incluye características como la longitud y profundidad del pico, la longitud de las aletas y la masa corporal de los pingüinos.

El modelo KNN no requiere un proceso de entrenamiento explícito, sino que toma decisiones basadas en la distancia entre los puntos de prueba y los de entrenamiento. Para ello, se utiliza la distancia euclidiana para medir la similitud entre puntos. Además, se implementa la validación cruzada para evaluar el rendimiento del modelo y se busca el valor óptimo de `k`, que es el número de vecinos a considerar. Por último, se visualizan los resultados mediante gráficos, incluyendo una matriz de confusión que permite analizar el desempeño del modelo en términos de predicciones correctas e incorrectas.

## Contenido
Este programa implementa el algoritmo de clasificación **K-Nearest Neighbors (KNN)** para predecir la especie de pingüinos utilizando características como la longitud del pico, la profundidad del pico y la longitud de las aletas. El algoritmo se entrena y evalúa utilizando un conjunto de datos proporcionado por `seaborn`. Además, se realiza validación cruzada, selección del valor óptimo de `k` y se visualizan los resultados a través de gráficos y la matriz de confusión.

### Librerías Utilizadas
```python
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import plotly_express as px
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
```
Estas librerías permiten la manipulación de datos, la visualización de gráficos en 2D y 3D, y el uso de métricas de evaluación, como la matriz de confusión, para medir el rendimiento del modelo KNN.

### Funciones y modelo
1. **euclidean_distance(x1, x2)**
```python
def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance
```
Esta función calcula la distancia euclidiana entre dos puntos, x1 y x2, lo que es fundamental en el algoritmo KNN para determinar los vecinos más cercanos.

2. **Clase KnnModel**
La clase KnnModel implementa el algoritmo KNN desde cero.
```python
def __init__(self, k=3):
    self.k = k
```
El constructor inicializa el modelo KNN con un valor de *k*, que define el número de vecinos más cercanos a considerar para la clasificación.

```python
def fit(self, X, y):
    self.X_train = np.array(X)
    self.y_train = np.array(y)
```
El método *fit()* simplemente almacena los datos de entrenamiento, ya que KNN es un algoritmo basado en ejemplos (no requiere un entrenamiento tradicional).

```python
def predict(self, X):
    predictions = [self._predict(x) for x in np.array(X)]
    return predictions
```
El método *predict()* realiza predicciones sobre los datos de prueba, llamando internamente a la función *_predict()*.

```python
def _predict(self, x):
    distances = [
        euclidean_distance(x, x_train) for x_train in self.X_train
    ]
    k_indices = np.argsort(distances)[: self.k]
    k_nearest_labels = [self.y_train[i] for i in k_indices]
    most_common = Counter(k_nearest_labels).most_common()
    return most_common[0][0]
```
El método *_predict()* calcula la distancia entre el punto de prueba *x* y todos los puntos de entrenamiento, selecciona los *k* vecinos más cercanos, y devuelve la clase más común entre ellos.

3. **accuracy_score(y_true, y_pred)**
```python
def accuracy_score(y_true, y_pred):
    correct_predictions = sum(y_true == y_pred)
    accuracy = correct_predictions / len(y_true)
    return accuracy
```
Esta función calcula la precisión del modelo dividiendo el número de predicciones correctas entre el total de predicciones.

4. **cross_validation(X, Y, model, training_size=0.7)**
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
        score = np.mean(predictions == Y_test)
        scores.append(score)

    return np.mean(scores)
```
La función *cross_validation()* realiza una validación cruzada en 10 iteraciones. En cada iteración, divide aleatoriamente los datos en entrenamiento y prueba, ajusta el modelo y calcula la precisión en el conjunto de prueba.

5. **best_k(X_train, Y_train, X_test, Y_test)**
```python
def best_k(X_train, Y_train, X_test, Y_test):
    for n in range(1, 16):
        KNN = KnnModel(k=n)
        KNN.fit(X_train, Y_train)
        y_train_predict = KNN.predict(X_train)
        y_test_predict = KNN.predict(X_test)

        train_acc = metrics.accuracy_score(Y_train, y_train_predict)
        test_acc = metrics.accuracy_score(Y_test, y_test_predict)

        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)

    plt.plot(range(1, 16), train_accuracy, label="Training Accuracy")
    plt.plot(range(1, 16), test_accuracy, label="Test Accuracy")
    plt.xlabel("n_neighbors")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. n_neighbors")
    plt.legend()
    plt.show()

    best_k = np.argmax(test_accuracy) + 1
    return best_k
```
Esta función busca el mejor valor de *k* entre 1 y 15 para el modelo KNN, comparando la precisión en los conjuntos de entrenamiento y prueba. Luego visualiza el rendimiento para diferentes valores de *k* y retorna el valor que optimiza la precisión.

6. **print_statistics(Y_test, Y_predict, validations)**
```python
def print_statistics(Y_test, Y_predict, validations):
    print(metrics.classification_report(Y_test, Y_predict))
    print(f"Validación cruzada: {validations}")
```
Imprime un reporte detallado de clasificación que incluye métricas como precisión, recall y f1-score, además del resultado de la validación cruzada.

7. **cm(Y_test, Y_predict)**
```python
def cm(Y_test, Y_predict):
    KNN_cm = confusion_matrix(Y_test, Y_predict)
    f, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(
        KNN_cm, 
        annot=True, 
        linewidth=0.7, 
        linecolor="cyan", 
        fmt="g", ax=ax, 
        cmap="BuPu"
    )
    plt.title("Matríz de confusión de modelo KNN")
    plt.xlabel("Y predict")
    plt.ylabel("Y test")
    plt.show()
```
Genera y visualiza una matriz de confusión que muestra el desempeño del modelo KNN en términos de predicciones correctas e incorrectas para cada clase.

### Bloque Principal
```python
if __name__ == "__main__":
    train_accuracy = []
    test_accuracy = []

    penguins = sns.load_dataset("penguins").dropna()
    penguins = penguins[COL_NAMES]

    X = penguins.drop("species", axis=1)
    Y = penguins["species"]

    fig = px.scatter_3d(
        penguins,
        x="bill_length_mm",
        y="bill_depth_mm",
        z="flipper_length_mm",
        color="species",
    )

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=9
    )

    best_k = best_k(X_train, Y_train, X_test, Y_test)

    KNN = KnnModel(k=best_k)
    KNN.fit(X_train, Y_train)

    Y_predict = KNN.predict(X_test)

    validations = cross_validation(np.array(X), np.array(Y), KNN)
    print_statistics(Y_test, Y_predict, validations)

    cm(Y_test, Y_predict)
```
En este bloque principal:

1. Se cargan y preparan los datos de pingüinos.
2. Se dividen los datos en conjuntos de entrenamiento y prueba.
3. Se busca el mejor valor de k para el modelo KNN.
4. Se entrena el modelo KNN con el valor óptimo de k y se realizan predicciones.
5. Se calculan las estadísticas de rendimiento del modelo y se visualiza la matriz de confusión.

### Resultados
<div class="charts">
    <div>
        <img src="../images/evidence/p04_img_01.jpg" alt="Gráfica de precisión vs n_neighbors">
        <span>Gráfica de precisión vs n_neighbors</span>
    </div>
    <div>
        <img src="../images/evidence/p04_img_02.jpg" alt="Matriz de confusión">
        <span>Matriz de confusión</span>
    </div>
    <div>
        <img src="../images/evidence/p04_img_04.jpg" alt="Gráfica de clasificación">
        <span>Gráfica de clasificación</span>
    </div>
</div>
<div class="screenshot">
        <img src="../images/evidence/p04_img_03.jpg" alt="Resultados">
        <span>Resultados</span>
</div>

## Conclusión

En este proyecto se ha implementado el algoritmo K-Nearest Neighbors (KNN) desde cero, aplicándolo a la clasificación de especies de pingüinos en función de varias características físicas. El uso del conjunto de datos de pingüinos permitió poner en práctica la capacidad del modelo KNN para resolver problemas de clasificación supervisada de manera eficiente.

Se observó que el valor del parámetro `k`, que representa el número de vecinos considerados, tiene un impacto directo en el rendimiento del modelo. Mediante la evaluación de diferentes valores de `k`, se seleccionó el valor óptimo para maximizar la precisión en el conjunto de prueba. Además, se utilizó la validación cruzada para medir la capacidad de generalización del modelo, obteniendo resultados consistentes en diferentes particiones de los datos.

El análisis de la matriz de confusión permitió identificar con mayor claridad los errores de clasificación y las clases que el modelo pudo predecir con mayor precisión. En general, KNN demostró ser un método eficaz y sencillo para resolver problemas de clasificación, aunque su rendimiento puede depender de la correcta elección de `k` y de la distribución de los datos. Este proyecto resalta la importancia de ajustar adecuadamente los parámetros del modelo y de utilizar técnicas de validación para garantizar un buen desempeño.