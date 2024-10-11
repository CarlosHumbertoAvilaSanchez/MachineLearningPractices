---
fileName: p03
title: Práctica 3 Regresión logística
student: Carlos Humberto Avila Sanchez
school: Centro Universitario de Ciencias Exactas e Ingenierías (CUCEI)
subject: Aprendizaje Máquina
teacher: Avila Cardenas Karla
classSection: Sección D01
---

## Introducción
La regresión logística es una técnica de clasificación ampliamente utilizada en problemas donde la variable dependiente es binaria. En este proyecto, se implementa un modelo de regresión logística desde cero utilizando descenso de gradiente, con el objetivo de clasificar muestras del conjunto de datos de vinos proporcionado por `scikit-learn`. Además, se compara esta implementación con métodos estándar de evaluación, como la matriz de confusión, y se calculan métricas de desempeño, incluyendo la precisión, exactitud y recall.

El conjunto de datos contiene tres clases de vinos, pero para simplificar el problema a una clasificación binaria, se filtran las muestras pertenecientes a dos de estas clases. Una vez entrenado el modelo, se evalúa su rendimiento mediante técnicas de validación cruzada y se visualizan los resultados de la clasificación a través de una matriz de confusión.

Este proyecto busca no solo implementar el modelo desde cero, sino también proporcionar una visión clara del rendimiento del modelo, combinando la evaluación numérica con la visualización gráfica.

## Contenido
Este programa tiene como propósito implementar un modelo de regresión logística para clasificar los datos del conjunto de datos de vinos de `sklearn`. El modelo está diseñado para resolver problemas de clasificación binaria, en este caso, clasificando las muestras entre dos clases de vino. Además, se calculan estadísticas como precisión, exactitud y recall, y se visualiza la matriz de confusión.

### Librerías Utilizadas
```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
```
Estas librerías permiten:
1. Leer y manipular los datos.
2. Implementar la regresión logística.
3. Calcular métricas de clasificación y visualizar los resultados mediante gráficas.

### Funciones y Clases
La clase `LogisticRegression` implementa un modelo de regresión logística desde cero, utilizando descenso de gradiente para ajustar los parámetros del modelo.

El constructor inicializa el modelo de regresión logística con una tasa de aprendizaje predeterminada y un número de iteraciones para el descenso de gradiente. También inicializa los pesos y el sesgo (bias) como `None`.

```python
def __init__(self, learning_rate=0.01, n_iters=1000):
    self.learning_rate = learning_rate
    self.n_iters = n_iters
    self.weights = None
    self.bias = None
    self.losses = []
```

Este método calcula la `sigmoide`, que es la función de activación utilizada en la regresión logística para convertir la salida del modelo en probabilidades.
```python
def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))
```

La función de pérdida (loss) utilizada aquí es la *log-loss*, que mide la diferencia entre las predicciones del modelo y los valores reales. La pérdida se minimiza durante el entrenamiento.
```python
def loss(self, h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
```

El método `fit()` entrena el modelo ajustando los pesos y el sesgo utilizando descenso de gradiente. En cada iteración:

1. Calcula el modelo lineal.
2. Aplica la función sigmoide para obtener las predicciones.
3. Calcula la pérdida y ajusta los pesos y el sesgo en función de los gradientes.

```python
def fit(self, X, y):
    num_samples, num_features = X.shape
    self.weights = np.zeros(num_features)
    self.bias = 0

    for _ in range(self.n_iters):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)

        loss = self.loss(y_predicted, y)
        self.losses.append(loss)

        dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
        db = (1 / num_samples) * np.sum(y_predicted - y)

        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
```

El método `predict()` genera las predicciones para nuevos datos. Usa la función sigmoide para obtener probabilidades y luego asigna clases **(1 o 0)** según un umbral de **0.5**.
```python
def predict(self, X):
    linear_model = np.dot(X, self.weights) + self.bias
    y_predicted = self.sigmoid(linear_model)
    y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
    return y_predicted_cls
```

la función `cross_validation()` realiza una validación cruzada simple, dividiendo los datos en un conjunto de entrenamiento y un conjunto de prueba, repitiendo el proceso 10 veces. Devuelve el promedio de las exactitudes (accuracy) obtenidas en cada iteración.
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

La función `getStatistics` calcula las métricas de exactitud (accuracy), precisión (precision) y recall a partir de los valores de la matriz de confusión.
```python
def getStatistics(TN, FP, FN, TP):
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return accuracy, precision, recall
```

`printStatistics()` imprime las estadísticas calculadas, incluyendo la matriz de confusión, precisión, exactitud y recall, además del resultado de la validación cruzada.
```python
def printStatistics(cm, precision, accuracy, recall, validations):
    print("Matriz de confusión:", cm)
    print("Precisión:", precision)
    print("Exactitud:", accuracy)
    print("Recall:", recall)
    print("Validaciones cruzadas:", validations)
```

La función `plotConfusionMatrix()` genera una visualización de la matriz de confusión en un gráfico 2D de colores. Muestra cómo se clasificaron las instancias verdaderas frente a las predichas.
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

### Bloque Principal
En el bloque principal:

1. Se cargan los datos del conjunto de datos de vinos y se filtran para que solo incluyan las dos primeras clases.
2. Los datos se dividen en conjuntos de entrenamiento y prueba.
3. Se entrena el modelo de regresión logística utilizando el conjunto de entrenamiento.
4. Se calculan las predicciones y la matriz de confusión.
5. Se calculan y muestran las estadísticas: precisión, exactitud y recall, junto con la validación cruzada.
6. Finalmente, se visualiza la matriz de confusión.

```python
if __name__ == "__main__":

    wine = datasets.load_wine()
    X = wine.data
    y = wine.target

    binary_indices = y != 2
    X = X[binary_indices]
    y = y[binary_indices]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression(learning_rate=0.0001, n_iters=1000)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()

    accuracy, precision, recall = getStatistics(TN, FP, FN, TP)

    validations = cross_validation(X, y, model)

    printStatistics(cm, precision, accuracy, recall, validations)
    plotConfusionMatrix(cm)
```

### Resultados
<div class="charts">
    <div>
        <img src="../images/evidence/p03_img_01.jpg" alt="Matriz de confusión">
        <span>Matriz de confusión</span>
    </div>
    <div class="screenshot">
        <img src="../images/evidence/p03_img_02.jpg" alt="Resultados">
        <span>Resultados</span>
    </div>
</div>

## Conclusión
En este proyecto se ha implementado un modelo de regresión logística desde cero, utilizando descenso de gradiente, para clasificar datos del conjunto de vinos en dos clases. A través del entrenamiento y evaluación del modelo, se ha demostrado la capacidad de la regresión logística para abordar problemas de clasificación binaria, logrando una buena precisión y exactitud en las predicciones.

La validación cruzada permitió evaluar la capacidad de generalización del modelo, mostrando un rendimiento estable en distintas particiones de los datos. Además, la matriz de confusión y las métricas derivadas, como la precisión y el recall, proporcionaron una evaluación clara del desempeño del modelo, identificando su capacidad tanto para clasificar correctamente como para minimizar errores de clasificación.

La visualización de la matriz de confusión ofreció una representación gráfica de los resultados, facilitando la interpretación del comportamiento del modelo. En conjunto, este proyecto destaca la utilidad de la regresión logística y las técnicas de validación en problemas de clasificación binaria.