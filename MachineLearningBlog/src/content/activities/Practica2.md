---
fileName: p02
title: Práctica 2 Regresión Múltiple
student: Carlos Humberto Avila Sanchez
school: Centro Universitario de Ciencias Exactas e Ingenierías (CUCEI)
subject: Aprendizaje Máquina
teacher: Avila Cardenas Karla
classSection: Sección D01
---
## Introdución
El análisis de regresión lineal múltiple es una técnica estadística ampliamente utilizada para modelar la relación entre una variable dependiente y varias variables independientes. En este contexto, el objetivo es predecir el rendimiento estudiantil en función de las horas de estudio y los puntajes previos. Esta práctica explora la implementación de un modelo de regresión lineal múltiple, tanto mediante una implementación personalizada utilizando el algoritmo de descenso de gradiente, como utilizando el modelo de regresión lineal de la biblioteca `scikit-learn`. 

Además, se incluye la visualización gráfica en 3D de los datos y de las predicciones del modelo, así como la aplicación de técnicas de validación cruzada (k-fold y simple) para evaluar el desempeño y la precisión del modelo. Esta combinación de herramientas ofrece una visión detallada del comportamiento del modelo y su capacidad para generalizar las predicciones a nuevos datos.

## Contenido
Este programa tiene como propósito implementar un modelo de regresión lineal múltiple utilizando tanto una implementación personalizada como el modelo de regresión lineal de `scikit-learn`. El objetivo es predecir el índice de rendimiento de los estudiantes basado en las horas de estudio y sus puntajes previos. El programa también incluye la visualización de las predicciones y la validación cruzada para evaluar el rendimiento del modelo.

### Librerías Utilizadas

El programa importa las siguientes librerías:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
```

Estas librerías permiten leer y manipular los datos, realizar el modelado de regresión y visualizar los resultados en 3D.

### Funciones y clases

La función *get_data()* se encarga de leer el archivo CSV Student_Performance.csv, que contiene la información sobre el rendimiento estudiantil. Luego, selecciona una muestra aleatoria del 1% de los datos para reducir la cantidad de datos utilizados y garantizar que el procesamiento sea eficiente. La función retorna un DataFrame de Pandas con la muestra.
```python
def get_data():
    data = pd.read_csv("Student_Performance.csv", encoding="latin-1")
    data = data.sample(frac=0.01, random_state=1)
    return data
```
La función *cross_validation()* implementa una validación cruzada básica. Divide aleatoriamente los datos en un conjunto de entrenamiento (70%) y un conjunto de prueba (30%) en varias iteraciones. Después de cada entrenamiento, se calcula el error cuadrático medio (MSE) de las predicciones y se retorna el promedio de los errores obtenidos en las 10 iteraciones.

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
        mse = np.mean((Y_test - predictions) ** 2)
        scores.append(mse)

    return np.mean(scores)
```
La función *k_fold_cross_validation()* realiza la validación cruzada utilizando el método K-Fold, donde los datos se dividen en k subconjuntos (o pliegues). En cada iteración, uno de los pliegues se utiliza para prueba y los restantes para entrenamiento. Se calcula el MSE para cada pliegue y se devuelve el promedio del error.

```python
def k_fold_cross_validation(X, Y, model, k=10):
    x_folds = np.array_split(X, k)
    y_folds = np.array_split(Y, k)
    scores = []

    for i in range(k):
        x_test = x_folds[i]
        y_test = y_folds[i]
        x_train = np.concatenate(x_folds[:i] + x_folds[i + 1:])
        y_train = np.concatenate(y_folds[:i] + y_folds[i + 1:])
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        mse = np.mean((y_test - predictions) ** 2)
        scores.append(mse)

    return np.mean(scores)
```
Esta clase implementa un modelo de regresión lineal múltiple desde cero utilizando descenso de gradiente. Se inicializa con una tasa de aprendizaje muy pequeña y un gran número de iteraciones para asegurar la convergencia.

```python
class MultipleLinearRegression:
    def __init__(self, learning_rate=0.0000001, n_iter=1000000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.coef_ = None
        self.intercept_ = None
```

El método *fit()* ajusta los coeficientes utilizando descenso de gradiente. Inicializa los coeficientes de manera aleatoria y los ajusta en cada iteración hasta minimizar el error cuadrático.

```python
def fit(self, X, y):
        X_b = np.column_stack((np.ones(len(X)), X))
        self.coef_ = np.random.rand(X_b.shape[1])

        for _ in range(self.n_iter):
            gradient = -2 * X_b.T @ (y - X_b @ self.coef_)
            self.coef_ -= self.learning_rate * gradient

        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]
```

El método *predict()* realiza predicciones basadas en los datos de entrada X utilizando los coeficientes ajustados durante el entrenamiento.

```python
def predict(self, X):
        X_b = np.column_stack((np.ones(len(X)), X))
        return X_b @ np.concatenate(([self.intercept_], self.coef_))
```
### Bloque principal
El bloque principal del código:

1. Carga los datos de rendimiento estudiantil y extrae las variables independientes Hours Studied y Previous Scores, junto con la variable dependiente Performance Index.
2. Entrena el modelo de regresión lineal múltiple personalizado y lo visualiza utilizando la función *plot_data()*. También imprime las estadísticas del modelo.
3. Entrena el modelo de LinearRegression de `scikit-learn` de manera similar.
4. Realiza validaciones cruzadas tanto con el modelo personalizado como con el modelo de `scikit-learn`, calculando el error cuadrático medio de las predicciones en cada caso.

```python
if __name__ == "__main__":
    data = get_data()
    X = np.array([
        data["Hours Studied"].values,
        data["Previous Scores"].values,
    ]).T

    Y = np.array(data["Performance Index"])

    modelG = MultipleLinearRegression()
    modelG.fit(X, Y)
    plot_data(modelG, X, Y)
    print_statistics(modelG)

    model = LinearRegression()
    model.fit(X, Y)
    plot_data(model, X, Y)
    print_statistics(model)

    print(
        "K-Fold validation: ", 
        k_fold_cross_validation(X, Y, modelG)
    )
    print(
        "Cross validation: ", 
        cross_validation(X, Y, model)
    )
```

### Resultados
<div class="charts">
    <div>
        <img src="../images/evidence/p02_img_01.jpg" alt="Gráfica con modelo personalizado">
        <span>Modelo Personalizado</span>
    </div>
    <div>
        <img src="../images/evidence/p02_img_02.jpg" alt="Gráfica con modelo sklearn">
        <span>Modelo sklearn</span>
    </div>
</div>

<div class="screenshot">
    <div>
        <img src="../images/evidence/p02_img_03.jpg" alt="Captura de pantalla de los resultados">
        <span>Captura de pantalla de los resultados</span>
    </div>
</div>





## Conclusión

Este proyecto demuestra la efectividad de la regresión lineal múltiple para modelar y predecir el rendimiento estudiantil en función de variables como las horas de estudio y los puntajes previos. A través de la implementación de un modelo personalizado utilizando descenso de gradiente, junto con el uso del modelo estándar de `scikit-learn`, se logró obtener una perspectiva comparativa sobre la precisión y eficiencia de ambos enfoques.

La visualización en 3D de los datos y las predicciones permitió una interpretación más clara de los resultados, mostrando cómo los modelos ajustan una superficie de predicción sobre los puntos de datos reales. Además, las técnicas de validación cruzada implementadas ayudaron a medir la capacidad de generalización de los modelos, proporcionando una evaluación del error en distintas particiones de los datos.

En general, la regresión lineal múltiple se confirma como una herramienta valiosa para analizar relaciones entre múltiples variables, y este proyecto destaca la importancia de la validación y la visualización en el proceso de construcción de modelos predictivos.