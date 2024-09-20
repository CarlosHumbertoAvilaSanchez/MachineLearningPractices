---
fileName: p01
title: Práctica 1 Regresión lineal simple
student: Carlos Humberto Avila Sanchez
school: Centro Universitario de Ciencias Exactas e Ingenierías (CUCEI)
subject: Aprendizaje Máquina
teacher: Avila Cardenas Karla
classSection: Sección D01
---

## Introducción
El presente reporte tiene como objetivo realizar un análisis de regresión lineal simple utilizando datos de compra de autos. El enfoque se centra en investigar la relación entre el salario anual de los compradores y el monto gastado en la compra de autos, utilizando un conjunto de datos que contiene ambas variables.

La regresión lineal es una técnica estadística que permite modelar la relación entre una variable dependiente y una o más variables independientes. En este ejercicio, emplearemos la variable salario anual como variable independiente para predecir la variable dependiente, que es el monto de compra de autos. A través de este modelo, se buscará establecer una fórmula que permita realizar predicciones basadas en el salario de un individuo.

Adicionalmente, el análisis incluye la comparación de predicciones utilizando la totalidad de los datos disponibles y una muestra aleatoria del 60% de los mismos, con el fin de evaluar la consistencia del modelo en diferentes tamaños de conjuntos de datos.

Este reporte proporcionará una visión clara de cómo la regresión lineal puede ser utilizada para modelar relaciones económicas y hacer predicciones, además de presentar métricas de error que permiten evaluar la precisión de dicho modelo.


## Contenido
A continuación se presentará el código implementado para predecir el monto de compra de autos en función del salario anual. Se utilizará un conjunto de datos de compras de autos y el análisis se realiza tanto con la totalidad de los datos como con una muestra del 60% de ellos.

### Librerías necesarias
Primero importamos las librerías necesarias para nuestro análisis:

```python
    import matplotlib.pyplot as plt 
    import pandas as pd
```
Estas librerías nos ayudarán a mostrar las gráficas para visualizar las predicciones de la regresión lineal y leer el archivo CSV que contiene los datos de las compras de autos.

### Bloque principal.
```python
    if __name__ == "__main__":
    # PREDICCIÓN CON EL TODOS LOS DATOS
    print("Predicción con todos los datos")
    car_purchasing_data = read_car_purchasing_data()
    linear_regression(car_purchasing_data)

    # PREDICCIÓN CON UNA MUESTRA DEL 60% DE LOS DATOS
    print("\nPredicción con una muestra del 60% de los datos")
    sample_60 = car_purchasing_data.sample(
        frac=0.6, random_state=1
    )
    linear_regression(sample_60)
```
En este bloque se ejecuta el programa principal. Primero, se realiza una predicción utilizando todos los datos de compra de autos. Posteriormente, se toma una muestra del 60% de los datos para realizar una segunda predicción con un subconjunto de los datos originales, permitiendo comparar la precisión de la regresión en ambos casos.

### Funciones.
1. **read_car_purchasing_data()**
```python
   def read_car_purchasing_data():
    data = pd.read_csv(
        "car_purchasing.csv", 
        usecols=[5, 8], 
        encoding="latin-1"
    )
    return data 
```
La función **read_car_purchasing_data** tiene como propósito leer los datos del archivo **car_purchasing.csv**, utilizando solo las columnas relevantes para nuestro análisis: la columna 5 que contiene el salario anual y la columna 8 que contiene el monto de compra de autos. La función retorna un DataFrame de Pandas con los datos filtrados.

2. **linear_regression(data)**
```python
def linear_regression(data):
    X = data["annual Salary"].values
    Y = data["car purchase amount"].values

    mean_x = get_mean(X)
    mean_y = get_mean(Y)

    slope = calculate_slope(mean_x, mean_y, X, Y)

    b = mean_y - slope * mean_x

    show_predictions(X, Y, slope, b)

    distances = [
        abs(yi - slope * xi - b) 
        for xi, yi in zip(X, Y)
    ]
    mse = sum(
        (yi - slope * xi - b) ** 2 
        for xi, yi in zip(X, Y)
    ) / len(X)

    print("Coeficiente (pendiente): ", slope)
    print("Intercepto: ", b)
    print("Suma de las distancias cuadradas", sum(
            d**2 for d in distances
        )
    )
    print("Error cuadrático medio: ", mse)
```
La función **linear_regression** se encarga de realizar el cálculo de la regresión lineal a partir de los datos proporcionados. Primero, extrae los valores del salario anual y del monto de compra de autos. Calcula las medias de ambos conjuntos de datos y utiliza esas medias para calcular la pendiente de la línea de regresión mediante la función **calculate_slope**. El valor de intersección (b) se obtiene restando el producto de la pendiente por la media del salario a la media del monto de compra. Luego, se muestran las predicciones y se calculan métricas como el error cuadrático medio y la suma de las distancias cuadradas entre los valores reales y los predichos.

3. **get_mean(data)**
```python
def get_mean(data):
    return sum(data) / len(data)
```
La función **get_mean** calcula la media de un conjunto de datos.

4. **calculate_slope(mean_x, mean_y, x, y)**
```python
def calculate_slope(mean_x, mean_y, x, y):
    numerator = sum(
        (xi - mean_x) * (yi - mean_y) 
        for xi, yi in zip(x, y)
    )
    denominator = sum(
        (xi - mean_x) ** 2 
        for xi in x
    )

    return numerator / denominator
```
La función **calculate_slope** calcula la pendiente de la línea de regresión utilizando la fórmula de mínimos cuadrados. La pendiente se obtiene como el cociente entre el numerador, que es la suma de los productos de las diferencias entre cada valor y su media, y el denominador, que es la suma de las diferencias cuadradas de los valores de la variable independiente.

5. **show_predictions(x, y, m, b)**
```python
def show_predictions(x, y, m, b):
    y_pred = [m * xi + b for xi in x]

    plt.scatter(x, y, c="blue")
    plt.plot(x, y_pred, c="red")
    plt.xlabel("Variable Independiente")
    plt.ylabel("Variable Dependiente")
    plt.show()
```
La función **show_predictions** se encarga de graficar los valores reales (con puntos dispersos) y la línea de regresión (en rojo), la cual se calcula utilizando los valores de la pendiente y el intercepto. De este modo, visualizamos la relación entre el salario anual y el monto de compra de autos.

### Resultados
<div class="charts">
    <div>
        <img src="../images/evidence/p01_img_01.jpg" alt="Gráfica con todos los datos">
        <span>Gráfica con todos los datos</span>
    </div>
    <div>
        <img src="../images/evidence/p01_img_02.jpg" alt="Gráfica con el 60% de los datos">
        <span>Gráfica con el 60% de los datos</span>
    </div>
</div>

<div class="screenshot">
    <div>
        <img src="../images/evidence/p01_img_03.jpg" alt="Captura de pantalla de los resultados">
        <span>Captura de pantalla de los resultados</span>
    </div>
</div>

## Conclusiones

El programa desarrollado permite realizar un análisis de regresión lineal para predecir el monto de compra de autos en función del salario anual, utilizando un conjunto de datos proporcionado. Mediante el cálculo de la pendiente y el intercepto, el programa puede generar una línea de regresión que describe la relación entre ambas variables. Además, se utiliza la visualización gráfica para observar cómo los datos se ajustan a dicha línea, lo que facilita la interpretación de los resultados.

Al aplicar la regresión tanto a la totalidad de los datos como a una muestra del 60%, el programa permite explorar la consistencia y precisión del modelo en diferentes escenarios. El cálculo del error cuadrático medio proporciona una métrica clara sobre la precisión de las predicciones generadas. En general, este programa ofrece una solución sencilla y efectiva para implementar y analizar un modelo de regresión lineal, que puede ser aplicado a otros conjuntos de datos con variables similares.
