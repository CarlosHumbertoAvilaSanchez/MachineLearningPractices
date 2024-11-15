---
fileName: p07
title: Práctica 7 k-medias (k-means)
student: Carlos Humberto Avila Sanchez
school: Centro Universitario de Ciencias Exactas e Ingenierías (CUCEI)
subject: Aprendizaje Máquina
teacher: Avila Cardenas Karla
classSection: Sección D01
---

## Introducción
El análisis de agrupamiento o *clustering* es una técnica de aprendizaje no supervisado cuyo objetivo es agrupar datos en subconjuntos o clústeres, de modo que los elementos dentro de un mismo clúster sean lo más similares posible entre sí y, al mismo tiempo, estén diferenciados de los elementos de otros clústeres. Esta técnica es fundamental en problemas donde no se dispone de etiquetas previas o clases definidas para los datos, y es común en campos como la biología, el marketing, la segmentación de usuarios y el análisis de patrones.

Uno de los algoritmos de clustering más populares y ampliamente utilizados es **K-Means**. Este método es conocido por su simplicidad y eficiencia, y se basa en la partición de los datos en `k` clústeres. Cada clúster está definido por un centroide, que representa el punto medio de los elementos que pertenecen a ese grupo. El algoritmo de K-Means itera entre dos pasos básicos: asignación de cada punto al centroide más cercano y actualización de los centroides en función de los puntos asignados.

### Funcionamiento del Algoritmo K-Means

1. **Inicialización**: Se seleccionan `k` puntos al azar para ser los centroides iniciales de los clústeres.
2. **Asignación**: Cada punto de datos es asignado al centroide más cercano, formando así un clúster.
3. **Actualización**: Los centroides se recalculan como el promedio de los puntos asignados a cada clúster.
4. **Convergencia**: Los pasos de asignación y actualización se repiten hasta que los centroides no cambian entre iteraciones o se alcanza el número máximo de iteraciones.

K-Means es particularmente eficaz para datos en los que los clústeres son de forma esférica y de tamaño similar. Sin embargo, puede presentar limitaciones cuando los datos contienen clústeres con diferentes formas o densidades. Además, el algoritmo es sensible a la elección de los centroides iniciales y puede converger a soluciones subóptimas si los centroides iniciales no están bien distribuidos. Para mitigar este problema, se suele ejecutar el algoritmo varias veces con diferentes inicializaciones y se selecciona la solución con el menor error cuadrático.

### Métricas de Evaluación para K-Means

La calidad de los clústeres en K-Means se evalúa comúnmente mediante el cálculo de la suma de las distancias cuadradas de cada punto a su centroide respectivo, también conocido como **inercia** o **suma de los errores cuadráticos dentro de los clústeres**. Esta métrica indica cuán compactos son los clústeres y ayuda a determinar la adecuación del valor de `k`. 

### Selección del Número de Clústeres `k`

Elegir el número óptimo de clústeres `k` es uno de los desafíos principales al usar K-Means. Existen varias técnicas para seleccionar el mejor valor de `k`, entre las que destaca el **método del codo**. Este método evalúa la inercia para diferentes valores de `k`, y se elige el valor donde la inercia deja de disminuir significativamente, formando un "codo" en la gráfica de inercia vs. número de clústeres.

### Aplicaciones de K-Means

El algoritmo K-Means tiene aplicaciones en diversos campos, incluyendo:

- **Segmentación de clientes**: Agrupando a los clientes en función de su comportamiento de compra, se pueden desarrollar estrategias de marketing personalizadas.
- **Compresión de imágenes**: Agrupando colores similares en una imagen para reducir la cantidad de colores, lo que ayuda a disminuir el tamaño de la imagen sin perder mucha calidad.
- **Biología y genética**: Agrupando genes o especies en función de similitudes genéticas para entender relaciones evolutivas.

### Objetivo de esta Práctica

En este proyecto, se implementa el algoritmo K-Means desde cero para agrupar datos del conjunto de datos iris. Este conjunto de datos es un clásico en el aprendizaje automático y contiene características de tres especies de flores. Sin embargo, para facilitar la visualización y el análisis, se seleccionan solo dos características y tres clústeres (uno por cada especie). Al implementar K-Means desde cero, se busca comprender a fondo el funcionamiento interno del algoritmo y visualizar los resultados para interpretar los patrones dentro de los datos. La visualización de los centroides y de los clústeres formados permite evaluar visualmente la calidad del agrupamiento y analizar cómo se distribuyen los datos en cada grupo.
## Contenido

### Librerías utilizadas 

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
```
Estas librerías permiten la manipulación de datos, la visualización de gráficos para representar los clústeres, y la carga del conjunto de datos iris.

### Clase K-means
La clase **KMeans** implementa el algoritmo de agrupamiento K-Means desde cero. Utiliza un enfoque iterativo para ajustar los centroides de cada clúster y asignar cada punto de datos al centroide más cercano.

*__init__(self, n_clusters, max_iter=300)*

```python
def __init__(self, n_clusters, max_iter=300):
    self.n_clusters = n_clusters
    self.max_iter = max_iter
```
El constructor inicializa el número de clústeres n_clusters y el número máximo de iteraciones max_iter que el algoritmo ejecutará para encontrar los centroides óptimos.

*fit(self, X)*

```python
def fit(self, X):
    self.centroids = X[
        np.random.choice(
            X.shape[0], self.n_clusters, replace=False
        )
    ]

    for _ in range(self.max_iter):
        labels = self._assign_labels(X)
        new_centroids = np.array(
            [
                X[labels == i].mean(axis=0) 
                for i in range(self.n_clusters)
            ]
        )
        if np.all(self.centroids == new_centroids):
            break
        self.centroids = new_centroids
    self.labels_ = self._assign_labels(X)
    return self
```

El método **fit()** ejecuta el algoritmo de K-Means y sigue los siguientes pasos:

1. Inicialización de centroides: Se seleccionan aleatoriamente n_clusters puntos de datos como centroides iniciales.
2. Asignación de etiquetas: Cada punto de datos se asigna al clúster cuyo centroide esté más cercano, utilizando la función interna _assign_labels.
3. Actualización de centroides: Los centroides se recalculan como el promedio de los puntos asignados a cada clúster.
4. Verificación de convergencia: Si los centroides no cambian entre iteraciones, el algoritmo detiene la ejecución, indicando que ha convergido.

*_assign_labels(self, X)*

```python
def _assign_labels(self, X):
    distances = np.sqrt(
        ((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2)
    )
    return np.argmin(distances, axis=1)
```

Este método privado calcula las **distancias euclidianas** entre cada punto de datos y los centroides. Asigna cada punto al centroide más cercano en función de la distancia mínima.

### Funciones adicionales

*load_iris()*

```python
def load_iris():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X = X[:, :2]
    return X, y
```
Esta función carga el conjunto de datos de iris y selecciona solo las dos primeras características (para visualización en 2D). Retorna las características **X** y las etiquetas reales **y**, que se utilizarán para comparar visualmente los resultados del modelo K-Means.

*plot_model(X, y, labels, centroids)*

```python
def plot_model(X, y, labels, centroids):
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap="viridis")
    plt.scatter(centroids[:, 0], centroids[:, 1], c="red", s=200, alpha=0.5)
    plt.show()
    print(centroids)
```

La función **plot_model()** visualiza los resultados del modelo K-Means, mostrando los puntos de datos coloreados por el clúster al que pertenecen y marcando los centroides en rojo. Esto permite ver cómo el modelo ha agrupado los datos.


### Bloque principal

```python
if __name__ == "__main__":
    X, y = load_iris()
    kmeans_custom = KMeans(n_clusters=3)
    kmeans_custom.fit(X)
    centroids = kmeans_custom.centroids
    labels = kmeans_custom.labels_
    plot_model(X, y, labels, centroids)
```
En el bloque principal:

1. Se carga el conjunto de datos iris utilizando la función **load_iris()**.
2. Se crea una instancia de la clase KMeans con n_clusters=3.
3. Se ajusta el modelo K-Means a los datos de X con el método **fit()**.
4. Finalmente, se visualizan los clústeres resultantes y los centroides utilizando la función **plot_model()**.

## Resultados.
<div class="charts">
    <div>
        <img src="../images/evidence/p06_img_01.jpg" alt="3 centroides">
        <span>3 centroides</span>
    </div>
    <div>
        <img src="../images/evidence/p06_img_02.jpg" alt="5 centroides">
        <span>5 centroides</span>
    </div>
     <div>
        <img src="../images/evidence/p06_img_03.jpg" alt="10 centroides">
        <span>10 centroides</span>
    </div>
</div>

## Conclusión

En esta práctica, se implementó y evaluó el algoritmo K-Means para agrupar datos del conjunto de iris, usando valores de `k` de 3, 5 y 10 para observar cómo el número de clústeres afecta la agrupación de los datos y la distribución de los centroides.

Al utilizar `k=3`, el modelo se ajusta bien a las tres especies de iris presentes en el conjunto de datos, mostrando agrupaciones coherentes con las etiquetas originales. Este valor de `k` proporcionó grupos compactos y centrados, lo que hace de este agrupamiento una buena representación de las clases en los datos.

Con `k=5`, el algoritmo K-Means creó más subgrupos dentro de cada clase, permitiendo un análisis más detallado de las similitudes entre las muestras. Aunque este valor de `k` segmenta la variabilidad en cada clúster, algunos grupos pueden superponerse, indicando una menor separación entre especies.

Finalmente, al utilizar `k=10`, se formaron más clústeres pequeños que resaltan variaciones locales en el conjunto de datos. Sin embargo, el aumento de clústeres también genera redundancia, pues algunos grupos no representan diferencias significativas en la estructura de los datos.

En conclusión, esta práctica permitió observar cómo el valor de `k` influye en la agrupación de datos en K-Means. Si bien `k=3` fue adecuado para este conjunto de datos, valores mayores de `k` proporcionan una mayor segmentación que puede ser útil para análisis específicos. La implementación y visualización de estos agrupamientos ofrece una comprensión clara de cómo K-Means puede adaptarse a distintos patrones en los datos según el valor de `k`.