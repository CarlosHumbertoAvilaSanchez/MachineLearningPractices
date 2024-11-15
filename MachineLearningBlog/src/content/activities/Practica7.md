---
fileName: p08
title: Práctica 8 SOM
student: Carlos Humberto Avila Sanchez
school: Centro Universitario de Ciencias Exactas e Ingenierías (CUCEI)
subject: Aprendizaje Máquina
teacher: Avila Cardenas Karla
classSection: Sección D01
---

## Introducción

 Los Mapas Autoorganizados, también conocidos como **Self-Organizing Maps (SOM)**, son un tipo de red neuronal no supervisada desarrollada por Teuvo Kohonen en la década de 1980. Su principal objetivo es reducir la dimensionalidad de los datos al mismo tiempo que se conserva su estructura topológica. En otras palabras, los SOM proyectan datos de alta dimensión en una cuadrícula bidimensional (o tridimensional) donde la relación entre los datos originales permanece intacta. Esta técnica es especialmente útil para explorar datos complejos y visualizar agrupamientos y patrones de datos que no poseen etiquetas previas, haciendo que los SOM se utilicen frecuentemente en problemas de minería de datos, análisis de patrones y clasificación no supervisada.

### Funcionamiento de los Mapas Autoorganizados

Un SOM consiste en una cuadrícula de neuronas, o nodos, cada uno de los cuales representa un vector de características (o peso) de la misma dimensión que los datos de entrada. El entrenamiento de un SOM se basa en los siguientes pasos:

1. **Inicialización**: Se inicializan los pesos de las neuronas de la cuadrícula de manera aleatoria.
2. **Identificación de la Unidad de Mejor Correspondencia (BMU)**: Para cada vector de entrada, se calcula la distancia euclidiana entre este vector y el vector de pesos de cada neurona. La neurona con el peso más cercano al vector de entrada se denomina Unidad de Mejor Correspondencia (BMU).
3. **Actualización de Pesos**: Los pesos de la BMU y de las neuronas vecinas se ajustan para que se parezcan más al vector de entrada. Este ajuste se realiza mediante una función de influencia que disminuye con la distancia a la BMU y que también decrece a medida que avanzan las iteraciones.
4. **Convergencia**: El proceso de ajuste se repite para cada vector de entrada durante un número determinado de iteraciones, permitiendo que los pesos de las neuronas converjan y reflejen la estructura de los datos de entrada en la cuadrícula.

### Parámetros Clave de un SOM

Para que el SOM se ajuste correctamente a los datos, es importante configurar algunos parámetros clave:

- **Tasa de aprendizaje**: Controla la magnitud de los cambios en los pesos. Comienza con un valor inicial y disminuye progresivamente en cada iteración, permitiendo que el modelo ajuste sus pesos de manera gradual.
- **Radio de vecindad**: Determina el área de influencia alrededor de la BMU. Las neuronas dentro de este radio se ajustan en función del vector de entrada. Al igual que la tasa de aprendizaje, este radio también disminuye con las iteraciones.
- **Dimensión de la cuadrícula**: La cantidad de neuronas en el SOM afecta la resolución y nivel de detalle con el que el SOM puede representar los datos. Cuadrículas más grandes permiten más detalles, pero aumentan el costo computacional.

### Ventajas y Limitaciones de los SOM

**Ventajas:**
- **Visualización**: Los SOM facilitan la visualización de datos complejos y de alta dimensionalidad en una cuadrícula 2D, permitiendo una interpretación más clara de los patrones subyacentes.
- **Preservación de la topología**: Los datos que son similares en el espacio de características original se proyectan en neuronas cercanas dentro del SOM, lo que permite identificar clústeres y relaciones.
- **Capacidad de agrupamiento**: Aunque no es un algoritmo de clustering en sentido estricto, los SOM agrupan datos de manera efectiva, lo que puede ser útil en tareas de segmentación.

**Limitaciones:**
- **Escalabilidad**: A medida que aumentan las dimensiones de los datos o la cantidad de datos, el costo computacional también se incrementa significativamente.
- **Dependencia de los parámetros iniciales**: La configuración de la tasa de aprendizaje, el radio de vecindad y el tamaño de la cuadrícula puede afectar considerablemente los resultados.
- **Dificultad para capturar estructuras complejas**: Para datos con estructuras intrincadas o clústeres de forma irregular, los SOM pueden no ser tan efectivos como otros algoritmos de agrupamiento.

### Aplicaciones de los SOM

Los Mapas Autoorganizados tienen aplicaciones en diversas áreas, tales como:

- **Segmentación de clientes**: Agrupación de clientes en segmentos con base en sus comportamientos de compra.
- **Genética**: Análisis y visualización de relaciones entre genes o patrones de expresión génica.
- **Reconocimiento de patrones**: Identificación de patrones de actividad en datos de sensores o imágenes.
- **Exploración de datos no estructurados**: Aplicación en minería de datos y análisis de datos sin etiquetas, como texto o datos de redes sociales.

### Objetivo de esta Práctica

En esta práctica, se implementa un SOM desde cero para analizar el conjunto de datos iris. Este conjunto de datos contiene características de tres especies de flores y es ampliamente utilizado para experimentos de clasificación y agrupamiento. El proceso comienza con la normalización de los datos, que es fundamental para que el SOM funcione correctamente. Posteriormente, se entrena el SOM y se visualizan los resultados en una cuadrícula donde cada celda representa el vector de pesos de una neurona. El resultado final permite observar cómo los datos de las diferentes especies de iris se distribuyen en la cuadrícula, evidenciando los patrones de similitud entre las muestras.

La implementación de este SOM facilita una comprensión práctica del funcionamiento de los Mapas Autoorganizados y muestra cómo este algoritmo puede agrupar y representar datos de alta dimensión en una estructura organizada y visualmente interpretable.

## Contenido

### Librerías utilizadas

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
```
Estas librerías permiten la manipulación de datos, la creación de gráficos y la carga del conjunto de datos iris.

### Funciones

*get_data()*

```python
def get_data():
    iris = load_iris()
    data = iris.data

    data = (data - data.min(axis=0)) / (
        data.max(axis=0) - data.min(axis=0)
    )

    return data
```
Esta función **get_data()** carga el conjunto de datos de iris y realiza una normalización de las características para escalarlas entre 0 y 1. Esta normalización es fundamental para que el SOM funcione adecuadamente, ya que los datos escalados permiten que las actualizaciones de los pesos sean consistentes a lo largo del mapa.

*find_bmu(input_vector, weights)*

```python
def find_bmu(input_vector, weights):
    distances = np.sqrt(
        ((weights - input_vector) ** 2).sum(axis=2)
    )
    bmu_index = np.unravel_index(
        np.argmin(distances, axis=None), distances.shape
    )
    return bmu_index
```
La función **find_bmu()** identifica la Unidad de Mejor Correspondencia (BMU) para un vector de entrada dado. Calcula la distancia euclidiana entre el vector de entrada y cada peso en el SOM, y selecciona el índice del peso más cercano como la BMU. Esta unidad es el nodo que más se parece al vector de entrada y se utiliza para actualizar los pesos del SOM.

*update_weights(input_vector, weights, bmu_index, t, max_iter, init_learning_rate=0.5, init_radius=3)*

```python
def update_weights(input_vector, 
                   weights, 
                   bmu_index, t, max_iter, 
                   init_learning_rate=0.5, 
                   init_radius=3):
    learning_rate = init_learning_rate * (1 - t / max_iter)
    radius = init_radius * (1 - t / max_iter)
    for i in range(som_x):
        for j in range(som_y):
            dist_to_bmu = np.sqrt(
                (i - bmu_index[0]) ** 2 + (j - bmu_index[1]) ** 2
            )

            if dist_to_bmu <= radius:
                influence = np.exp(
                    -(dist_to_bmu**2) / (2 * (radius**2))
                )

                weights[i, j] += (
                    learning_rate * influence * (
                        input_vector - weights[i, j]
                    )
                )
```
La función **update_weights()** ajusta los pesos de las neuronas en el SOM en cada iteración. Dado el índice de la BMU, esta función:

1. Calcula la tasa de aprendizaje y el radio de vecindad que disminuyen gradualmente a medida que avanzan las iteraciones.

2. Actualiza los pesos de las neuronas dentro del radio de vecindad de la BMU, aplicando una influencia exponencial que disminuye con la distancia a la BMU. Esto permite que los nodos cercanos a la BMU también se adapten al vector de entrada, preservando así la estructura topológica del SOM.

*train(max_iter)*

```python
def train(max_iter):
    for t in range(max_iter):
        input_vector = data[np.random.randint(0, data.shape[0])]
        bmu_index = find_bmu(input_vector, weights)
        update_weights(
            input_vector, 
            weights, 
            bmu_index, 
            t, 
            max_iter
        )

    print("Entrenamiento completo.")
    plt.figure(figsize=(8, 8))

    for i in range(som_x):
        for j in range(som_y):
            weight = weights[i, j]

            plt.fill_between(
                [i, i + 1], 
                [j, j], 
                [j + 1, j + 1], 
                color=weight, 
                edgecolor="k"
            )
```

El método **train()** ejecuta el entrenamiento del SOM a lo largo de max_iter iteraciones. En cada iteración:

1. Selecciona aleatoriamente un vector de entrada.
2. Identifica la BMU para este vector.
3. Actualiza los pesos de los nodos cercanos a la BMU.
4. Al finalizar, el SOM muestra un gráfico donde cada celda de la cuadrícula se colorea según el peso final de cada neurona.

### Bloque principal

```python
if __name__ == "__main__":
    max_iter = 1000
    som_x, som_y = 15, 15
    data = get_data()
    input_dim = data.shape[1]

    weights = np.random.rand(som_x, som_y, input_dim)

    train(max_iter)

    plt.title("Mapa Autoorganizado de Iris")
    plt.axis("off")
    plt.show()
```
En el bloque principal:

1. Se establecen los parámetros para el entrenamiento, incluyendo el número de iteraciones y el tamaño de la cuadrícula del SOM.
2. Se normalizan los datos de iris con *get_data()*.
3. Se inicializan los pesos del SOM aleatoriamente.
4. Se entrena el SOM utilizando la función *train()*.
5. Finalmente, se visualiza el mapa SOM, que muestra la estructura topológica de los datos de iris proyectada en un espacio bidimensional.

### Resultados

<div class="charts">
    <div>
        <img src="../images/evidence/p07_img_01.jpg" alt="Mapa 5 X 5">
        <span>Mapa 5 X 5</span>
    </div>
    <div>
        <img src="../images/evidence/p07_img_02.jpg" alt="Mapa 10 X 10">
        <span>Mapa 10 X 10</span>
    </div>
     <div>
        <img src="../images/evidence/p07_img_03.jpg" alt="Mapa 15 X 15">
        <span>Mapa 15 X 15</span>
    </div>
</div>

## Conclusión

En esta práctica se implementó un Mapa Autoorganizado (SOM) para analizar y visualizar el conjunto de datos de iris, utilizando una cuadrícula de neuronas que representa las relaciones entre los datos de alta dimensión. Al entrenar el SOM, se logró proyectar los datos de las tres especies de flores en una cuadrícula bidimensional, manteniendo la estructura topológica de los datos originales y agrupando de forma natural las muestras similares.

El SOM permitió observar cómo los datos de las distintas especies de iris se agrupan en regiones específicas de la cuadrícula, mostrando la efectividad de este modelo para conservar patrones en el espacio de características. La visualización final de los pesos de cada neurona como colores en la cuadrícula ofreció una representación clara y accesible de la agrupación de datos, ayudando a comprender las similitudes y diferencias entre las especies.

Esta práctica demostró que los SOM son una herramienta poderosa para la reducción de dimensionalidad y la exploración visual de datos complejos, aunque su rendimiento puede depender de la configuración de parámetros como la tasa de aprendizaje, el radio de vecindad y la dimensión de la cuadrícula. En conjunto, la implementación de este SOM ofrece una visión clara sobre el funcionamiento y utilidad de los Mapas Autoorganizados en problemas de agrupamiento y análisis no supervisado de datos.