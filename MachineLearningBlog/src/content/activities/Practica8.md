---
fileName: p09
title: Práctica 9 Modelado basado en agentes
student: Carlos Humberto Avila Sanchez
school: Centro Universitario de Ciencias Exactas e Ingenierías (CUCEI)
subject: Aprendizaje Máquina
teacher: Avila Cardenas Karla
classSection: Sección D01
---

## Introducción
En esta práctica exploramos el modelado basado en agentes (ABM, por sus siglas en inglés) mediante la simulación de un enjambre de abejas que interactúan con su entorno. Este enfoque es ideal para modelar sistemas complejos, ya que representa entidades individuales (los agentes) que siguen reglas simples y generan comportamientos emergentes cuando interactúan.

El modelo incluye abejas como agentes individuales, que se mueven dentro de una cuadrícula y responden a señales químicas simuladas en su entorno. Estas señales se difunden y evaporan con el tiempo, creando dinámicas que afectan el comportamiento de los agentes.

La simulación tiene como objetivos:

Representar cómo los agentes adaptan su movimiento basado en estímulos locales.
Analizar fenómenos emergentes como la autoorganización o patrones formados por interacciones simples.
Este ejercicio es una introducción práctica al ABM, usado en disciplinas como la biología, ecología y ciencias sociales, para comprender sistemas donde las interacciones locales generan comportamientos globales.

## Contenido

### Librerías externas
- **random**: Proporciona funciones para generar números aleatorios.
- **numpy**: Utilizada para manejar operaciones matriciales y numéricas de manera eficiente.
- **pygame**: Utilizada para manejar gráficos y eventos en la simulación.

```python
import random
import numpy as np
import pygame
```

### Constantes

- `WIDTH`: Ancho de la ventana de la simulación (800 px).
- `HEIGHT`: Altura de la ventana de la simulación (800 px).
- `GRID_SIZE`: Tamaño de la cuadrícula (50 celdas por lado).
- `CELL_SIZE`: Tamaño de cada celda en píxeles (`WIDTH // GRID_SIZE`).
- `NUM_BEES`: Número total de abejas en la simulación (100).
- `EVAPORATION_RATE`: Tasa de evaporación de la química en el ambiente (0.9).
- `DIFFUSION_RATE`: Tasa de difusión de la química entre las celdas (0.1).

```python
WIDTH, HEIGHT = 800, 800  
GRID_SIZE = 50  
CELL_SIZE = WIDTH // GRID_SIZE  
NUM_BEES = 100  
EVAPORATION_RATE = 0.9  
DIFFUSION_RATE = 0.1
```

### Clases

#### Bee
Representa a una abeja en la simulación.

#### Atributos
- `x`: Posición X de la abeja en la cuadrícula.
- `y`: Posición Y de la abeja en la cuadrícula.
- `angle`: Ángulo de movimiento de la abeja (en grados).
- `color`: Color de la abeja en tonos amarillos.

#### Métodos
- `__init__()`: Inicializa la posición, ángulo y color aleatorios de la abeja.

```python
def __init__(self):
        self.x = random.randint(0, GRID_SIZE - 1)
        self.y = random.randint(0, GRID_SIZE - 1)
        self.angle = random.uniform(0, 360)
        self.color = (200 + random.randint(-30, 30), 200, 0)
```

- `move()`: Actualiza la posición de la abeja basándose en la química del ambiente y su velocidad.

```python
def move(self):
        cx, cy = int(self.x), int(self.y)
        self.angle += 4 * chemical[cx, cy]

        speed = 1 + ((chemical[cx, cy] ** 2) / 60)
        self.x += speed * np.cos(np.radians(self.angle))
        self.y += speed * np.sin(np.radians(self.angle))

        self.x = self.x % GRID_SIZE
        self.y = self.y % GRID_SIZE

        chemical[cx, cy] += 2
```

### Funciones

#### `diffuse_chemical()`
Simula la difusión de la química en el ambiente, actualizando los valores de cada celda en función de sus vecinos.
```python
def diffuse_chemical():
    global chemical
    new_chemical = np.copy(chemical)
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            total = 0
            neighbors = 0
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    total += chemical[nx, ny]
                    neighbors += 1
            new_chemical[x, y] += DIFFUSION_RATE * (
                total / neighbors - chemical[x, y]
            )
    chemical = new_chemical
```

#### `evaporate_chemical()`
Reduce la cantidad de química en todas las celdas, simulando la evaporación.

```python
def evaporate_chemical():
    global chemical
    chemical *= EVAPORATION_RATE
```
#### `draw(bees, screen)`
Dibuja la cuadrícula y las abejas en la pantalla.

##### Parámetros
- `bees`: Lista de objetos `Bee`.
- `screen`: Superficie de pygame donde se dibujará.

```python
def draw(bees, screen):
    
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            value = min(chemical[x, y], 20)
            color = (value * 12, value * 12, value * 12) 
            pygame.draw.rect(
                screen, color, (
                    x * CELL_SIZE, 
                    y * CELL_SIZE, CELL_SIZE, 
                    CELL_SIZE
                )
            )

    for bee in bees:
        px, py = int(bee.x * CELL_SIZE), 
                 int(bee.y * CELL_SIZE)
        pygame.draw.circle(
            screen, 
            bee.color, 
            (px, py), 
            CELL_SIZE // 4
        )
```

#### `simulation(bees, screen, clock)`
Ejecuta el ciclo principal de la simulación.

##### Parámetros
- `bees`: Lista de abejas en la simulación.
- `screen`: Superficie de pygame para mostrar los gráficos.
- `clock`: Objeto de pygame para controlar la velocidad de actualización.

```python
def simulation(bees, screen, clock):
    running = True
    ticks = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        for bee in bees:
            bee.move()

        if ticks % 10 == 0:
            diffuse_chemical()
            evaporate_chemical()

        screen.fill((0, 0, 0))
        draw(bees, screen)
        pygame.display.flip()

        ticks += 1
        clock.tick(30)
    pygame.quit()
```

### Bloque Principal

El bloque principal inicializa los elementos necesarios para la simulación y la ejecuta.

1. Crea una lista de `Bee` con `NUM_BEES` instancias.
2. Inicializa pygame y configura la ventana gráfica.
3. Crea la cuadrícula `chemical` como una matriz de ceros (`numpy`).
4. Ejecuta la función `simulation()` para comenzar el bucle principal.

```python
if __name__ == "__main__":
    bees = [Bee() for _ in range(NUM_BEES)]
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Simulación de abejas")
    clock = pygame.time.Clock()
    chemical = np.zeros((GRID_SIZE, GRID_SIZE))
    simulation(bees, screen, clock)
```

### Resultados
<div class="charts">
    <div>
        <img src="../images/evidence/p08_img_01.jpg" alt="Inicio de la simulación">
        <span>Inicio de la simulación</span>
    </div>
    <div>
        <img src="../images/evidence/p08_img_02.jpg" alt="Resultado de la simulación">
        <span>Resultado</span>
    </div>
</div>

## Conclusión

La práctica demostró cómo el modelado basado en agentes (ABM) permite explorar dinámicas complejas mediante la interacción de agentes individuales con su entorno. A través de la simulación de un enjambre de abejas, observamos cómo reglas locales simples, como el movimiento guiado por estímulos químicos, pueden generar comportamientos emergentes en un sistema.

El uso de herramientas como pygame y numpy facilitó la implementación gráfica y computacional del modelo, permitiendo visualizar la evolución de los patrones químicos y las trayectorias de las abejas. Además, las dinámicas de difusión y evaporación proporcionaron un entorno cambiante que enriqueció el comportamiento de los agentes.

Este ejercicio destaca la utilidad del ABM para comprender fenómenos donde las interacciones locales producen efectos globales, siendo aplicable a áreas como la ecología, el diseño urbano y la inteligencia artificial. La simulación realizada no solo refuerza conceptos teóricos, sino que también sienta las bases para estudios más avanzados en modelado de sistemas complejos.