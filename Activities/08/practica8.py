import random

import numpy as np
import pygame

WIDTH, HEIGHT = 800, 800
GRID_SIZE = 50
CELL_SIZE = WIDTH // GRID_SIZE
NUM_BEES = 100
EVAPORATION_RATE = 0.9
DIFFUSION_RATE = 0.1


# Crear las abejas
class Bee:
    def __init__(self):
        self.x = random.randint(0, GRID_SIZE - 1)
        self.y = random.randint(0, GRID_SIZE - 1)
        self.angle = random.uniform(0, 360)
        self.color = (200 + random.randint(-30, 30), 200, 0)

    def move(self):
        cx, cy = int(self.x), int(self.y)
        self.angle += 4 * chemical[cx, cy]

        speed = 1 + ((chemical[cx, cy] ** 2) / 60)
        self.x += speed * np.cos(np.radians(self.angle))
        self.y += speed * np.sin(np.radians(self.angle))

        self.x = self.x % GRID_SIZE
        self.y = self.y % GRID_SIZE

        chemical[cx, cy] += 2


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
            new_chemical[x, y] += DIFFUSION_RATE * (total / neighbors - chemical[x, y])
    chemical = new_chemical


def evaporate_chemical():
    global chemical
    chemical *= EVAPORATION_RATE


def draw(bees, screen):
    # Dibujar la cuadrícula
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            value = min(chemical[x, y], 20)
            color = (value * 12, value * 12, value * 12)  # Escalar a tonos de gris
            pygame.draw.rect(
                screen, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            )

    # Dibujar las abejas
    for bee in bees:
        px, py = int(bee.x * CELL_SIZE), int(bee.y * CELL_SIZE)
        pygame.draw.circle(screen, bee.color, (px, py), CELL_SIZE // 4)


def simulation(bees, screen, clock):
    running = True
    ticks = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Actualizar las abejas
        for bee in bees:
            bee.move()

        # Actualizar la química cada 10 ticks
        if ticks % 10 == 0:
            diffuse_chemical()
            evaporate_chemical()

        # Dibujar
        screen.fill((0, 0, 0))
        draw(bees, screen)
        pygame.display.flip()

        ticks += 1
        clock.tick(30)
    pygame.quit()


# Bucle principal
if __name__ == "__main__":
    bees = [Bee() for _ in range(NUM_BEES)]
    # Inicializar pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Simulación de abejas")
    clock = pygame.time.Clock()

    # Crear la cuadrícula de "patches"
    chemical = np.zeros((GRID_SIZE, GRID_SIZE))
    simulation(bees, screen, clock)
