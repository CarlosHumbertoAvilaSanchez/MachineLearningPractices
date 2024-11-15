import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris


def get_data():
    iris = load_iris()
    data = iris.data

    data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

    return data


def find_bmu(input_vector, weights):
    distances = np.sqrt(((weights - input_vector) ** 2).sum(axis=2))
    bmu_index = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
    return bmu_index


def update_weights(
    input_vector, weights, bmu_index, t, max_iter, init_learning_rate=0.5, init_radius=3
):
    learning_rate = init_learning_rate * (1 - t / max_iter)
    radius = init_radius * (1 - t / max_iter)
    for i in range(som_x):
        for j in range(som_y):
            dist_to_bmu = np.sqrt((i - bmu_index[0]) ** 2 + (j - bmu_index[1]) ** 2)

            if dist_to_bmu <= radius:
                influence = np.exp(-(dist_to_bmu**2) / (2 * (radius**2)))

                weights[i, j] += (
                    learning_rate * influence * (input_vector - weights[i, j])
                )


def train(max_iter):
    for t in range(max_iter):
        input_vector = data[np.random.randint(0, data.shape[0])]
        bmu_index = find_bmu(input_vector, weights)
        update_weights(input_vector, weights, bmu_index, t, max_iter)

    print("Entrenamiento completo.")
    plt.figure(figsize=(8, 8))

    for i in range(som_x):
        for j in range(som_y):
            weight = weights[i, j]

            plt.fill_between(
                [i, i + 1], [j, j], [j + 1, j + 1], color=weight, edgecolor="k"
            )


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
