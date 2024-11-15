import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


class KMeans:
    def __init__(self, n_clusters, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        # Paso 1: Inicialización de centroides
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            # Paso 2: Asignación de puntos al centroide más cercano
            labels = self._assign_labels(X)
            # Paso 3: Actualización de centroides
            new_centroids = np.array(
                [X[labels == i].mean(axis=0) for i in range(self.n_clusters)]
            )
            # Comprobación de convergencia
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids
            # Visualización
            # plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap="viridis")
            # plt.scatter(
            #     self.centroids[:, 0], self.centroids[:, 1], c="red", s=200, alpha=0.5
            # )
            # plt.show()
        self.labels_ = self._assign_labels(X)
        return self

    def _assign_labels(self, X):
        # Calcular distancias de cada punto a los centroides
        distances = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
        # Asignar etiquetas según la distancia mínima
        return np.argmin(distances, axis=1)


def load_iris():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X = X[:, :2]
    return X, y


def plot_model(X, y, labels, centroids):
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap="viridis")
    plt.scatter(centroids[:, 0], centroids[:, 1], c="red", s=200, alpha=0.5)
    plt.show()
    print(centroids)


if __name__ == "__main__":
    X, y = load_iris()
    kmeans_custom = KMeans(n_clusters=10)
    kmeans_custom.fit(X)
    centroids = kmeans_custom.centroids
    labels = kmeans_custom.labels_
    plot_model(X, y, labels, centroids)
