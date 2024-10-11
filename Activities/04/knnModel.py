from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import plotly_express as px
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

COL_NAMES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "species",
]


def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance


class KnnModel:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        predictions = [self._predict(x) for x in np.array(X)]
        return predictions

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]


def accuracy_score(y_true, y_pred):
    correct_predictions = sum(y_true == y_pred)
    accuracy = correct_predictions / len(y_true)
    return accuracy


def cross_validation(X, Y, model, training_size=0.7):
    n = len(X)
    training_size = int(n * training_size)
    scores = []

    for _ in range(10):
        idx = np.random.permutation(n)
        X_train, X_test = X[idx][:training_size], X[idx][training_size:]
        Y_train, Y_test = Y[idx][:training_size], Y[idx][training_size:]

        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        score = np.mean(predictions == Y_test)
        scores.append(score)

    return np.mean(scores)


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


def print_statistics(Y_test, Y_predict, validations):
    print(metrics.classification_report(Y_test, Y_predict))
    print(f"Validación cruzada: {validations}")


def cm(Y_test, Y_predict):
    KNN_cm = confusion_matrix(Y_test, Y_predict)
    f, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(
        KNN_cm, annot=True, linewidth=0.7, linecolor="cyan", fmt="g", ax=ax, cmap="BuPu"
    )
    plt.title("Matríz de confusión de modelo KNN")
    plt.xlabel("Y predict")
    plt.ylabel("Y test")
    plt.show()


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
