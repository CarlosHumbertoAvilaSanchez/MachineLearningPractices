import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.losses = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

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

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls


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


def getStatistics(TN, FP, FN, TP):
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return accuracy, precision, recall


def printStatistics(cm, precision, accuracy, recall, validations):
    print("Matriz de confusión:", cm)

    print("Precisión:", precision)
    print("Exactitud:", accuracy)
    print("Recall:", recall)
    print("Validaciones cruzadas:", validations)


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
