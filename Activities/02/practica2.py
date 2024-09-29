import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def get_data():
    data = pd.read_csv("Student_Performance.csv", encoding="latin-1")
    data = data.sample(frac=0.01, random_state=1)
    return data


def plot_data(model, X, Y):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(X[:, 0], X[:, 1], Y, c="b", marker="o")

    xx, yy = np.meshgrid(X[:, 0], X[:, 1])

    zz = model.intercept_ + model.coef_[0] * xx + model.coef_[1] * yy
    ax.plot_surface(xx, yy, zz, color="r", alpha=0.01)

    ax.set_xlabel("Hours Studied")
    ax.set_ylabel("Previous Scores")
    ax.set_zlabel("Performance Index")
    plt.show()


def print_statistics(model):
    print("Intercepto: ", model.intercept_)
    print("Coeficientes: ", model.coef_)


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
        mse = np.mean((Y_test - predictions) ** 2)
        scores.append(mse)

    return np.mean(scores)


def k_fold_cross_validation(X, Y, model, k=10):
    x_folds = np.array_split(X, k)
    y_folds = np.array_split(Y, k)
    scores = []

    for i in range(k):
        x_test = x_folds[i]
        y_test = y_folds[i]
        x_train = np.concatenate(x_folds[:i] + x_folds[i + 1 :])
        y_train = np.concatenate(y_folds[:i] + y_folds[i + 1 :])
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        mse = np.mean((y_test - predictions) ** 2)
        scores.append(mse)

    return np.mean(scores)


class MultipleLinearRegression:
    def __init__(self, learning_rate=0.0000001, n_iter=1000000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X_b = np.column_stack((np.ones(len(X)), X))
        self.coef_ = np.random.rand(X_b.shape[1])

        for _ in range(self.n_iter):
            gradient = -2 * X_b.T @ (y - X_b @ self.coef_)
            self.coef_ -= self.learning_rate * gradient

        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]

    def predict(self, X):
        X_b = np.column_stack((np.ones(len(X)), X))
        return X_b @ np.concatenate(([self.intercept_], self.coef_))


if __name__ == "__main__":
    data = get_data()
    X = np.array(
        [
            data["Hours Studied"].values,
            data["Previous Scores"].values,
        ]
    ).T

    Y = np.array(data["Performance Index"])

    modelG = MultipleLinearRegression()
    modelG.fit(X, Y)
    plot_data(modelG, X, Y)
    print_statistics(modelG)

    model = LinearRegression()
    model.fit(X, Y)
    plot_data(model, X, Y)
    print_statistics(model)

    print("K-Folk validation: ", k_fold_cross_validation(X, Y, modelG))
    print("Cross validation: ", cross_validation(X, Y, model))
