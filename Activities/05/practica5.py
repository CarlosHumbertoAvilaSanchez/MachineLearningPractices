import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC


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


def getStatistics(TN, FP, FN, TP):
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return accuracy, precision, recall


def cm_statistics(cm):
    TN, FP, FN, TP = cm.ravel()
    accuracy, precision, recall = getStatistics(TN, FP, FN, TP)
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")


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
        cm = confusion_matrix(Y_test, predictions)
        cm_statistics(cm)
        score = np.mean(predictions == Y_test)
        scores.append(score)

    return np.mean(scores)


def best_C():
    param_grid = {"C": np.logspace(-5, 7, 20)}
    grid = GridSearchCV(
        estimator=SVC(kernel="rbf", gamma="scale"),
        param_grid=param_grid,
        scoring="accuracy",
        n_jobs=-1,
        cv=3,
        verbose=0,
        return_train_score=True,
    )
    grid.fit(X=X_train, y=y_train)
    return grid.best_params_["C"], grid.best_estimator_


def plotSVM(X, y, X_test, y_test, y_pred, model, features_names):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color="r", label="Setosa")
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color="g", label="Versicolor")
    plt.scatter(
        X_test[y_test != y_pred][:, 0],
        X_test[y_test != y_pred][:, 1],
        color="b",
        label="Misclassified",
    )

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)
    ax.contour(
        XX,
        YY,
        Z,
        colors="k",
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=["--", "-", "--"],
    )

    plt.xlabel(features_names[0])
    plt.ylabel(features_names[1])
    plt.legend()
    plt.show()


if __name__ == "__main__":

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X = X[:, :2]

    X = X[y != 2]
    y = y[y != 2]

    features_names = iris.feature_names
    target_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    best_c, model = best_C()
    print(f"El mejor valor de C es: {best_c}")
    cross_validation(X, y, model)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plotConfusionMatrix(cm)
    cm_statistics(cm)

    plotSVM(X, y, X_test, y_test, y_pred, model, features_names)
