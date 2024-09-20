import matplotlib.pyplot as plt
import pandas as pd


def read_car_purchasing_data():
    data = pd.read_csv("car_purchasing.csv", usecols=[5, 8], encoding="latin-1")
    return data


def linear_regression(data):
    X = data["annual Salary"].values
    Y = data["car purchase amount"].values

    mean_x = get_mean(X)
    mean_y = get_mean(Y)

    slope = calculate_slope(mean_x, mean_y, X, Y)

    b = mean_y - slope * mean_x

    show_predictions(X, Y, slope, b)

    print("Pendiente: ", slope)
    print("Intercepto: ", b)


def show_predictions(x, y, m, b):
    y_pred = [m * xi + b for xi in x]

    plt.scatter(x, y, c="blue")
    plt.plot(x, y_pred, c="red")
    plt.xlabel("Variable Independiente")
    plt.ylabel("Variable Dependiente")
    plt.show()


def get_mean(data):
    return sum(data) / len(data)


def calculate_slope(mean_x, mean_y, x, y):
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator = sum((xi - mean_x) ** 2 for xi in x)

    return numerator / denominator


if __name__ == "__main__":
    # PREDICCIÓN CON EL TODOS LOS DATOS
    print("Predicción con todos los datos")
    car_purchasing_data = read_car_purchasing_data()
    linear_regression(car_purchasing_data)

    # PREDICCIÓN CON UNA MUESTRA DEL 60% DE LOS DATOS
    print("\nPredicción con una muestra del 60% de los datos")
    sample_60 = car_purchasing_data.sample(frac=0.6, random_state=1)
    linear_regression(sample_60)
