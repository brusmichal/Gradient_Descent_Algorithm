import numpy as np
import autograd as ag
import math
import matplotlib.pyplot as plt


def f1(x):
    return x ** 4


def f2(x):
    e = math.e
    return 1.5 - (e ** (-x[0] ** 2 - x[1] ** 2)) - 0.5 * (e ** (-(x[0] - 1) ** 2 - (x[1] + 2) ** 2))


def grad_f1(x):
    return 4 * x ** 3


def grad_f2(x):
    x1 = x[0]
    x2 = x[1]
    return np.array([2 * x1 * np.exp(-x1 ** 2 - x2 ** 2) + (x1 - 1) * np.exp(-(x1 - 1) ** 2 - (x2 + 2) ** 2),
                     2 * x2 * np.exp(-x1 ** 2 - x2 ** 2) + (x2 + 2) * np.exp(-(x1 - 1) ** 2 - (x2 + 2) ** 2)
                     ])


def gradient_descent(function, dimensions, max_x, max_iter, learn_rate=0.001):
    grad_f = ag.grad(function)
    x = np.random.uniform(-max_x, max_x, dimensions)
    points_history = np.array([x])
    for _ in range(max_iter - 1):
        diff = grad_f(x)
        x = x - learn_rate * diff
        x[x < -max_x] = -max_x
        x[x > max_x] = max_x
        points_history = np.append(points_history, [x], axis=0)

    optimum = np.array([x, function(x)])
    return optimum, points_history

#funkcja 1

# x_steps, final_result = gradient_descent(grad_f1, 8, 0.00391, 1000, 0.01)
#
# x_points = np.arange(-10, 10, 0.5)
# y_points = f1(x_points)
# y_steps = f1(x_steps)
#
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Function f1 with gradient descent steps")
# plt.plot(x_points, y_points)
# plt.plot(x_steps, y_steps)
# print(x_steps, final_result)
# plt.show()

# uruchomienie algorytmu


# wykres poziomicowy

learn_rates = np.array([0.0005, 0.005, 0.05, 0.5])
max_x = 5
max_iter = 100
func = f2

for j in range(len(learn_rates)):
    plt.figure(figsize=(30, 10))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        x = np.arange(-max_x, max_x, 0.5)
        y = np.arange(-max_x, max_x, 0.5)
        X, Y = np.meshgrid(x, y)
        Z = np.empty(X.shape)
        for k in range(X.shape[0]):
            for l in range(X.shape[1]):
                Z[k, l] = func(np.array([X[k, l], Y[k, l]]))

        plt.contour(X, Y, Z, 20)
        minimum, steps = gradient_descent(func, 2, max_x, max_iter, learn_rates[j])
        for m in range(len(steps[:-1])):
             plt.arrow(steps[m][0], steps[m][1], steps[m+1][0] - steps[m][0], steps[m+1][1] - steps[m][1])

    plt.show()

# learn_rates = np.array([0.0005, 0.005, 0.05, 0.5])
# max_iter = 5000
# max_x = 5
# func = f1
# stats = np.zeros((len(learn_rates), 100, 2, 5000))
# x = np.linspace(0, max_iter)
# for j in range(len(learn_rates)):
#     for i in range(100):
#         steps = gradient_descent(func, 1, max_x, max_iter, learn_rates[j])[1]
#         stats[j][i][0] = steps
#         stats[j][i][1] = func(steps)
#
#
#     plt.plot(x, )
# plt.title("Wpływ lr na szybkość zbliżania się do optimum wraz ze wzrostem iteracji")
# plt.xlabel("Liczba iteracji")
# plt.ylabel("Wartość funkcji f(x)")
# plt.grid(b=True)
# plt.show()
