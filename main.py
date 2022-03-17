import numpy as np
import autograd as ag
import math
import matplotlib.pyplot as plt


def f1(x):
    return x ** 4


def grad_f1(x):
    return 4 * x ** 3


def f2(x):
    e = math.e
    return 1.5 - (e ** ((-x[0]) ** 2 - x[1] ** 2)) - 0.5 * (e ** ((-(x[0] - 1)) ** 2 - (x[1] + 2) ** 2))


def grad_f2(x):
    x1 = x[0]
    x2 = x[1]
    return np.array([2 * x1 * np.exp(-x1 ** 2 - x2 ** 2) + (x1 - 1) * np.exp(-(x1 - 1) ** 2 - (x2 + 2) ** 2),
                     2 * x2 * np.exp(-x1 ** 2 - x2 ** 2) + (x2 + 2) * np.exp(-(x1 - 1) ** 2 - (x2 + 2) ** 2)
                     ])


def gradient_descent(function, dimensions, max_iter, learn_rate=0.00007):

    upper_bound = 100
    x = np.random.uniform(-upper_bound, upper_bound, dimensions)
    points_history = np.array([x])
    for _ in range(max_iter):
        diff = grad_f2(x)
        x = x - learn_rate * diff
        x[x < -upper_bound] = -upper_bound
        x[x > upper_bound] = upper_bound
        points_history = np.append(points_history, [x], axis=0)

    optimum = np.array([x, function(x)])
    return optimum, points_history


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
# MAX_X = 100
# PLOT_STEP = 0.1
#
# x_arr = np.arange(-MAX_X, MAX_X, PLOT_STEP)
# y_arr = np.arange(-MAX_X, MAX_X, PLOT_STEP)
# X, Y = np.meshgrid(x_arr, y_arr)
# Z = np.empty(X.shape)
# func = f2
#
# for i in range(X.shape[0]):
#     for j in range(X.shape[1]):
#         Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
#
# plt.contour(X, Y, Z, 20)
#
# # uruchomienie algorytmu
result, steps = gradient_descent(f2, 2, 100)
print(f"Optimum w punkcie: {result[0]} wynosi {result[1]}.")
print("Kroki:", steps)
#
# for i in range(len(steps) - 1):
#     plt.arrow(steps[i], func(steps[i]), steps[i + 1] - steps[i], func(steps[i + 1]) - func(steps[i]), head_width=3,
#               head_length=6, fc='k', ec='k')
#
# plt.show()
#
