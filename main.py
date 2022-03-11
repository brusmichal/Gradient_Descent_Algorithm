import math
import numdifftools as nd
import matplotlib.pyplot as plt
import numpy as np


def f1(x):
    return x ** 4


def grad_f1(x):
    return 4 * x ** 3


def f2(x):
    return 1.5 - math.exp(-x[0] ** 2 - x[1] ** 2) - 0.5 * math.exp(-(x[0] - 1) ** 2 - (x[1] + 2) ** 2)


def grad_f2(x):
    return np.array([[2 * x[0] * math.exp(-x[0] ** 2 - x[1] ** 2) +
                      (x[0] - 1) * math.exp(-(x[0] - 1) ** 2 - (x[1] + 2) ** 2)],
                     [2 * x[1] * math.exp(-x[0] ** 2 - x[1] ** 2) +
                      (x[1] + 2) * math.exp(-(x[0] - 1) ** 2 - (x[1] + 2) ** 2)]
                     ])


def gradient_descent(grad_func, start_point, learn_rate, max_iter, accuracy):
    steps = np.array([start_point])
    x = start_point
    for _ in range(max_iter):
        diff = grad_func(x)
        if abs(x - diff) < accuracy:
            break
        x = x - learn_rate * diff
        steps = np.append(steps, x)
    return steps, x


x_steps, final_result = gradient_descent(grad_f1, 8, 0.007, 100, 0.01)

x_points = np.arange(-10, 10, 0.1)
y_points = f1(x_points)
y_steps = f1(x_steps)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Function f1 with gradient descent steps")
plt.plot(x_points, y_points)
plt.plot(x_steps, y_steps)
print(x_steps, final_result)
plt.show()
